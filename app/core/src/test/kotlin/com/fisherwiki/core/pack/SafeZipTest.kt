package com.fisherwiki.core.pack

import com.google.common.truth.Truth.assertThat
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

/**
 * Packs are untrusted input, so these tests are adversarial by design: each one
 * builds an archive that a hostile pack author would produce and asserts we
 * refuse it.
 */
class SafeZipTest {

    // ---------------------------------------------------------------- names

    @Test
    fun `accepts ordinary entry names`() {
        for (name in listOf("manifest.json", "model.onnx", "data/species.sqlite", "a/b/c.bin")) {
            assertThat(SafeZip.sanitizeEntryName(name)).isEqualTo(name)
        }
    }

    @Test
    fun `rejects parent directory traversal`() {
        for (name in listOf(
            "../evil.so",
            "../../etc/passwd",
            "a/../../b",
            "data/../../../escape.txt",
        )) {
            assertThrows<SafeZip.UnsafeArchiveException>("should reject $name") {
                SafeZip.sanitizeEntryName(name)
            }
        }
    }

    @Test
    fun `rejects absolute and drive qualified names`() {
        for (name in listOf("/etc/passwd", "C:/windows/system32/evil.dll", "D:data.bin")) {
            assertThrows<SafeZip.UnsafeArchiveException> { SafeZip.sanitizeEntryName(name) }
        }
    }

    @Test
    fun `rejects backslash separators`() {
        // On Windows a backslash is a separator, so "..\\..\\x" traverses even
        // though it contains no forward-slash ".." segment.
        assertThrows<SafeZip.UnsafeArchiveException> {
            SafeZip.sanitizeEntryName("..\\..\\windows\\evil.dll")
        }
        assertThrows<SafeZip.UnsafeArchiveException> {
            SafeZip.sanitizeEntryName("data\\species.sqlite")
        }
    }

    @Test
    fun `rejects empty and overlong names`() {
        assertThrows<SafeZip.UnsafeArchiveException> { SafeZip.sanitizeEntryName("") }
        assertThrows<SafeZip.UnsafeArchiveException> {
            SafeZip.sanitizeEntryName("x".repeat(300))
        }
    }

    @Test
    fun `rejects NUL in name`() {
        assertThrows<SafeZip.UnsafeArchiveException> {
            SafeZip.sanitizeEntryName("model\u0000.onnx")
        }
    }

    @Test
    fun `allows a trailing slash for directory entries`() {
        assertThat(SafeZip.sanitizeEntryName("data/")).isEqualTo("data/")
    }

    // ------------------------------------------------------------ extraction

    private fun zipOf(tmp: File, entries: List<Pair<String, ByteArray>>): File {
        val f = File(tmp, "test-${entries.hashCode()}.zip")
        ZipOutputStream(f.outputStream()).use { zos ->
            for ((name, bytes) in entries) {
                zos.putNextEntry(ZipEntry(name))
                zos.write(bytes)
                zos.closeEntry()
            }
        }
        return f
    }

    @Test
    fun `extracts a well formed archive`(@TempDir tmp: File) {
        val zip = zipOf(
            tmp,
            listOf(
                "manifest.json" to """{"x":1}""".toByteArray(),
                "model.onnx" to ByteArray(1024) { it.toByte() },
            ),
        )
        val dest = File(tmp, "out")
        val files = SafeZip.extract(zip, dest)

        assertThat(files.keys).containsExactly("manifest.json", "model.onnx")
        assertThat(files.getValue("model.onnx").length()).isEqualTo(1024)
    }

    @Test
    fun `refuses to extract a traversal entry`(@TempDir tmp: File) {
        // ZipOutputStream happily writes this name; the defence must be ours.
        val zip = zipOf(tmp, listOf("../escaped.txt" to "pwned".toByteArray()))
        val dest = File(tmp, "out")

        assertThrows<SafeZip.UnsafeArchiveException> { SafeZip.extract(zip, dest) }
        assertThat(File(tmp, "escaped.txt").exists()).isFalse()
    }

    @Test
    fun `enforces the entry allow list`(@TempDir tmp: File) {
        val zip = zipOf(
            tmp,
            listOf(
                "manifest.json" to "{}".toByteArray(),
                "stowaway.bin" to ByteArray(16),
            ),
        )
        val dest = File(tmp, "out")

        // A pack cannot smuggle an unlisted file past verification by simply
        // omitting it from the manifest.
        val ex = assertThrows<SafeZip.UnsafeArchiveException> {
            SafeZip.extract(zip, dest, allowedNames = setOf("manifest.json"))
        }
        assertThat(ex.message).contains("stowaway.bin")
    }

    @Test
    fun `rejects an archive with too many entries`(@TempDir tmp: File) {
        val many = (0..SafeZip.MAX_ENTRIES + 10).map { "f$it.bin" to ByteArray(1) }
        val zip = zipOf(tmp, many)
        assertThrows<SafeZip.UnsafeArchiveException> {
            SafeZip.extract(zip, File(tmp, "out"))
        }
    }

    @Test
    fun `rejects a zip bomb by compression ratio`(@TempDir tmp: File) {
        // 8 MB of zeros compresses to a few KB: ratio far above the cap.
        val zip = zipOf(tmp, listOf("bomb.bin" to ByteArray(8 * 1024 * 1024)))
        val ex = assertThrows<SafeZip.UnsafeArchiveException> {
            SafeZip.extract(zip, File(tmp, "out"))
        }
        assertThat(ex.message).contains("compression ratio")
    }

    // ---------------------------------------------------------------- hashes

    @Test
    fun `sha256 of known input`(@TempDir tmp: File) {
        // Well-known vector: sha256("abc")
        assertThat(SafeZip.sha256("abc".toByteArray())).isEqualTo(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
    }

    @Test
    fun `sha256 of file matches sha256 of its bytes`(@TempDir tmp: File) {
        val bytes = ByteArray(5000) { (it * 31).toByte() }
        val f = File(tmp, "blob.bin").also { it.writeBytes(bytes) }
        assertThat(SafeZip.sha256(f)).isEqualTo(SafeZip.sha256(bytes))
    }

    @Test
    fun `readEntry refuses an oversized entry`(@TempDir tmp: File) {
        val zip = zipOf(tmp, listOf("manifest.json" to ByteArray(1024)))
        assertThrows<SafeZip.UnsafeArchiveException> {
            SafeZip.readEntry(zip, "manifest.json", maxBytes = 100)
        }
    }

    @Test
    fun `readEntry returns exact bytes`(@TempDir tmp: File) {
        val payload = """{"format_version":1}""".toByteArray()
        val zip = zipOf(tmp, listOf("manifest.json" to payload))
        assertThat(SafeZip.readEntry(zip, "manifest.json")).isEqualTo(payload)
    }
}
