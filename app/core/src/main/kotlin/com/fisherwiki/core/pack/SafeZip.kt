package com.fisherwiki.core.pack

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.security.MessageDigest
import java.util.zip.ZipFile

/**
 * Hardened ZIP extraction for untrusted pack archives.
 *
 * Every guard here corresponds to a real, published attack on archive handling.
 * None of them are theoretical, and none are enforced by [ZipFile] itself:
 *
 * - **Path traversal ("Zip Slip").** An entry named `../../databases/app.db`
 *   escapes the destination directory. We reject any entry whose name is
 *   absolute, contains a `..` segment, contains a backslash or a drive letter,
 *   or whose resolved canonical path is not inside the destination.
 * - **Zip bombs.** A 40 KB archive can expand to many GB. Both the per-entry
 *   uncompressed size and the total extracted size are capped, and the caps are
 *   enforced *while streaming*, not by trusting the declared size in the header.
 * - **Absurd entry counts.** Capped, so a million-entry archive cannot exhaust
 *   file handles or inodes.
 * - **Symlinks.** The ZIP format can encode a unix symlink via the external
 *   attributes field. [java.util.zip.ZipFile] does not expose that field and
 *   this extractor never calls any link-creating API: an entry marked as a
 *   symlink is written as an ordinary file whose contents are the link target
 *   string, which is inert. We do not attempt to detect or honour such entries,
 *   because honouring them is the only way they become dangerous.
 * - **Declared-vs-actual mismatch.** After extraction the byte count and
 *   SHA-256 are compared against the manifest by [PackVerifier]. A pack whose
 *   payload differs from what it claims is rejected even though extraction
 *   itself succeeded.
 * - **Allow-listed entry names.** [extract] optionally accepts the exact set of
 *   names the manifest declares and rejects anything else, so a pack cannot
 *   smuggle extra files past verification by simply not listing them.
 *
 * Nothing extracted here is ever executed, loaded as a library, or used as a
 * class path. Packs carry data only.
 */
object SafeZip {

    /** Hard ceiling on one extracted entry (models are the largest, ~50 MB). */
    const val MAX_ENTRY_BYTES: Long = 512L * 1024 * 1024

    /** Hard ceiling on a whole pack once extracted. */
    const val MAX_TOTAL_BYTES: Long = 2L * 1024 * 1024 * 1024

    /** Hard ceiling on entry count. A pack has a handful of files. */
    const val MAX_ENTRIES: Int = 256

    /** Reject archives whose declared compression ratio is implausible. */
    const val MAX_COMPRESSION_RATIO: Double = 200.0

    class UnsafeArchiveException(message: String) : IOException(message)

    /**
     * Validate an entry name without touching the filesystem.
     *
     * Returns the normalised relative name, or throws. Kept separate from
     * extraction so it can be unit-tested exhaustively against hostile inputs.
     */
    fun sanitizeEntryName(raw: String): String {
        if (raw.isEmpty()) throw UnsafeArchiveException("empty entry name")
        if (raw.length > 255) throw UnsafeArchiveException("entry name too long")

        // Backslashes are a directory separator on Windows; an archive built to
        // exploit that will not contain them on the POSIX-legal path.
        if (raw.contains('\\')) {
            throw UnsafeArchiveException("entry name contains a backslash: $raw")
        }
        if (raw.startsWith("/")) {
            throw UnsafeArchiveException("absolute entry name: $raw")
        }
        // Windows drive-letter or UNC forms.
        if (raw.length >= 2 && raw[1] == ':') {
            throw UnsafeArchiveException("drive-qualified entry name: $raw")
        }
        if (raw.contains('\u0000')) {
            throw UnsafeArchiveException("entry name contains NUL: $raw")
        }

        // A trailing '/' marks a directory entry and is the only legal empty
        // segment; anything else means the name was not normalised, and we
        // require packs to be built cleanly rather than normalising for them.
        val body = raw.removeSuffix("/")
        for (seg in body.split('/')) {
            when {
                seg == ".." ->
                    throw UnsafeArchiveException("entry name escapes the archive root: $raw")
                seg == "." ->
                    throw UnsafeArchiveException("entry name has a '.' segment: $raw")
                seg.isEmpty() ->
                    throw UnsafeArchiveException("entry name has an empty segment: $raw")
            }
        }
        return raw
    }


    /**
     * Extract [zipFile] into [destDir], enforcing every guard above.
     *
     * @return map of entry name to extracted [File].
     */
    fun extract(
        zipFile: File,
        destDir: File,
        allowedNames: Set<String>? = null,
    ): Map<String, File> {
        if (!destDir.exists() && !destDir.mkdirs()) {
            throw IOException("cannot create destination ${destDir.path}")
        }
        val destRoot = destDir.canonicalFile
        val out = LinkedHashMap<String, File>()
        var totalWritten = 0L

        ZipFile(zipFile).use { zf ->
            val entries = zf.entries().toList()
            if (entries.size > MAX_ENTRIES) {
                throw UnsafeArchiveException(
                    "archive has ${entries.size} entries, limit is $MAX_ENTRIES"
                )
            }
            for (entry in entries) {
                val name = sanitizeEntryName(entry.name)
                if (allowedNames != null && !entry.isDirectory && name !in allowedNames) {
                    throw UnsafeArchiveException("unexpected entry in pack: $name")
                }

                val target = File(destRoot, name).canonicalFile
                if (!target.path.startsWith(destRoot.path + File.separator) &&
                    target.path != destRoot.path
                ) {
                    throw UnsafeArchiveException("entry resolves outside destination: $name")
                }

                if (entry.isDirectory) {
                    if (!target.isDirectory && !target.mkdirs()) {
                        throw IOException("cannot create directory ${target.path}")
                    }
                    continue
                }

                // Reject implausible declared ratios before spending any I/O.
                val declared = entry.size
                val compressed = entry.compressedSize
                if (declared > MAX_ENTRY_BYTES) {
                    throw UnsafeArchiveException(
                        "entry $name declares ${declared} bytes, limit is $MAX_ENTRY_BYTES"
                    )
                }
                if (declared > 0 && compressed > 0 &&
                    declared.toDouble() / compressed.toDouble() > MAX_COMPRESSION_RATIO
                ) {
                    throw UnsafeArchiveException(
                        "entry $name has compression ratio " +
                            "${declared / compressed}, limit is $MAX_COMPRESSION_RATIO"
                    )
                }

                target.parentFile?.mkdirs()
                val written = zf.getInputStream(entry).use { input ->
                    target.outputStream().use { output ->
                        copyBounded(
                            input,
                            output,
                            entryLimit = MAX_ENTRY_BYTES,
                            totalRemaining = MAX_TOTAL_BYTES - totalWritten,
                            name = name,
                        )
                    }
                }
                totalWritten += written
                out[name] = target
            }
        }
        return out
    }

    /**
     * Copy while enforcing limits on the *actual* stream, not the header.
     * A crafted archive can declare a small uncompressed size and then emit
     * gigabytes, so the declared value is only ever used as an early reject.
     */
    private fun copyBounded(
        input: InputStream,
        output: OutputStream,
        entryLimit: Long,
        totalRemaining: Long,
        name: String,
    ): Long {
        val buf = ByteArray(64 * 1024)
        var written = 0L
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            written += n
            if (written > entryLimit) {
                throw UnsafeArchiveException("entry $name exceeded $entryLimit bytes while extracting")
            }
            if (written > totalRemaining) {
                throw UnsafeArchiveException("archive exceeded $MAX_TOTAL_BYTES total bytes")
            }
            output.write(buf, 0, n)
        }
        return written
    }

    /** Read a single entry into memory, bounded. Used for `manifest.json`. */
    fun readEntry(zipFile: File, entryName: String, maxBytes: Long = 4L * 1024 * 1024): ByteArray {
        val safe = sanitizeEntryName(entryName)
        ZipFile(zipFile).use { zf ->
            val entry = zf.getEntry(safe)
                ?: throw UnsafeArchiveException("missing entry $safe")
            if (entry.size > maxBytes) {
                throw UnsafeArchiveException("$safe declares ${entry.size} bytes, limit $maxBytes")
            }
            zf.getInputStream(entry).use { input ->
                val buf = java.io.ByteArrayOutputStream()
                copyBounded(input, buf, maxBytes, maxBytes, safe)
                return buf.toByteArray()
            }
        }
    }

    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().toHex()
    }

    fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

    private fun ByteArray.toHex(): String {
        val sb = StringBuilder(size * 2)
        for (b in this) {
            val v = b.toInt() and 0xFF
            sb.append("0123456789abcdef"[v ushr 4])
            sb.append("0123456789abcdef"[v and 0x0F])
        }
        return sb.toString()
    }
}
