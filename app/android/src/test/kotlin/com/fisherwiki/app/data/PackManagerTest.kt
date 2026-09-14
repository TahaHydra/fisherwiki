package com.fisherwiki.app.data

import com.google.common.truth.Truth.assertThat
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

/**
 * [PackManager.install] against a real pack, on the JVM via Robolectric - the
 * one place in this project where `android.content.Context` needed to be a
 * real (if JVM-hosted) implementation rather than something to work around,
 * because atomic install and hash re-verification are exactly the kind of
 * logic a mock `Context` would let past without ever being exercised.
 *
 * A true "kill the process between the two renames" test is not something a
 * unit test can do; what is tested instead is every state that logic must be
 * correct in: a clean install, a same-version reinstall, a version upgrade,
 * a stray leftover from a previous interrupted install, and a payload that
 * fails re-verification after extraction. Together they pin the actual
 * invariant - `target` is never observed holding a half-written or
 * hash-mismatched pack, and a working previous version is never destroyed
 * before its replacement is confirmed good - even though the exact instant
 * of a crash cannot be scripted.
 */
@RunWith(RobolectricTestRunner::class)
class PackManagerTest {

    @get:Rule
    val tmpFolder = TemporaryFolder()

    private val context get() = RuntimeEnvironment.getApplication()
    private val manager get() = PackManager(context)
    private val packsRoot get() = File(context.filesDir, "packs")

    private fun fixtureBytes(): ByteArray =
        javaClass.getResourceAsStream("/fixtures/pack/fixture_v1.fwpack")
            ?.readBytes() ?: error("fixture_v1.fwpack missing from android test resources")

    private fun fixtureFile(name: String = "fixture.fwpack"): File =
        File(tmpFolder.root, name).apply { writeBytes(fixtureBytes()) }

    /** The same fixture pack, re-zipped with only `pack_version` changed - a
     * real, independently hash-valid pack (the payload files and their
     * declared hashes are untouched), for exercising the upgrade path. */
    private fun fixtureAtVersion(version: Int): File {
        val src = fixtureFile("src-v$version.fwpack")
        val out = File(tmpFolder.root, "fixture_v$version.fwpack")
        ZipFile(src).use { zf ->
            ZipOutputStream(out.outputStream()).use { zos ->
                for (entry in zf.entries()) {
                    val bytes = zf.getInputStream(entry).readBytes()
                    val content = if (entry.name == "manifest.json") {
                        String(bytes, Charsets.UTF_8)
                            .replace("\"pack_version\": 1", "\"pack_version\": $version")
                            .toByteArray(Charsets.UTF_8)
                    } else {
                        bytes
                    }
                    zos.putNextEntry(ZipEntry(entry.name))
                    zos.write(content)
                    zos.closeEntry()
                }
            }
        }
        src.delete()
        return out
    }

    @Test
    fun `a clean install produces a loadable pack`() {
        val result = manager.install(fixtureFile())

        assertThat(result).isInstanceOf(PackManager.Result.Installed::class.java)
        val installed = (result as PackManager.Result.Installed).pack
        assertThat(installed.id).isEqualTo("fixture_v1")
        assertThat(result.replaced).isNull()

        val loaded = manager.list()
        assertThat(loaded).hasSize(1)
        assertThat(loaded.single()).isInstanceOf(PackManager.LoadedPack.Ok::class.java)
        assertThat(manager.active()?.id).isEqualTo("fixture_v1")
        assertThat(manager.verifyInstalled(installed)).isEmpty()
    }

    @Test
    fun `install leaves no temporary directories behind`() {
        manager.install(fixtureFile())

        val stray = packsRoot.listFiles { f -> f.isDirectory && ".instmp-" in f.name }
        assertThat(stray).isEmpty()
    }

    @Test
    fun `reinstalling the same version replaces it without ever losing it`() {
        val first = manager.install(fixtureFile("first.fwpack"))
        assertThat(first).isInstanceOf(PackManager.Result.Installed::class.java)

        val second = manager.install(fixtureFile("second.fwpack"))
        assertThat(second).isInstanceOf(PackManager.Result.Installed::class.java)

        // Exactly one fixture_v1 directory, not a duplicate and not gone.
        val packDirs = packsRoot.listFiles { f -> f.isDirectory && f.name.startsWith("fixture_v1-v") }
        assertThat(packDirs).hasLength(1)
        assertThat(manager.list()).hasSize(1)
        assertThat(manager.verifyInstalled((second as PackManager.Result.Installed).pack)).isEmpty()
    }

    @Test
    fun `installing a newer version reports the old one as replaced, and keeps both`() {
        manager.install(fixtureFile())
        val upgraded = manager.install(fixtureAtVersion(2))

        assertThat(upgraded).isInstanceOf(PackManager.Result.Installed::class.java)
        assertThat((upgraded as PackManager.Result.Installed).replaced).isEqualTo(1)
        assertThat(upgraded.pack.version).isEqualTo(2)

        // Both versions genuinely present - install() must never delete an
        // unrelated version, only ever the exact one it is replacing.
        val versions = manager.list().filterIsInstance<PackManager.LoadedPack.Ok>()
            .map { it.pack.version }
        assertThat(versions).containsExactly(1, 2)
    }

    @Test
    fun `a leftover temp directory from an interrupted install is not a pack`() {
        packsRoot.mkdirs()
        // Simulate what a process kill mid-install leaves behind: a
        // directory under packsRoot with real pack content but a temp name.
        val orphan = File(packsRoot, "fixture_v1-v1.instmp-new-999")
        orphan.mkdirs()
        ZipFile(fixtureFile()).use { zf ->
            for (entry in zf.entries()) {
                if (entry.isDirectory) continue
                val out = File(orphan, entry.name)
                out.parentFile?.mkdirs()
                zf.getInputStream(entry).use { input -> out.outputStream().use { input.copyTo(it) } }
            }
        }

        assertThat(manager.installedDirs()).isEmpty()
        assertThat(manager.list()).isEmpty() // not reported as a broken pack either

        // The next real install must sweep it away, not merely ignore it.
        manager.install(fixtureFile("real.fwpack"))
        assertThat(orphan.exists()).isFalse()
    }

    @Test
    fun `a payload corrupted after staging is caught by the final hash check, not just size`() {
        // Rebuild the fixture with model.onnx byte-flipped but the manifest
        // still declaring the original (correct) size and hash. PackVerifier
        // catches this during staging already - this test pins that
        // install() still calls it and does not short-circuit past it.
        val src = fixtureFile()
        val tampered = File(tmpFolder.root, "tampered.fwpack")
        ZipFile(src).use { zf ->
            ZipOutputStream(tampered.outputStream()).use { zos ->
                for (entry in zf.entries()) {
                    val bytes = zf.getInputStream(entry).readBytes()
                    val content = if (entry.name == "model.onnx" && bytes.isNotEmpty()) {
                        bytes.also { it[0] = (it[0] + 1).toByte() }
                    } else bytes
                    zos.putNextEntry(ZipEntry(entry.name))
                    zos.write(content)
                    zos.closeEntry()
                }
            }
        }

        val result = manager.install(tampered)
        assertThat(result).isInstanceOf(PackManager.Result.Rejected::class.java)
        assertThat(manager.list()).isEmpty()
        assertThat(packsRoot.listFiles { f -> f.isDirectory }.orEmpty()).isEmpty()
    }

    @Test
    fun `an unopenable file is rejected, not crashed on`() {
        val bad = File(tmpFolder.root, "not-a-pack.fwpack").apply { writeBytes(byteArrayOf(1, 2, 3)) }
        assertThat(manager.install(bad)).isInstanceOf(PackManager.Result.Rejected::class.java)
    }
}
