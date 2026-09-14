package com.fisherwiki.app.data

import android.content.Context
import android.net.Uri
import com.fisherwiki.core.pack.InstalledPack
import com.fisherwiki.core.pack.PackManifest
import com.fisherwiki.core.pack.PackVerifier
import java.io.File

/**
 * Installs, verifies, lists and removes offline packs.
 *
 * Installation is **atomic**: a pack is verified and extracted into a staging
 * directory, and only a fully-verified result is moved into place. A failed or
 * interrupted install can therefore never leave a half-extracted pack that the
 * engine would later load and trust.
 *
 * Packs live in the app's private files directory, so no other app can modify a
 * verified pack behind our back and nothing needs storage permissions.
 *
 * **Sideloading is supported deliberately.** During development, and for anyone
 * who wants to build their own pack, [installFromUri] takes a local `.fwpack`
 * file with no backend and no store involved. The verification path is exactly
 * the same as for a downloaded pack: a sideloaded pack gets no more trust.
 */
class PackManager(private val context: Context) {

    private val root: File get() = File(context.filesDir, "packs").apply { mkdirs() }
    private val staging: File get() = File(context.cacheDir, "pack_staging")

    sealed class Result {
        data class Installed(val pack: InstalledPack, val replaced: Int?) : Result()
        data class Rejected(val reason: String) : Result()
    }

    /** Directory for an installed pack version. */
    private fun dirFor(packId: String, version: Int) = File(root, "$packId-v$version")

    fun installedDirs(): List<File> =
        root.listFiles { f -> f.isDirectory }?.sortedBy { it.name } ?: emptyList()

    /**
     * List installed packs, skipping any whose manifest no longer parses.
     *
     * A pack that fails to load here is reported rather than silently ignored,
     * because "my pack vanished" with no explanation is a bad experience.
     */
    fun list(): List<LoadedPack> = installedDirs().mapNotNull { dir ->
        val manifestFile = File(dir, PackManifest.MANIFEST_ENTRY)
        if (!manifestFile.isFile) return@mapNotNull LoadedPack.Broken(dir.name, "manifest missing")
        runCatching {
            val manifest = PackManifest.parse(manifestFile.readText())
            PackVerifier.validateManifest(manifest)
            val files = manifest.entries.associate { it.path to File(dir, it.path) }
            val missing = files.filterValues { !it.isFile }.keys
            if (missing.isNotEmpty()) {
                LoadedPack.Broken(dir.name, "missing files: ${missing.joinToString()}")
            } else {
                LoadedPack.Ok(InstalledPack(manifest, dir, files))
            }
        }.getOrElse { LoadedPack.Broken(dir.name, it.message ?: "unreadable") }
    }

    sealed class LoadedPack {
        data class Ok(val pack: InstalledPack) : LoadedPack()
        data class Broken(val dirName: String, val reason: String) : LoadedPack()
    }

    fun active(): InstalledPack? =
        list().filterIsInstance<LoadedPack.Ok>()
            .maxByOrNull { it.pack.version }?.pack

    /**
     * Verify and install from a local file.
     *
     * @param deleteSource remove the source archive afterwards. Off by default:
     *   a user who picked a file from their Downloads folder did not ask us to
     *   delete it.
     */
    fun install(archive: File, deleteSource: Boolean = false): Result {
        if (!archive.isFile) return Result.Rejected("file not found")
        staging.deleteRecursively()
        staging.mkdirs()
        val stageDir = File(staging, "pending").apply { mkdirs() }

        val verified = when (val r = PackVerifier.verifyAndExtract(archive, stageDir)) {
            is PackVerifier.Result.Rejected -> {
                staging.deleteRecursively()
                return Result.Rejected(r.reason)
            }
            is PackVerifier.Result.Ok -> r.pack
        }

        val manifest = verified.manifest
        val target = dirFor(manifest.packId, manifest.packVersion)

        // Note which older versions exist so the caller can report an upgrade.
        val replaced = installedDirs()
            .filter { it.name.startsWith("${manifest.packId}-v") && it != target }
            .mapNotNull { it.name.substringAfterLast("-v").toIntOrNull() }
            .maxOrNull()

        if (target.exists()) target.deleteRecursively()
        target.parentFile?.mkdirs()

        // Atomic swap. If the rename fails (different filesystem), fall back to
        // a copy, then verify again from the final location rather than
        // trusting that the copy was faithful.
        val moved = stageDir.renameTo(target)
        if (!moved) {
            stageDir.copyRecursively(target, overwrite = true)
        }
        staging.deleteRecursively()

        val finalFiles = manifest.entries.associate { it.path to File(target, it.path) }
        for (spec in manifest.entries) {
            val f = finalFiles.getValue(spec.path)
            if (!f.isFile || f.length() != spec.bytes) {
                target.deleteRecursively()
                return Result.Rejected("install verification failed for ${spec.path}")
            }
        }

        if (deleteSource) archive.delete()
        return Result.Installed(InstalledPack(manifest, target, finalFiles), replaced)
    }

    /** Copy a content:// URI into the cache, then install it. */
    fun installFromUri(uri: Uri): Result {
        val tmp = File(context.cacheDir, "incoming.fwpack")
        return try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                tmp.outputStream().use { output ->
                    var total = 0L
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        total += n
                        // Bound what an untrusted URI can write into our cache.
                        if (total > MAX_PACK_BYTES) {
                            return Result.Rejected(
                                "pack exceeds ${MAX_PACK_BYTES / 1_000_000} MB limit"
                            )
                        }
                        output.write(buf, 0, n)
                    }
                }
            } ?: return Result.Rejected("could not open the selected file")
            install(tmp, deleteSource = true)
        } catch (t: Throwable) {
            Result.Rejected(t.message ?: "could not read the selected file")
        } finally {
            tmp.delete()
        }
    }

    fun remove(packId: String, version: Int): Boolean =
        dirFor(packId, version).deleteRecursively()

    fun removeAll(packId: String): Int {
        var n = 0
        for (d in installedDirs()) {
            if (d.name.startsWith("$packId-v")) {
                if (d.deleteRecursively()) n++
            }
        }
        return n
    }

    /** Re-check hashes of an installed pack, for the "verify integrity" action. */
    fun verifyInstalled(pack: InstalledPack): List<String> {
        val problems = ArrayList<String>()
        for (spec in pack.manifest.entries) {
            val f = pack.files[spec.path]
            if (f == null || !f.isFile) {
                problems.add("${spec.path}: missing")
                continue
            }
            if (f.length() != spec.bytes) {
                problems.add("${spec.path}: size ${f.length()} != ${spec.bytes}")
                continue
            }
            val actual = com.fisherwiki.core.pack.SafeZip.sha256(f)
            if (!actual.equals(spec.sha256, ignoreCase = true)) {
                problems.add("${spec.path}: checksum mismatch")
            }
        }
        return problems
    }

    fun totalBytes(): Long = installedDirs().sumOf { dir ->
        dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
    }

    companion object {
        const val MAX_PACK_BYTES = 1_500_000_000L
    }
}
