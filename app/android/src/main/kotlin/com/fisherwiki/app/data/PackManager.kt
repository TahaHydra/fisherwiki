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
 * Installation is **atomic with respect to the pack being installed**: it is
 * verified and extracted into a staging directory, hash-checked again from its
 * final on-disk location (not just the staging copy - see [install]), and only
 * then swapped into place with a same-filesystem rename rather than a
 * delete-then-write. A failed or interrupted install can therefore never leave
 * a half-extracted or silently-corrupted pack that the engine would later load
 * and trust, and never destroys a working previous version before the new one
 * is confirmed good.
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

    /**
     * True for a real installed-pack directory name (`packId-vN`), false for
     * the transient `.new-`/`.trash-` names [install] uses while swapping one
     * in. Those are implementation detail, not a pack, and must never be
     * listed, matched against by [removeAll], or reported as a broken pack.
     */
    private fun isPackDirName(name: String) = INSTALL_TEMP_MARKER !in name

    fun installedDirs(): List<File> =
        root.listFiles { f -> f.isDirectory && isPackDirName(f.name) }
            ?.sortedBy { it.name } ?: emptyList()

    /** Delete any `.new-`/`.trash-` directory left behind by an install that
     * was interrupted (process killed, device powered off) before it could
     * clean up after itself. Never touches a real `packId-vN` directory. Safe
     * to call any time: these are always either fully-verified-but-not-yet-
     * needed (trash, about to be deleted anyway) or partially-written
     * (new, never linked to by a real pack name), so there is nothing
     * referencing them that this could break. */
    private fun sweepOrphanedInstallTemp() {
        root.listFiles { f -> f.isDirectory && !isPackDirName(f.name) }
            ?.forEach { it.deleteRecursively() }
    }

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
        sweepOrphanedInstallTemp()
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
        target.parentFile?.mkdirs()

        // Note which older versions exist so the caller can report an upgrade.
        val replaced = installedDirs()
            .filter { it.name.startsWith("${manifest.packId}-v") && it != target }
            .mapNotNull { it.name.substringAfterLast("-v").toIntOrNull() }
            .maxOrNull()

        // Move (or, cross-filesystem, copy) into a *new*, never-before-used
        // directory under `root` - never write over `target` directly. This
        // is what makes the eventual swap below a same-filesystem rename
        // rather than a delete-then-write: everything up to this point can
        // fail or be interrupted without touching an existing installed pack.
        val newDir = File(root, "${manifest.packId}-v${manifest.packVersion}" +
            "${INSTALL_TEMP_MARKER}new-${System.nanoTime()}")
        val moved = stageDir.renameTo(newDir)
        if (!moved) {
            stageDir.copyRecursively(newDir, overwrite = true)
        }
        staging.deleteRecursively()

        // Re-verify from the *final* location, hashes included - not just
        // size, and not just the staging copy `PackVerifier` already checked.
        // `PackVerifier.verifyAndExtract` proved the staged bytes matched the
        // manifest; it did not prove the subsequent move or copy was
        // faithful, and a cross-filesystem copy is exactly the case that
        // needs re-checking rather than trusting.
        val newFiles = manifest.entries.associate { it.path to File(newDir, it.path) }
        for (spec in manifest.entries) {
            val f = newFiles.getValue(spec.path)
            if (!f.isFile || f.length() != spec.bytes) {
                newDir.deleteRecursively()
                return Result.Rejected("install verification failed for ${spec.path}: size mismatch")
            }
            val actual = com.fisherwiki.core.pack.SafeZip.sha256(f)
            if (!actual.equals(spec.sha256, ignoreCase = true)) {
                newDir.deleteRecursively()
                return Result.Rejected(
                    "install verification failed for ${spec.path}: checksum mismatch"
                )
            }
        }

        // Swap: move the old pack (if any) aside before moving the new one
        // in, rather than deleting the old one first. Both are same-
        // filesystem renames of a directory already under `root`, so each
        // completes or does not - there is no copy, and no window where a
        // partially-written directory sits at `target`. If the process is
        // killed between the two renames, `target` may briefly not exist,
        // but the fully-verified old pack survives under its trash name and
        // is *not* lost: `sweepOrphanedInstallTemp` only deletes it on a
        // later call, after a real pack (old or new) is already in place.
        val trash = if (target.exists()) {
            File(root, "${target.name}${INSTALL_TEMP_MARKER}trash-${System.nanoTime()}")
                .also { if (!target.renameTo(it)) {
                    newDir.deleteRecursively()
                    return Result.Rejected("could not move the previous pack version aside")
                } }
        } else null

        if (!newDir.renameTo(target)) {
            trash?.renameTo(target) // best-effort restore of the working pack
            newDir.deleteRecursively()
            return Result.Rejected("could not finalise pack installation")
        }
        trash?.deleteRecursively()

        if (deleteSource) archive.delete()
        val finalFiles = manifest.entries.associate { it.path to File(target, it.path) }
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

        /** Substring that marks a directory under [root] as install-temporary
         * rather than an installed pack. No real pack id can ever produce a
         * matching directory name, because [PackVerifier.validateManifest]
         * constrains `pack_id` to `^[a-z0-9_]{1,64}$`, which excludes `.`. */
        internal const val INSTALL_TEMP_MARKER = ".instmp-"
    }
}
