package com.fisherwiki.core.pack

import java.io.File

/**
 * Validates a pack archive before any of its contents are used.
 *
 * The order of checks is deliberate and each step is cheap relative to the
 * next, so a hostile or corrupt file is rejected as early as possible:
 *
 * 1. read and parse `manifest.json` only (bounded, in memory);
 * 2. check [PackManifest.formatVersion] and [PackManifest.minEngineVersion];
 * 3. sanity-check the manifest's own declarations (sizes, hash shape, class
 *    count, normalisation vectors);
 * 4. extract, with the manifest's entry names as an allow-list;
 * 5. verify every extracted file's size and SHA-256 against the manifest.
 *
 * Only after step 5 does the caller get an [InstalledPack].
 */
object PackVerifier {

    sealed class Result {
        data class Ok(val pack: InstalledPack) : Result()
        data class Rejected(val reason: String, val cause: Throwable? = null) : Result()
    }

    private val HEX64 = Regex("^[0-9a-f]{64}$")

    /** Parse and validate the manifest without extracting the payload. */
    fun readManifest(archive: File): PackManifest {
        val bytes = SafeZip.readEntry(archive, PackManifest.MANIFEST_ENTRY)
        val manifest = PackManifest.parse(bytes.decodeToString())
        validateManifest(manifest)
        return manifest
    }

    /**
     * Structural checks on the manifest's self-consistency.
     *
     * These are cheap and catch both corruption and a class of malicious pack
     * that would otherwise be caught only after we had extracted 2 GB.
     */
    fun validateManifest(m: PackManifest) {
        require(m.formatVersion == PackManifest.CURRENT_FORMAT_VERSION) {
            "unsupported pack format ${m.formatVersion}, " +
                "this build reads ${PackManifest.CURRENT_FORMAT_VERSION}"
        }
        require(m.minEngineVersion <= PackManifest.ENGINE_VERSION) {
            "pack requires engine version ${m.minEngineVersion}, " +
                "this build is ${PackManifest.ENGINE_VERSION}"
        }
        require(m.packId.isNotBlank() && m.packId.matches(Regex("^[a-z0-9_]{1,64}$"))) {
            "invalid pack_id '${m.packId}'"
        }
        require(m.packVersion >= 1) { "invalid pack_version ${m.packVersion}" }

        require(m.model.runtime == "onnx") {
            "unsupported model runtime '${m.model.runtime}'"
        }
        require(m.model.inputSize in 64..1024) {
            "implausible model input size ${m.model.inputSize}"
        }
        require(m.model.numClasses in 1..100_000) {
            "implausible class count ${m.model.numClasses}"
        }
        require(m.model.inputMean.size == 3 && m.model.inputStd.size == 3) {
            "input_mean/input_std must have 3 channels"
        }
        require(m.model.inputStd.all { it > 1e-6f }) {
            "input_std contains a zero, which would divide by zero at preprocess"
        }
        require(m.model.quantization in setOf("fp32", "fp16", "int8")) {
            "unknown quantization '${m.model.quantization}'"
        }

        val cal = m.model.calibration
        require(cal.temperature > 0f && cal.temperature < 100f) {
            "implausible calibration temperature ${cal.temperature}"
        }
        require(cal.unknownThreshold in 0f..1f) {
            "unknown_threshold must be a probability"
        }
        require(cal.marginThreshold in 0f..1f) { "margin_threshold must be a probability" }
        require(cal.entropyThreshold in 0f..1f) { "entropy_threshold must be normalised" }

        for (e in m.entries) {
            require(e.sha256.matches(HEX64)) { "entry ${e.path} has a malformed sha256" }
            require(e.bytes in 1..SafeZip.MAX_ENTRY_BYTES) {
                "entry ${e.path} declares an implausible size ${e.bytes}"
            }
            // Throws if the name is unsafe.
            SafeZip.sanitizeEntryName(e.path)
        }
        val names = m.entries.map { it.path }
        require(names.size == names.toSet().size) { "duplicate entry paths in manifest" }

        require(m.corpus.imageCount >= 0 && m.corpus.classCount >= 0) {
            "negative corpus counts"
        }
        require(m.corpus.classCount <= m.model.numClasses) {
            "corpus declares ${m.corpus.classCount} classes but the model has " +
                "${m.model.numClasses}; the label mapping cannot be complete"
        }
    }

    /**
     * Fully verify and unpack [archive] into [destDir].
     *
     * [destDir] is expected to be empty or to be overwritten; the caller is
     * responsible for installing into a staging directory and renaming, so that
     * a failed verification never leaves a half-installed pack in place.
     */
    fun verifyAndExtract(archive: File, destDir: File): Result {
        val manifest = try {
            readManifest(archive)
        } catch (t: Throwable) {
            return Result.Rejected("manifest rejected: ${t.message}", t)
        }

        val allowed = manifest.entries.map { it.path }.toSet() + PackManifest.MANIFEST_ENTRY
        val extracted = try {
            SafeZip.extract(archive, destDir, allowedNames = allowed)
        } catch (t: Throwable) {
            return Result.Rejected("extraction rejected: ${t.message}", t)
        }

        for (spec in manifest.entries) {
            val f = extracted[spec.path]
                ?: return Result.Rejected("manifest lists ${spec.path} but the archive lacks it")
            if (f.length() != spec.bytes) {
                return Result.Rejected(
                    "${spec.path}: expected ${spec.bytes} bytes, extracted ${f.length()}"
                )
            }
            val actual = SafeZip.sha256(f)
            if (!actual.equals(spec.sha256, ignoreCase = true)) {
                return Result.Rejected(
                    "${spec.path}: sha256 mismatch (expected ${spec.sha256}, got $actual)"
                )
            }
        }

        return Result.Ok(
            InstalledPack(
                manifest = manifest,
                root = destDir,
                files = manifest.entries.associate { it.path to extracted.getValue(it.path) },
            )
        )
    }
}

/** A pack whose integrity has been fully verified and whose files are on disk. */
data class InstalledPack(
    val manifest: PackManifest,
    val root: File,
    val files: Map<String, File>,
) {
    val id: String get() = manifest.packId
    val version: Int get() = manifest.packVersion

    val modelFile: File get() = files.getValue(manifest.model.file.path)
    val databaseFile: File get() = files.getValue(manifest.database.path)
    val labelsFile: File get() = files.getValue(manifest.labels.path)
    val geoPriorFile: File? get() = manifest.geoPrior?.let { files[it.path] }

    /**
     * True when this pack's model was trained only on media that permits
     * commercial redistribution. Surfaced in the UI so a research build is
     * never mistaken for a shippable one.
     */
    val commercialSafe: Boolean get() = manifest.corpus.commercialSafe
}
