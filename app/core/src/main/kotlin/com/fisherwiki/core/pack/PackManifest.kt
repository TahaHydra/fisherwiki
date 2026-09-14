package com.fisherwiki.core.pack

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * On-disk description of a FisherWiki offline pack.
 *
 * A pack is the unit of everything the app needs to identify fish in one region
 * without a network: a model, its class mapping, a species database subset, a
 * geographic prior and the attribution data required by the media licences.
 *
 * Packs are **untrusted input**. They may arrive from a CDN, a sideload, or a
 * file the user was sent. Therefore:
 *
 * - the manifest is parsed with a strict, closed schema (unknown keys rejected);
 * - every payload entry carries its own SHA-256 and declared size;
 * - nothing in a pack is executable, and no field is ever interpreted as a path
 *   to load code from;
 * - [formatVersion] is checked before anything else is read, so a future format
 *   fails cleanly instead of being half-parsed.
 *
 * See `docs/OFFLINE_PACK_FORMAT.md` for the full specification.
 */
@Serializable
data class PackManifest(
    /** Format of this manifest. Bumped only for breaking changes. */
    @SerialName("format_version") val formatVersion: Int,

    /** Stable pack identity, e.g. `europe_freshwater`. */
    @SerialName("pack_id") val packId: String,

    /** Monotonic content version for this [packId]. */
    @SerialName("pack_version") val packVersion: Int,

    @SerialName("display_name") val displayName: String,
    val description: String = "",

    /** ISO-8601 UTC build timestamp. */
    @SerialName("built_at") val builtAt: String,

    /** Regions this pack claims to cover, matching `fwdata.regions` ids. */
    val regions: List<RegionSpec> = emptyList(),

    val model: ModelSpec,
    val database: FileSpec,
    val labels: FileSpec,

    /** Optional: per-class geographic occurrence prior. */
    @SerialName("geo_prior") val geoPrior: FileSpec? = null,

    /** Optional: attribution bundle required when media licences demand it. */
    val attributions: FileSpec? = null,

    /** Provenance of the corpus this pack's model was trained on. */
    val corpus: CorpusSpec,

    /**
     * Minimum app engine version able to interpret this pack. The app refuses a
     * pack requiring a newer engine rather than guessing at unknown semantics.
     */
    @SerialName("min_engine_version") val minEngineVersion: Int = 1,
) {
    val entries: List<FileSpec>
        get() = listOfNotNull(model.file, database, labels, geoPrior, attributions)

    companion object {
        /** Format this build writes and can read. */
        const val CURRENT_FORMAT_VERSION = 1

        /** Engine capability level of this build. */
        const val ENGINE_VERSION = 1

        const val MANIFEST_ENTRY = "manifest.json"

        /**
         * Strict JSON: unknown keys are an error, not something to ignore.
         * A pack containing fields we do not understand may be relying on
         * semantics we will not apply, and silently dropping them is how an
         * "identification" ends up meaning something other than intended.
         */
        val json: Json = Json {
            ignoreUnknownKeys = false
            isLenient = false
            explicitNulls = false
        }

        fun parse(text: String): PackManifest = json.decodeFromString(serializer(), text)
    }
}

/** One file carried inside a pack archive. */
@Serializable
data class FileSpec(
    /** Path **inside the archive**. Must be a plain relative name; see [SafeZip]. */
    val path: String,
    /** Lowercase hex SHA-256 of the file's bytes. */
    val sha256: String,
    /** Exact byte length, checked before and after extraction. */
    val bytes: Long,
)

@Serializable
data class ModelSpec(
    val file: FileSpec,
    /** `onnx` today. Present so a future runtime change is explicit. */
    val runtime: String = "onnx",
    /** Square input resolution in pixels. */
    @SerialName("input_size") val inputSize: Int,
    /** Per-channel mean/std used at training time, in RGB order, 0..1 scale. */
    @SerialName("input_mean") val inputMean: List<Float>,
    @SerialName("input_std") val inputStd: List<Float>,
    /** Name of the input tensor, as exported. */
    @SerialName("input_name") val inputName: String = "input",
    /** Name of the logits output tensor. */
    @SerialName("output_name") val outputName: String = "logits",
    /** Optional embedding output, used for metric-learning fallbacks. */
    @SerialName("embedding_name") val embeddingName: String? = null,
    @SerialName("num_classes") val numClasses: Int,
    val architecture: String,
    /** `fp32`, `fp16` or `int8`. */
    val quantization: String = "fp32",
    val calibration: CalibrationSpec,
)

/**
 * Confidence calibration and the open-set rejection rule.
 *
 * Raw softmax over a closed class set is systematically overconfident and has
 * no way to express "this is a fish I was never trained on" - let alone "this
 * is a boot". Both are handled here rather than in UI code, so that every
 * caller gets the same honest numbers.
 */
@Serializable
data class CalibrationSpec(
    /** Temperature for scaling logits. >1 softens an overconfident model. */
    val temperature: Float = 1.0f,
    /** Below this calibrated top-1 probability the result is reported as unknown. */
    @SerialName("unknown_threshold") val unknownThreshold: Float = 0.35f,
    /**
     * Minimum margin between top-1 and top-2 calibrated probabilities. A
     * confident-looking prediction that barely beats its runner-up is a
     * coin flip between two similar species and is reported as uncertain.
     */
    @SerialName("margin_threshold") val marginThreshold: Float = 0.08f,
    /**
     * Maximum normalised predictive entropy tolerated before falling back to a
     * coarser taxonomic rank.
     */
    @SerialName("entropy_threshold") val entropyThreshold: Float = 0.85f,
    /** Per-class thresholds, keyed by class index, for classes needing them. */
    @SerialName("per_class_threshold") val perClassThreshold: Map<String, Float> = emptyMap(),
    /** Measured expected calibration error on the held-out set, for display. */
    @SerialName("expected_calibration_error") val expectedCalibrationError: Float? = null,
)

@Serializable
data class RegionSpec(
    val id: String,
    val name: String,
    /** `freshwater`, `marine`, `brackish` or `any`. */
    val water: String = "any",
    /** `[latMin, latMax, lonMin, lonMax]` boxes in WGS84 degrees. */
    val boxes: List<List<Double>> = emptyList(),
)

/** Where the training data came from and under what licence. */
@Serializable
data class CorpusSpec(
    val name: String,
    /** `production`, `production_sa` or `research_nc`. */
    @SerialName("license_policy") val licensePolicy: String,
    /**
     * False for research-only corpora. The app surfaces this so a build that
     * must not be distributed commercially cannot be mistaken for one that can.
     */
    @SerialName("commercial_safe") val commercialSafe: Boolean,
    @SerialName("image_count") val imageCount: Int,
    @SerialName("class_count") val classCount: Int,
    val sources: List<String> = emptyList(),
    /** SHA-256 of the corpus manifest, tying this model to exact training data. */
    @SerialName("corpus_sha256") val corpusSha256: String? = null,
    /** Git commit of the training code. */
    @SerialName("code_commit") val codeCommit: String? = null,
)
