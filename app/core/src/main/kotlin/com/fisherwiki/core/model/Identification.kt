package com.fisherwiki.core.model

/**
 * Domain types for an identification result.
 *
 * The product principle these encode: **the dangerous failure is a confident
 * wrong answer, not an admitted unknown.** So the result type makes uncertainty
 * a first-class outcome rather than a low number attached to a species name.
 * A caller cannot accidentally render "European Perch" for a result the engine
 * considers unidentifiable, because in that case there is no species to render.
 */

/** Taxonomic rank at which an identification is being asserted. */
enum class Rank {
    SPECIES,
    GENUS,
    FAMILY,
    ORDER,
    CLASS;

    val displayName: String
        get() = name.lowercase().replaceFirstChar { it.uppercase() }
}

/**
 * A canonical taxon. [id] is the `fw_taxon_id` minted by the dataset pipeline
 * and is stable across model retrains and taxonomy revisions.
 */
data class Taxon(
    val id: Long,
    val scientificName: String,
    val rank: Rank,
    val commonName: String? = null,
    val genus: String? = null,
    val family: String? = null,
    val order: String? = null,
    val klass: String? = null,
)

/** One ranked possibility within a result. */
data class Candidate(
    val taxon: Taxon,
    /** Calibrated probability in 0..1 *after* temperature scaling. */
    val probability: Float,
    /** Probability from visual evidence alone, before any geographic prior. */
    val visualProbability: Float,
    /** Multiplier applied by the geographic prior; 1.0 when no location. */
    val geoFactor: Float = 1f,
    /** Model class index this candidate came from, for debugging and logging. */
    val classIndex: Int = -1,
    /** True when the geo prior has no occurrence data for this taxon at all. */
    val geoUnknown: Boolean = true,
) {
    val percent: Int get() = Math.round(probability * 100f)
}

/**
 * How sure the engine is, as a category rather than a bare number.
 *
 * The UI keys its emphasis off this, so the rules live in one place and are
 * testable. A raw probability is not enough: 0.55 with a runner-up at 0.05 is a
 * different situation from 0.55 with a runner-up at 0.50, and only the second is
 * a coin flip between two species.
 */
enum class Certainty {
    /** Clear winner, comfortably above threshold and well clear of runner-up. */
    CONFIDENT,

    /** Above threshold, but the runner-up is close. Show the comparison. */
    AMBIGUOUS,

    /** Not confident enough for a species claim; a coarser rank is offered. */
    COARSE_ONLY,

    /** Nothing trustworthy to say. */
    UNKNOWN,
}

/**
 * The outcome of identifying one or more photographs.
 *
 * [best] is deliberately nullable. When [certainty] is [Certainty.UNKNOWN]
 * there is no best candidate, and the type forces callers to handle that.
 */
data class Identification(
    val certainty: Certainty,
    val best: Candidate?,
    val alternatives: List<Candidate>,
    /**
     * Present when the engine backed off from species to a coarser rank
     * because the species-level evidence was too weak.
     */
    val coarseFallback: Candidate? = null,
    /** Calibrated entropy of the full distribution, normalised to 0..1. */
    val normalizedEntropy: Float = 1f,
    /** Margin between top-1 and top-2 calibrated probabilities. */
    val margin: Float = 0f,
    /** Number of photographs fused into this result. */
    val photoCount: Int = 1,
    /** Whether a geographic prior was applied. */
    val geoApplied: Boolean = false,
    val modelVersion: String = "",
    val packId: String = "",
    val packVersion: Int = 0,
    /** Wall-clock inference time, for the benchmark surface in Settings. */
    val inferenceMillis: Long = 0,
) {
    /** True when the engine is making no species-level claim. */
    val isUncertain: Boolean
        get() = certainty == Certainty.UNKNOWN || certainty == Certainty.COARSE_ONLY

    /** Everything worth showing, best first. */
    val ranked: List<Candidate>
        get() = listOfNotNull(best) + alternatives

    companion object {
        fun unknown(
            alternatives: List<Candidate> = emptyList(),
            normalizedEntropy: Float = 1f,
            photoCount: Int = 1,
            packId: String = "",
            packVersion: Int = 0,
            inferenceMillis: Long = 0,
        ) = Identification(
            certainty = Certainty.UNKNOWN,
            best = null,
            alternatives = alternatives,
            normalizedEntropy = normalizedEntropy,
            photoCount = photoCount,
            packId = packId,
            packVersion = packVersion,
            inferenceMillis = inferenceMillis,
        )
    }
}
