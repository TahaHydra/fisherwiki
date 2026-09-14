package com.fisherwiki.cli

import com.fisherwiki.core.infer.Calibration
import com.fisherwiki.core.model.Candidate
import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.model.Identification
import com.fisherwiki.core.pack.CalibrationSpec
import com.fisherwiki.core.rank.CandidateRanker

/**
 * Turns an [Identification] into the human-readable explanation this CLI
 * exists to show.
 *
 * [explain] specifically does not re-derive *why* a result landed where it
 * did from scratch. `Identification` does not carry a rejection reason field,
 * but `:core` already has a pure, public function whose only job is exactly
 * this - [Calibration.reject] - unused by the Android UI today but written
 * for precisely this purpose. Reconstructing the minimal [Calibration.Distribution]
 * it needs from fields `Identification` already exposes (`margin`,
 * `normalizedEntropy`, and the original top-1 candidate) and calling that
 * real function is the difference between *reporting* a production decision
 * and *reimplementing* one.
 */
object Report {

    fun decisionLabel(certainty: Certainty): String = when (certainty) {
        Certainty.CONFIDENT -> "SPECIES"
        Certainty.AMBIGUOUS -> "SPECIES (ambiguous)"
        Certainty.COARSE_ONLY -> "GENUS"
        Certainty.UNKNOWN -> "UNCERTAIN"
    }

    fun candidateLine(rank: Int, c: Candidate): String {
        val common = c.taxon.commonName ?: "-"
        val geo = if (c.geoUnknown) "" else "  (geo x%.2f)".format(c.geoFactor)
        return "    %2d. %6.2f%%  %-32s %s%s"
            .format(rank, c.probability * 100, c.taxon.scientificName, common, geo)
    }

    /**
     * Fallback/rejection reason for [id], derived from the exact same public
     * calibration function and thresholds the ranker used to decide [id]'s
     * certainty in the first place.
     */
    fun explain(id: Identification, calibration: CalibrationSpec): String = when (id.certainty) {
        Certainty.CONFIDENT ->
            "none - identified with a clear margin over the runner-up"

        // The 2.5x multiplier is CandidateRanker.rank's own literal constant
        // for CONFIDENT-vs-AMBIGUOUS, reproduced here rather than exposed as
        // a new public constant, since it is a display nuance rather than a
        // decision this CLI makes on its own.
        Certainty.AMBIGUOUS ->
            ("none - identified, but the runner-up is close (margin %.4f is below " +
                "the ambiguity bound of %.4f = 2.5 x margin_threshold); shown as " +
                "\"likely, but check\" rather than a plain identification")
                .format(id.margin, calibration.marginThreshold * 2.5f)

        Certainty.COARSE_ONLY, Certainty.UNKNOWN -> {
            val top1 = id.ranked.firstOrNull()
            if (top1 == null) {
                "NO_CANDIDATE: every one of the model's classes failed to resolve to a " +
                    "taxon in this pack's species database - a pack-consistency problem, " +
                    "not a confidence one"
            } else {
                val reason = rejectionReasonFor(top1, id, calibration)
                val speciesPart = describeReason(reason, top1, id, calibration)
                val fallbackPart = if (id.certainty == Certainty.COARSE_ONLY) {
                    val g = requireNotNull(id.coarseFallback)
                    ("; species claim rejected, but genus mass %.1f%% clears the genus " +
                        "threshold of %.0f%%, so \"%s\" is offered at genus rank").format(
                        g.probability * 100,
                        CandidateRanker.COARSE_THRESHOLD * 100,
                        g.taxon.scientificName,
                    )
                } else {
                    ("; no genus cleared the %.0f%% genus threshold either (mass was spread " +
                        "across more than one genus, or too thin everywhere) - nothing is claimed")
                        .format(CandidateRanker.COARSE_THRESHOLD * 100)
                }
                "$speciesPart$fallbackPart"
            }
        }
    }

    /**
     * Reconstructs the [Calibration.Distribution] `Calibration.reject` needs,
     * from fields `Identification` already carries. Only `topIndices[0]`,
     * `topProbabilities[0]`, `margin` and `normalizedEntropy` are read by
     * `reject`, so `probabilities` is left empty rather than fabricated.
     */
    private fun rejectionReasonFor(
        top1: Candidate,
        id: Identification,
        calibration: CalibrationSpec,
    ): Calibration.RejectionReason {
        val dist = Calibration.Distribution(
            probabilities = FloatArray(0),
            topIndices = intArrayOf(top1.classIndex),
            topProbabilities = floatArrayOf(top1.probability),
            margin = id.margin,
            normalizedEntropy = id.normalizedEntropy,
        )
        return Calibration.reject(dist, calibration)
    }

    private fun describeReason(
        reason: Calibration.RejectionReason,
        top1: Candidate,
        id: Identification,
        calibration: CalibrationSpec,
    ): String = when (reason) {
        Calibration.RejectionReason.NONE ->
            // Reachable in principle if margin/entropy improved between the
            // real ranking pass and this reconstruction, which should not
            // happen; reported plainly rather than asserted against, since
            // this function only explains, it does not re-decide.
            "species-level checks did not themselves reject %s at %.1f%%"
                .format(top1.taxon.scientificName, top1.probability * 100)

        Calibration.RejectionReason.LOW_CONFIDENCE ->
            "LOW_CONFIDENCE: top species %s at %.1f%% is below unknown_threshold %.1f%%"
                .format(top1.taxon.scientificName, top1.probability * 100, calibration.unknownThreshold * 100)

        Calibration.RejectionReason.NARROW_MARGIN ->
            ("NARROW_MARGIN: top-2 margin %.4f is below margin_threshold %.4f - the model is " +
                "confident it is one of (at least) two similar species and does not know which")
                .format(id.margin, calibration.marginThreshold)

        Calibration.RejectionReason.HIGH_ENTROPY ->
            ("HIGH_ENTROPY: normalised entropy %.4f exceeds entropy_threshold %.4f - probability " +
                "mass is spread across many classes rather than concentrated on one")
                .format(id.normalizedEntropy, calibration.entropyThreshold)

        Calibration.RejectionReason.BELOW_CLASS_THRESHOLD ->
            ("BELOW_CLASS_THRESHOLD: %s carries its own, stricter threshold in this pack's " +
                "calibration, and %.1f%% does not clear it")
                .format(top1.taxon.scientificName, top1.probability * 100)
    }
}
