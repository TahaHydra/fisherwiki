package com.fisherwiki.cli

import com.fisherwiki.core.model.Candidate
import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.model.Identification
import com.fisherwiki.core.model.RejectionReason
import com.fisherwiki.core.pack.CalibrationSpec
import com.fisherwiki.core.rank.CandidateRanker

/**
 * Turns an [Identification] into the human-readable explanation this CLI
 * exists to show.
 *
 * [explain] reads [Identification.rejectionReason] directly - the real field
 * the production ranker sets when it makes the decision, not a reconstruction
 * of it. That field did not always exist: `Identification` originally
 * computed the reason and then discarded it, which meant an earlier version
 * of this file had to rebuild a [com.fisherwiki.core.infer.Calibration.Distribution]
 * from margin/entropy/top-1 just to call `Calibration.reject` a second time
 * and re-derive the same answer downstream. Adding the field to production
 * removed that reconstruction entirely - this now *reports* a decision
 * instead of re-deriving one, which is the whole point of this CLI existing
 * on top of `:core` rather than beside it.
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

    /** Fallback/rejection reason for [id], read straight from the field the
     * ranker set - see the class kdoc. */
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
            val speciesPart = describeReason(id.rejectionReason, top1, id, calibration)
            val fallbackPart = when {
                id.certainty == Certainty.COARSE_ONLY -> {
                    val g = requireNotNull(id.coarseFallback)
                    ("; species claim rejected, but genus mass %.1f%% clears the genus " +
                        "threshold of %.0f%%, so \"%s\" is offered at genus rank").format(
                        g.probability * 100,
                        CandidateRanker.COARSE_THRESHOLD * 100,
                        g.taxon.scientificName,
                    )
                }
                top1 != null -> // UNKNOWN, but a species-level reason exists: genus was tried
                    ("; no genus cleared the %.0f%% genus threshold either (mass was spread " +
                        "across more than one genus, or too thin everywhere) - nothing is claimed")
                        .format(CandidateRanker.COARSE_THRESHOLD * 100)
                else -> "" // NO_CANDIDATE already says everything there is to say
            }
            "$speciesPart$fallbackPart"
        }
    }

    private fun describeReason(
        reason: RejectionReason,
        top1: Candidate?,
        id: Identification,
        calibration: CalibrationSpec,
    ): String = when (reason) {
        RejectionReason.NO_CANDIDATE ->
            "NO_CANDIDATE: every one of the model's classes failed to resolve to a taxon " +
                "in this pack's species database - a pack-consistency problem, not a " +
                "confidence one"

        RejectionReason.NONE ->
            // Not expected for COARSE_ONLY/UNKNOWN - reported plainly rather
            // than asserted against, since this function only explains the
            // real field, it does not second-guess it.
            "no threshold rejected the species claim, yet certainty is ${id.certainty}"

        RejectionReason.LOW_CONFIDENCE ->
            "LOW_CONFIDENCE: top species %s at %.1f%% is below unknown_threshold %.1f%%"
                .format(
                    top1?.taxon?.scientificName ?: "?", (top1?.probability ?: 0f) * 100,
                    calibration.unknownThreshold * 100,
                )

        RejectionReason.NARROW_MARGIN ->
            ("NARROW_MARGIN: top-2 margin %.4f is below margin_threshold %.4f - the model is " +
                "confident it is one of (at least) two similar species and does not know which")
                .format(id.margin, calibration.marginThreshold)

        RejectionReason.HIGH_ENTROPY ->
            ("HIGH_ENTROPY: normalised entropy %.4f exceeds entropy_threshold %.4f - probability " +
                "mass is spread across many classes rather than concentrated on one")
                .format(id.normalizedEntropy, calibration.entropyThreshold)

        RejectionReason.BELOW_CLASS_THRESHOLD ->
            ("BELOW_CLASS_THRESHOLD: %s carries its own, stricter threshold in this pack's " +
                "calibration, and %.1f%% does not clear it")
                .format(top1?.taxon?.scientificName ?: "?", (top1?.probability ?: 0f) * 100)
    }
}
