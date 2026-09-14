package com.fisherwiki.core.rank

import com.fisherwiki.core.infer.Calibration
import com.fisherwiki.core.model.Candidate
import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.model.Identification
import com.fisherwiki.core.model.Rank
import com.fisherwiki.core.model.Taxon
import com.fisherwiki.core.pack.CalibrationSpec
import kotlin.math.ln
import kotlin.math.max

/**
 * Turns model logits into a ranked, honest [Identification].
 *
 * Ordering of concerns, which is also the order they are applied:
 *
 * 1. **Calibrate.** Temperature-scale the logits so the numbers mean something.
 * 2. **Apply the geographic prior**, bounded and attenuated by visual
 *    confidence, then renormalise. Visual evidence stays primary; see
 *    [GeoPrior] for why the prior can never zero a species out.
 * 3. **Decide certainty** from three independent signals, not one threshold.
 * 4. **Fall back to a coarser rank** rather than guessing a species, when the
 *    species-level evidence does not support a claim.
 *
 * The class does no I/O and holds no Android types, so all of this is directly
 * unit-testable, which matters because these rules are the product.
 */
class CandidateRanker(
    private val calibration: CalibrationSpec,
    private val geoPrior: GeoPrior? = null,
    private val taxonResolver: (Int) -> Taxon?,
    /** classIndex -> genus key, for coarse fallback by aggregation. */
    private val genusOfClass: IntArray? = null,
    private val genusName: (Int) -> String? = { null },
) {

    data class Location(val latitude: Double, val longitude: Double)

    /**
     * Rank a single photograph's logits.
     *
     * @param location optional; when null, [Candidate.geoFactor] is 1.0 for all
     *   candidates and the result is purely visual.
     */
    fun rank(
        logits: FloatArray,
        location: Location? = null,
        topK: Int = 5,
        packId: String = "",
        packVersion: Int = 0,
        inferenceMillis: Long = 0,
    ): Identification {
        val dist = Calibration.calibrate(logits, calibration, topK = max(topK, 5))
        val visualBest = dist.bestProbability

        val adjusted: FloatArray
        val geoApplied: Boolean
        if (location != null && geoPrior != null) {
            adjusted = FloatArray(dist.probabilities.size)
            var sum = 0.0
            for (i in dist.probabilities.indices) {
                val f = geoPrior.factor(i, location.latitude, location.longitude, visualBest)
                val v = dist.probabilities[i] * f
                adjusted[i] = v
                sum += v
            }
            val inv = (1.0 / max(sum, 1e-12)).toFloat()
            for (i in adjusted.indices) adjusted[i] *= inv
            geoApplied = true
        } else {
            adjusted = dist.probabilities
            geoApplied = false
        }

        val order = Calibration.topK(adjusted, max(topK, 5))
        val candidates = order.asList().mapNotNull { idx ->
            val taxon = taxonResolver(idx) ?: return@mapNotNull null
            Candidate(
                taxon = taxon,
                probability = adjusted[idx],
                visualProbability = dist.probabilities[idx],
                geoFactor = if (geoApplied && geoPrior != null) {
                    geoPrior.factor(idx, location!!.latitude, location.longitude, visualBest)
                } else 1f,
                classIndex = idx,
                geoUnknown = geoPrior?.isUnknown(idx) ?: true,
            )
        }

        if (candidates.isEmpty()) {
            return Identification.unknown(
                normalizedEntropy = dist.normalizedEntropy,
                packId = packId, packVersion = packVersion,
                inferenceMillis = inferenceMillis,
            )
        }

        // Recompute margin and entropy on the *adjusted* distribution: the geo
        // prior can create or destroy ambiguity, and the user is shown the
        // adjusted numbers, so the certainty decision must use them too.
        val adjMargin =
            if (candidates.size >= 2) candidates[0].probability - candidates[1].probability
            else candidates[0].probability
        val adjEntropy = Calibration.normalizedEntropy(adjusted)

        val best = candidates.first()
        val reason = rejectionFor(best, adjMargin, adjEntropy)

        return when (reason) {
            Calibration.RejectionReason.NONE -> Identification(
                certainty = if (adjMargin < calibration.marginThreshold * 2.5f)
                    Certainty.AMBIGUOUS else Certainty.CONFIDENT,
                best = best,
                alternatives = candidates.drop(1).take(topK - 1),
                normalizedEntropy = adjEntropy,
                margin = adjMargin,
                geoApplied = geoApplied,
                packId = packId,
                packVersion = packVersion,
                inferenceMillis = inferenceMillis,
            )
            else -> coarseOrUnknown(
                candidates = candidates,
                adjusted = adjusted,
                margin = adjMargin,
                entropy = adjEntropy,
                topK = topK,
                geoApplied = geoApplied,
                packId = packId,
                packVersion = packVersion,
                inferenceMillis = inferenceMillis,
            )
        }
    }

    private fun rejectionFor(
        best: Candidate,
        margin: Float,
        entropy: Float,
    ): Calibration.RejectionReason {
        val perClass = calibration.perClassThreshold[best.classIndex.toString()]
        if (perClass != null && best.probability < perClass) {
            return Calibration.RejectionReason.BELOW_CLASS_THRESHOLD
        }
        if (best.probability < calibration.unknownThreshold) {
            return Calibration.RejectionReason.LOW_CONFIDENCE
        }
        if (margin < calibration.marginThreshold) {
            return Calibration.RejectionReason.NARROW_MARGIN
        }
        if (entropy > calibration.entropyThreshold) {
            return Calibration.RejectionReason.HIGH_ENTROPY
        }
        return Calibration.RejectionReason.NONE
    }

    /**
     * When a species claim is not supportable, try to make a *genus* claim.
     *
     * The aggregation is deliberate: if the model spreads 0.30/0.28/0.15 across
     * three *Sebastes*, no species clears the bar but "this is a *Sebastes*" is
     * supported at 0.73 and is genuinely useful to an angler. If the mass is
     * spread across three different families instead, nothing is claimed.
     */
    private fun coarseOrUnknown(
        candidates: List<Candidate>,
        adjusted: FloatArray,
        margin: Float,
        entropy: Float,
        topK: Int,
        geoApplied: Boolean,
        packId: String,
        packVersion: Int,
        inferenceMillis: Long,
    ): Identification {
        val genusMap = genusOfClass
        if (genusMap != null) {
            val byGenus = HashMap<Int, Float>()
            for (i in adjusted.indices) {
                if (i >= genusMap.size) continue
                byGenus[genusMap[i]] = (byGenus[genusMap[i]] ?: 0f) + adjusted[i]
            }
            val topGenus = byGenus.maxByOrNull { it.value }
            if (topGenus != null && topGenus.value >= COARSE_THRESHOLD) {
                val exemplar = candidates.firstOrNull {
                    it.classIndex < genusMap.size && genusMap[it.classIndex] == topGenus.key
                }
                val name = genusName(topGenus.key)
                    ?: exemplar?.taxon?.genus
                    ?: exemplar?.taxon?.scientificName?.substringBefore(' ')
                if (name != null) {
                    val coarse = Candidate(
                        taxon = Taxon(
                            id = -1L - topGenus.key,
                            scientificName = name,
                            rank = Rank.GENUS,
                            genus = name,
                            family = exemplar?.taxon?.family,
                        ),
                        probability = topGenus.value,
                        visualProbability = topGenus.value,
                        classIndex = -1,
                        geoUnknown = true,
                    )
                    return Identification(
                        certainty = Certainty.COARSE_ONLY,
                        best = null,
                        alternatives = candidates.take(topK),
                        coarseFallback = coarse,
                        normalizedEntropy = entropy,
                        margin = margin,
                        geoApplied = geoApplied,
                        packId = packId,
                        packVersion = packVersion,
                        inferenceMillis = inferenceMillis,
                    )
                }
            }
        }
        return Identification.unknown(
            alternatives = candidates.take(topK),
            normalizedEntropy = entropy,
            packId = packId,
            packVersion = packVersion,
            inferenceMillis = inferenceMillis,
        )
    }

    /**
     * Fuse several photographs of the same fish (Expert ID).
     *
     * Uses an entropy-weighted **arithmetic** mean of calibrated probabilities
     * (the "sum rule"), not a log-space/geometric mean (the "product rule").
     *
     * This was implemented the other way round first, on the plausible-sounding
     * argument that log-space averaging is the correct combination under
     * independence. A test caught that it is exactly backwards for our failure
     * mode. Worked example, three frames of one fish, two agreeing and one
     * confidently wrong (a shot of the water, or the angler's hand):
     *
     * ```
     * frame A/B  p = [0.988, 0.007, 0.002, 0.002]   -> class 0
     * frame C    p = [1e-5,  1.000, 1e-5,  1e-5 ]   -> class 1, very loudly
     *
     * product rule (log space) -> [-4.16, -3.28, ...]  argmax = 1   WRONG
     * sum rule     (arithmetic) -> [0.647, 0.350, ...]  argmax = 0   correct
     * ```
     *
     * The product rule gives any single frame a veto: one near-zero probability
     * drives that class's log to about -12 and no amount of agreement elsewhere
     * recovers it. The sum rule degrades gracefully instead. This matches the
     * long-standing result in Kittler et al., *On Combining Classifiers* (1998),
     * that the sum rule is markedly more resilient to individual estimation
     * errors than the product rule.
     *
     * Frames the model finds uninformative (high entropy) are down-weighted
     * rather than dropped, since a blurry tail shot still carries some
     * evidence.
     */
    fun fuse(
        perPhotoLogits: List<FloatArray>,
        location: Location? = null,
        topK: Int = 5,
        packId: String = "",
        packVersion: Int = 0,
        inferenceMillis: Long = 0,
    ): Identification {
        require(perPhotoLogits.isNotEmpty()) { "no photographs to fuse" }
        if (perPhotoLogits.size == 1) {
            return rank(perPhotoLogits[0], location, topK, packId, packVersion, inferenceMillis)
        }
        val n = perPhotoLogits[0].size
        require(perPhotoLogits.all { it.size == n }) {
            "all photographs must come from the same model"
        }

        val accum = DoubleArray(n)
        var weightSum = 0.0
        for (logits in perPhotoLogits) {
            val p = Calibration.softmax(logits, calibration.temperature)
            val h = Calibration.normalizedEntropy(p)
            // Weight in (0, 1]: a maximally uninformative frame still counts a
            // little, an informative one counts fully.
            val w = (1.0 - h).coerceIn(0.05, 1.0)
            for (i in 0 until n) {
                accum[i] += w * p[i].toDouble()
            }
            weightSum += w
        }

        // Hand the fused *probabilities* back to rank() as log-probabilities.
        // softmax(ln p) == p exactly, so ranking with temperature 1 reproduces
        // the fused distribution without applying temperature a second time
        // (it was already applied per frame above).
        val fusedLogits = FloatArray(n) {
            ln(max(accum[it] / weightSum, 1e-12)).toFloat()
        }
        val neutral = calibration.copy(temperature = 1.0f)
        return CandidateRanker(neutral, geoPrior, taxonResolver, genusOfClass, genusName)
            .rank(fusedLogits, location, topK, packId, packVersion, inferenceMillis)
            .copy(photoCount = perPhotoLogits.size)
    }

    companion object {
        /** Mass that must accumulate on one genus before we will name it. */
        const val COARSE_THRESHOLD = 0.55f
    }
}
