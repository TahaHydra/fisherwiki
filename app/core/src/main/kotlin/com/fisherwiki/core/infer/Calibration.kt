package com.fisherwiki.core.infer

import com.fisherwiki.core.pack.CalibrationSpec
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Turns raw logits into probabilities we are willing to show a user, and
 * decides when we are not willing to show any species at all.
 *
 * Three separate problems are handled here, and conflating them is how apps end
 * up confidently naming the wrong fish:
 *
 * **1. Overconfidence.** A softmax over a closed class set is systematically
 * overconfident: a modern classifier will happily report 0.97 on an input it
 * gets wrong. Temperature scaling (Guo et al., 2017) divides the logits by a
 * single scalar T fitted on held-out data. It cannot change *which* class wins,
 * so it never costs accuracy - it only makes the number mean something.
 *
 * **2. Open set.** The model has ~2,000 classes; the world has ~35,000 fish
 * species plus boots, hands, lures and dogs. Softmax always sums to 1, so an
 * unseen input still produces a winner. We therefore reject on three distinct
 * signals rather than one threshold:
 *
 *   - *top-1 probability* below [CalibrationSpec.unknownThreshold];
 *   - *margin* between top-1 and top-2 below [CalibrationSpec.marginThreshold] -
 *     this is what catches confusable species pairs, where the model is sure
 *     it is one of two things and has no idea which;
 *   - *normalised entropy* above [CalibrationSpec.entropyThreshold] - this is
 *     what catches "the model is mildly attracted to forty classes at once",
 *     which is the signature of an out-of-distribution photograph.
 *
 * A result can fail any one of these and be reported as uncertain even though
 * the other two look fine. That is deliberate.
 *
 * **3. Per-class thresholds.** Some classes are systematically harder or have
 * weaker training support; [CalibrationSpec.perClassThreshold] lets the
 * evaluation pipeline raise the bar for those specific classes without
 * penalising the rest.
 */
object Calibration {

    /** Result of turning logits into a usable distribution. */
    data class Distribution(
        val probabilities: FloatArray,
        val topIndices: IntArray,
        val topProbabilities: FloatArray,
        val margin: Float,
        val normalizedEntropy: Float,
    ) {
        val best: Int get() = topIndices.firstOrNull() ?: -1
        val bestProbability: Float get() = topProbabilities.firstOrNull() ?: 0f

        // Data class with array members: equals/hashCode must compare contents.
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Distribution) return false
            return probabilities.contentEquals(other.probabilities) &&
                topIndices.contentEquals(other.topIndices) &&
                topProbabilities.contentEquals(other.topProbabilities) &&
                margin == other.margin &&
                normalizedEntropy == other.normalizedEntropy
        }

        override fun hashCode(): Int {
            var r = probabilities.contentHashCode()
            r = 31 * r + topIndices.contentHashCode()
            r = 31 * r + topProbabilities.contentHashCode()
            r = 31 * r + margin.hashCode()
            r = 31 * r + normalizedEntropy.hashCode()
            return r
        }
    }

    /**
     * Numerically stable softmax with temperature.
     *
     * The max is subtracted before exponentiating; without that, a logit of
     * ~90 overflows float32 and the whole distribution becomes NaN. Fish
     * classifiers trained with label smoothing routinely produce logits in that
     * range, so this is a real failure, not a theoretical one.
     */
    fun softmax(logits: FloatArray, temperature: Float = 1f): FloatArray {
        require(logits.isNotEmpty()) { "empty logits" }
        require(temperature > 0f) { "temperature must be positive, was $temperature" }
        val scaled = FloatArray(logits.size) { logits[it] / temperature }
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in scaled) if (v > maxLogit) maxLogit = v
        var sum = 0.0
        val out = FloatArray(scaled.size)
        for (i in scaled.indices) {
            val e = exp((scaled[i] - maxLogit).toDouble())
            out[i] = e.toFloat()
            sum += e
        }
        val inv = (1.0 / sum).toFloat()
        for (i in out.indices) out[i] = out[i] * inv
        return out
    }

    /**
     * Shannon entropy normalised to 0..1 by `log(numClasses)`.
     *
     * Normalising matters because packs have different class counts: a raw
     * entropy of 2.0 nats is near-uniform for a 10-class model and extremely
     * peaked for a 2,000-class one, so an absolute threshold would mean
     * different things in different packs.
     */
    fun normalizedEntropy(probabilities: FloatArray): Float {
        if (probabilities.size <= 1) return 0f
        var h = 0.0
        for (p in probabilities) {
            if (p > 1e-12f) h -= p * ln(p.toDouble())
        }
        return (h / ln(probabilities.size.toDouble())).toFloat().coerceIn(0f, 1f)
    }

    /** Indices of the [k] largest values, descending. Partial selection, no full sort. */
    fun topK(values: FloatArray, k: Int): IntArray {
        val n = values.size
        val kk = minOf(k, n)
        if (kk <= 0) return IntArray(0)
        val idx = IntArray(kk) { -1 }
        val best = FloatArray(kk) { Float.NEGATIVE_INFINITY }
        for (i in 0 until n) {
            val v = values[i]
            if (v <= best[kk - 1]) continue
            var pos = kk - 1
            while (pos > 0 && best[pos - 1] < v) {
                best[pos] = best[pos - 1]
                idx[pos] = idx[pos - 1]
                pos--
            }
            best[pos] = v
            idx[pos] = i
        }
        return idx
    }

    /** Apply temperature, compute top-k, margin and entropy in one pass. */
    fun calibrate(logits: FloatArray, spec: CalibrationSpec, topK: Int = 5): Distribution {
        val probs = softmax(logits, spec.temperature)
        val top = topK(probs, topK)
        val topProbs = FloatArray(top.size) { probs[top[it]] }
        val margin = if (topProbs.size >= 2) topProbs[0] - topProbs[1] else topProbs.firstOrNull() ?: 0f
        return Distribution(
            probabilities = probs,
            topIndices = top,
            topProbabilities = topProbs,
            margin = margin,
            normalizedEntropy = normalizedEntropy(probs),
        )
    }

    /** Why a result was judged uncertain, so the UI can say something useful. */
    enum class RejectionReason {
        NONE,
        LOW_CONFIDENCE,
        NARROW_MARGIN,
        HIGH_ENTROPY,
        BELOW_CLASS_THRESHOLD,
    }

    /**
     * Decide whether the top class may be reported as a species-level answer.
     *
     * Returns [RejectionReason.NONE] when it may. Checks run in order of how
     * informative they are to a user.
     */
    fun reject(dist: Distribution, spec: CalibrationSpec): RejectionReason {
        val p = dist.bestProbability
        val classThreshold = spec.perClassThreshold[dist.best.toString()]
        if (classThreshold != null && p < classThreshold) {
            return RejectionReason.BELOW_CLASS_THRESHOLD
        }
        if (p < spec.unknownThreshold) return RejectionReason.LOW_CONFIDENCE
        if (dist.margin < spec.marginThreshold) return RejectionReason.NARROW_MARGIN
        if (dist.normalizedEntropy > spec.entropyThreshold) return RejectionReason.HIGH_ENTROPY
        return RejectionReason.NONE
    }

    /**
     * Fit a temperature on held-out logits by minimising NLL.
     *
     * Used by the offline evaluation pipeline, and kept here so the fitting and
     * the applying code cannot drift apart. Golden-section search over a single
     * scalar is plenty: the NLL is convex in `log T`.
     */
    fun fitTemperature(
        logits: Array<FloatArray>,
        labels: IntArray,
        lo: Float = 0.05f,
        hi: Float = 10f,
        iterations: Int = 60,
    ): Float {
        require(logits.size == labels.size) { "logits/labels length mismatch" }
        require(logits.isNotEmpty()) { "no validation data to fit temperature on" }

        fun nll(t: Float): Double {
            var total = 0.0
            for (i in logits.indices) {
                val p = softmax(logits[i], t)
                total -= ln(max(p[labels[i]], 1e-12f).toDouble())
            }
            return total / logits.size
        }

        var a = lo
        var b = hi
        val phi = 0.6180339887f
        var c = b - (b - a) * phi
        var d = a + (b - a) * phi
        var fc = nll(c)
        var fd = nll(d)
        repeat(iterations) {
            if (fc < fd) {
                b = d; d = c; fd = fc
                c = b - (b - a) * phi; fc = nll(c)
            } else {
                a = c; c = d; fc = fd
                d = a + (b - a) * phi; fd = nll(d)
            }
        }
        return (a + b) / 2f
    }

    /**
     * Expected Calibration Error with equal-width bins.
     *
     * Reported in the pack manifest and shown in Settings, because a user who
     * is told "87%" deserves that number to mean something close to "right 87
     * times in 100".
     */
    fun expectedCalibrationError(
        confidences: FloatArray,
        correct: BooleanArray,
        bins: Int = 15,
    ): Float {
        require(confidences.size == correct.size) { "length mismatch" }
        if (confidences.isEmpty()) return 0f
        val binConf = DoubleArray(bins)
        val binAcc = DoubleArray(bins)
        val binCount = IntArray(bins)
        for (i in confidences.indices) {
            val b = ((confidences[i] * bins).toInt()).coerceIn(0, bins - 1)
            binConf[b] += confidences[i].toDouble()
            binAcc[b] += if (correct[i]) 1.0 else 0.0
            binCount[b]++
        }
        var ece = 0.0
        val n = confidences.size.toDouble()
        for (b in 0 until bins) {
            if (binCount[b] == 0) continue
            val avgConf = binConf[b] / binCount[b]
            val avgAcc = binAcc[b] / binCount[b]
            ece += (binCount[b] / n) * kotlin.math.abs(avgConf - avgAcc)
        }
        return ece.toFloat()
    }
}
