package com.fisherwiki.core.infer

import com.fisherwiki.core.pack.CalibrationSpec
import com.google.common.truth.Truth.assertThat
import kotlin.math.abs
import kotlin.random.Random
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class CalibrationTest {

    private val spec = CalibrationSpec(
        temperature = 1.0f,
        unknownThreshold = 0.35f,
        marginThreshold = 0.08f,
        entropyThreshold = 0.85f,
    )

    // ------------------------------------------------------------- softmax

    @Test
    fun `softmax sums to one`() {
        val p = Calibration.softmax(floatArrayOf(1f, 2f, 3f, 4f))
        assertThat(p.sum()).isWithin(1e-5f).of(1f)
    }

    @Test
    fun `softmax does not overflow on large logits`() {
        // Regression: exp(90) overflows float32. Models trained with label
        // smoothing routinely emit logits in this range, so a naive softmax
        // returns NaN for exactly the confident predictions users see most.
        val p = Calibration.softmax(floatArrayOf(90f, 89f, 1f, -50f))
        assertThat(p.none { it.isNaN() }).isTrue()
        assertThat(p.sum()).isWithin(1e-4f).of(1f)
        assertThat(p[0]).isGreaterThan(p[1])
    }

    @Test
    fun `softmax handles very negative logits`() {
        val p = Calibration.softmax(floatArrayOf(-1000f, -1001f, -999f))
        assertThat(p.none { it.isNaN() }).isTrue()
        assertThat(p.sum()).isWithin(1e-4f).of(1f)
    }

    @Test
    fun `temperature above one softens the distribution`() {
        val logits = floatArrayOf(6f, 2f, 1f, 0f)
        val sharp = Calibration.softmax(logits, 1f)
        val soft = Calibration.softmax(logits, 3f)
        assertThat(soft[0]).isLessThan(sharp[0])
        assertThat(Calibration.normalizedEntropy(soft))
            .isGreaterThan(Calibration.normalizedEntropy(sharp))
    }

    @Test
    fun `temperature never changes the argmax`() {
        // This is the property that makes temperature scaling safe: it fixes
        // the numbers without ever changing which species wins.
        val rng = Random(7)
        repeat(200) {
            val logits = FloatArray(50) { rng.nextFloat() * 20f - 10f }
            val base = Calibration.softmax(logits, 1f).withIndex().maxBy { it.value }.index
            for (t in listOf(0.2f, 0.5f, 2f, 5f, 9f)) {
                val idx = Calibration.softmax(logits, t).withIndex().maxBy { it.value }.index
                assertThat(idx).isEqualTo(base)
            }
        }
    }

    @Test
    fun `zero or negative temperature is rejected`() {
        assertThrows<IllegalArgumentException> {
            Calibration.softmax(floatArrayOf(1f, 2f), 0f)
        }
        assertThrows<IllegalArgumentException> {
            Calibration.softmax(floatArrayOf(1f, 2f), -1f)
        }
    }

    // ------------------------------------------------------------- entropy

    @Test
    fun `uniform distribution has normalised entropy one`() {
        val p = FloatArray(100) { 0.01f }
        assertThat(Calibration.normalizedEntropy(p)).isWithin(1e-4f).of(1f)
    }

    @Test
    fun `one hot distribution has entropy zero`() {
        val p = FloatArray(100).also { it[3] = 1f }
        assertThat(Calibration.normalizedEntropy(p)).isWithin(1e-4f).of(0f)
    }

    @Test
    fun `entropy normalisation makes class counts comparable`() {
        // A raw entropy of ~2.3 nats is near-uniform over 10 classes but very
        // peaked over 2000. Without normalising, one threshold cannot serve
        // packs of different sizes.
        val tenUniform = FloatArray(10) { 0.1f }
        val twoThousandUniform = FloatArray(2000) { 1f / 2000f }
        assertThat(Calibration.normalizedEntropy(tenUniform))
            .isWithin(1e-4f).of(Calibration.normalizedEntropy(twoThousandUniform))
    }

    // ---------------------------------------------------------------- topK

    @Test
    fun `topK returns indices in descending order`() {
        val v = floatArrayOf(0.1f, 0.9f, 0.5f, 0.7f, 0.2f)
        assertThat(Calibration.topK(v, 3).toList()).containsExactly(1, 3, 2).inOrder()
    }

    @Test
    fun `topK handles k larger than the array`() {
        val v = floatArrayOf(0.3f, 0.7f)
        assertThat(Calibration.topK(v, 10).toList()).containsExactly(1, 0).inOrder()
    }

    @Test
    fun `topK matches a full sort on random data`() {
        val rng = Random(11)
        repeat(100) {
            val v = FloatArray(500) { rng.nextFloat() }
            val fast = Calibration.topK(v, 5).toList()
            val slow = v.withIndex().sortedByDescending { it.value }.take(5).map { it.index }
            assertThat(fast).isEqualTo(slow)
        }
    }

    // ----------------------------------------------------------- rejection

    @Test
    fun `confident well separated prediction is accepted`() {
        val dist = Calibration.calibrate(floatArrayOf(8f, 1f, 0.5f, 0f), spec)
        assertThat(Calibration.reject(dist, spec)).isEqualTo(Calibration.RejectionReason.NONE)
    }

    @Test
    fun `low confidence is rejected`() {
        // Spread mass so nothing reaches 0.35.
        val dist = Calibration.calibrate(FloatArray(20) { if (it == 0) 1.2f else 1f }, spec)
        assertThat(dist.bestProbability).isLessThan(spec.unknownThreshold)
        assertThat(Calibration.reject(dist, spec))
            .isEqualTo(Calibration.RejectionReason.LOW_CONFIDENCE)
    }

    @Test
    fun `narrow margin between two similar species is rejected`() {
        // This is the confusable-pair case: the model is sure it is one of two
        // things and has no idea which. Top-1 alone looks fine here.
        val dist = Calibration.calibrate(floatArrayOf(5.0f, 4.98f, -3f, -4f), spec)
        assertThat(dist.bestProbability).isGreaterThan(spec.unknownThreshold)
        assertThat(Calibration.reject(dist, spec))
            .isEqualTo(Calibration.RejectionReason.NARROW_MARGIN)
    }

    @Test
    fun `per class threshold can veto an otherwise acceptable result`() {
        val strict = spec.copy(perClassThreshold = mapOf("0" to 0.95f))
        val dist = Calibration.calibrate(floatArrayOf(4f, 0f, 0f, 0f), strict)
        assertThat(dist.best).isEqualTo(0)
        assertThat(Calibration.reject(dist, strict))
            .isEqualTo(Calibration.RejectionReason.BELOW_CLASS_THRESHOLD)
    }

    @Test
    fun `per class threshold does not affect other classes`() {
        val strict = spec.copy(perClassThreshold = mapOf("3" to 0.99f))
        val dist = Calibration.calibrate(floatArrayOf(6f, 0f, 0f, 0f), strict)
        assertThat(Calibration.reject(dist, strict)).isEqualTo(Calibration.RejectionReason.NONE)
    }

    // --------------------------------------------------- temperature fitting

    @Test
    fun `fitTemperature recovers a known overconfidence`() {
        // Build a synthetic overconfident model: true logits scaled up by 2.5.
        // Fitting should recover roughly that factor.
        val rng = Random(3)
        val numClasses = 12
        val n = 800
        val logits = Array(n) { FloatArray(numClasses) }
        val labels = IntArray(n)
        for (i in 0 until n) {
            val label = rng.nextInt(numClasses)
            labels[i] = label
            for (c in 0 until numClasses) {
                logits[i][c] = (rng.nextFloat() * 2f - 1f) * 1.5f
            }
            // Correct 70% of the time, so the model should not be certain.
            if (rng.nextFloat() < 0.70f) logits[i][label] += 3.0f
            else logits[i][(label + 1) % numClasses] += 3.0f
            for (c in 0 until numClasses) logits[i][c] *= 2.5f
        }
        val t = Calibration.fitTemperature(logits, labels)
        assertThat(t).isGreaterThan(1.5f)
        assertThat(t).isLessThan(6.0f)

        // And the fitted temperature must reduce calibration error.
        fun ece(temp: Float): Float {
            val conf = FloatArray(n)
            val correct = BooleanArray(n)
            for (i in 0 until n) {
                val p = Calibration.softmax(logits[i], temp)
                val best = p.withIndex().maxBy { it.value }
                conf[i] = best.value
                correct[i] = best.index == labels[i]
            }
            return Calibration.expectedCalibrationError(conf, correct)
        }
        assertThat(ece(t)).isLessThan(ece(1f))
    }

    @Test
    fun `fitTemperature rejects empty data`() {
        assertThrows<IllegalArgumentException> {
            Calibration.fitTemperature(emptyArray(), IntArray(0))
        }
    }

    // ---------------------------------------------------------------- ECE

    @Test
    fun `perfectly calibrated predictions have near zero ECE`() {
        // 1000 predictions at confidence 0.8, exactly 80% of them correct.
        val n = 1000
        val conf = FloatArray(n) { 0.8f }
        val correct = BooleanArray(n) { it < 800 }
        assertThat(Calibration.expectedCalibrationError(conf, correct)).isLessThan(0.01f)
    }

    @Test
    fun `overconfident predictions have high ECE`() {
        val n = 1000
        val conf = FloatArray(n) { 0.99f }
        val correct = BooleanArray(n) { it < 500 }   // only 50% right
        assertThat(Calibration.expectedCalibrationError(conf, correct)).isGreaterThan(0.4f)
    }

    @Test
    fun `ECE of empty input is zero not NaN`() {
        assertThat(Calibration.expectedCalibrationError(FloatArray(0), BooleanArray(0)))
            .isEqualTo(0f)
    }
}
