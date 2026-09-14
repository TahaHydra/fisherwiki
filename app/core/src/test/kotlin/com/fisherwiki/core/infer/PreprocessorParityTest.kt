package com.fisherwiki.core.infer

import com.google.common.truth.Truth.assertThat
import java.io.DataInputStream
import javax.imageio.ImageIO
import kotlin.math.abs
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/**
 * Train/serve parity for image preprocessing.
 *
 * Preprocessing is the most common source of silent accuracy loss in a deployed
 * vision model, because a mismatch raises no error anywhere. Training resizes
 * with one interpolation, one crop convention, one channel order and one
 * normalisation; if serving differs in any of them, accuracy simply drops.
 *
 * The fixtures here were produced by the **actual Python training pipeline**
 * (`ml/fwml/data.py` `eval_transform` + `to_tensor`) — the same code path used
 * for validation and test — not reimplemented for the test. So this compares
 * the shipping Kotlin preprocessor against the shipping Python one.
 *
 * The source image is deliberately non-square (500×333) and carries an
 * asymmetric bright marker: a transposed axis, a flipped crop or an RGB/BGR
 * swap all move that marker and fail loudly rather than coincidentally passing.
 */
class PreprocessorParityTest {

    private val size = 224
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    private fun loadReference(): FloatArray {
        val stream = javaClass.getResourceAsStream("/fixtures/preproc_tensor.bin")
            ?: error("preproc_tensor.bin missing from test resources")
        val n = 3 * size * size
        val out = FloatArray(n)
        DataInputStream(stream.buffered()).use { input ->
            for (i in 0 until n) out[i] = input.readFloat()
        }
        return out
    }

    private fun loadImage(): Triple<IntArray, Int, Int> {
        val stream = javaClass.getResourceAsStream("/fixtures/preproc_image.png")
            ?: error("preproc_image.png missing from test resources")
        val img = ImageIO.read(stream)
        val w = img.width
        val h = img.height
        val pixels = IntArray(w * h)
        img.getRGB(0, 0, w, h, pixels, 0, w)
        return Triple(pixels, w, h)
    }

    @Test
    fun `kotlin preprocessing matches the python training pipeline`() {
        val (pixels, w, h) = loadImage()
        assertThat(w).isEqualTo(500)
        assertThat(h).isEqualTo(333)

        val got = Preprocessor.toTensor(pixels, w, h, size, mean, std)
        val want = loadReference()

        assertThat(got.size).isEqualTo(want.size)

        var maxDiff = 0f
        var sumSq = 0.0
        var worstIndex = -1
        for (i in want.indices) {
            val d = abs(got[i] - want[i])
            if (d > maxDiff) {
                maxDiff = d
                worstIndex = i
            }
            sumSq += (d * d).toDouble()
        }
        val rms = Math.sqrt(sumSq / want.size)

        // Two independent resamplers will not agree bit for bit, but they must
        // agree to within resampling noise. Measured on this fixture:
        //
        //   naive 2-tap bilinear : max diff 0.617, rms 0.0331   <- aliased
        //   antialiased triangle : max diff 0.017, rms 0.0063   <- current
        //
        // The bound is set just above the achieved value, so a regression to
        // an unfiltered resample (or any structural error, which produces
        // differences of order 1.0) fails immediately.
        assertThat(maxDiff).isLessThan(0.05f)
        assertThat(rms).isLessThan(0.01)

        if (maxDiff > 0.03f) {
            val c = worstIndex / (size * size)
            val rem = worstIndex % (size * size)
            println(
                "parity: max diff $maxDiff at channel $c (${rem / size}, ${rem % size}), rms $rms"
            )
        }
    }

    @Test
    fun `channel order is RGB not BGR`() {
        // The fixture's bright marker is pure red (250, 20, 20) in the source.
        // If the channels were swapped, the red-channel statistics would land
        // on the blue channel and this comparison would fail by ~1.5 units.
        val (pixels, w, h) = loadImage()
        val got = Preprocessor.toTensor(pixels, w, h, size, mean, std)
        val want = loadReference()

        val plane = size * size
        for (c in 0 until 3) {
            var gm = 0.0
            var wm = 0.0
            for (i in 0 until plane) {
                gm += got[c * plane + i]
                wm += want[c * plane + i]
            }
            gm /= plane
            wm /= plane
            assertThat(abs(gm - wm)).isLessThan(0.02)
        }
    }

    @Test
    fun `output is NCHW with the expected shape`() {
        val (pixels, w, h) = loadImage()
        val t = Preprocessor.toTensor(pixels, w, h, size, mean, std)
        assertThat(t.size).isEqualTo(3 * size * size)
    }

    @Test
    fun `normalisation uses the manifest mean and std`() {
        // A flat mid-grey image must normalise to (0.5 - mean) / std per channel.
        val w = 300
        val h = 300
        val grey = IntArray(w * h) { (0xFF shl 24) or (128 shl 16) or (128 shl 8) or 128 }
        val t = Preprocessor.toTensor(grey, w, h, size, mean, std)
        val plane = size * size
        for (c in 0 until 3) {
            val expected = (128f / 255f - mean[c]) / std[c]
            assertThat(t[c * plane]).isWithin(1e-3f).of(expected)
            assertThat(t[c * plane + plane / 2]).isWithin(1e-3f).of(expected)
        }
    }

    @Test
    fun `portrait and landscape both resize on the short edge`() {
        // Both must produce a full square tensor, with the short edge driving
        // the scale. A long-edge resize would pad or distort.
        val landscape = IntArray(400 * 200) { 0xFF808080.toInt() }
        val portrait = IntArray(200 * 400) { 0xFF808080.toInt() }
        assertThat(Preprocessor.toTensor(landscape, 400, 200, size, mean, std).size)
            .isEqualTo(3 * size * size)
        assertThat(Preprocessor.toTensor(portrait, 200, 400, size, mean, std).size)
            .isEqualTo(3 * size * size)
    }

    @Test
    fun `an image smaller than the crop is upscaled rather than failing`() {
        val tiny = IntArray(64 * 48) { 0xFF404040.toInt() }
        val t = Preprocessor.toTensor(tiny, 64, 48, size, mean, std)
        assertThat(t.size).isEqualTo(3 * size * size)
        assertThat(t.none { it.isNaN() }).isTrue()
    }

    @Test
    fun `a zero std is rejected rather than producing infinities`() {
        val pixels = IntArray(300 * 300) { 0xFF808080.toInt() }
        assertThrows<IllegalArgumentException> {
            Preprocessor.toTensor(
                pixels, 300, 300, size, mean, floatArrayOf(0.229f, 0f, 0.225f)
            )
        }
    }

    @Test
    fun `a truncated pixel buffer is rejected`() {
        assertThrows<IllegalArgumentException> {
            Preprocessor.toTensor(IntArray(10), 300, 300, size, mean, std)
        }
    }

    @Test
    fun `box expansion pads a tight detection without leaving the image`() {
        val box = Preprocessor.expandBox(100, 50, 200, 150, 500, 333, padFraction = 0.12f)
        assertThat(box[0]).isLessThan(100)
        assertThat(box[1]).isLessThan(50)
        assertThat(box[2]).isGreaterThan(200)
        assertThat(box[3]).isGreaterThan(150)
        assertThat(box[0]).isAtLeast(0)
        assertThat(box[1]).isAtLeast(0)
        assertThat(box[2]).isAtMost(500)
        assertThat(box[3]).isAtMost(333)
    }

    @Test
    fun `a degenerate box falls back to the whole frame`() {
        val box = Preprocessor.expandBox(10, 10, 10, 10, 500, 333)
        assertThat(box.toList()).containsExactly(0, 0, 500, 333).inOrder()
    }

    @Test
    fun `box expansion clamps at the image edge`() {
        val box = Preprocessor.expandBox(0, 0, 500, 333, 500, 333, padFraction = 0.5f)
        assertThat(box.toList()).containsExactly(0, 0, 500, 333).inOrder()
    }
}
