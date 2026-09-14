package com.fisherwiki.cli

import com.google.common.truth.Truth.assertThat
import com.google.common.truth.Truth.assertWithMessage
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.IIOImage
import javax.imageio.ImageIO
import javax.imageio.ImageWriteParam
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource

/**
 * Getting EXIF orientation wrong is a silent-accuracy-loss bug, not a crash:
 * the model would still run, just on a rotated image, and nothing would ever
 * say so. `PreprocessorParityTest` in `:core` guards the equivalent risk for
 * resizing/cropping; this is that same discipline applied to orientation.
 *
 * Three independent layers, deliberately not round-tripped through a single
 * lossy JPEG comparison (which would only ever prove the two halves agree with
 * each other, not that either is correct, and JPEG compression would corrupt
 * the exact-pixel checks this test relies on):
 *
 * 1. [applyOrientation]'s geometry, checked pixel-by-pixel against coordinate
 *    formulas transcribed directly from the EXIF spec's textual definitions
 *    ("mirror horizontal", "rotate 90 CW", ...) - a genuinely independent
 *    derivation from the `AffineTransform` matrices under test, not the same
 *    reasoning checked against itself.
 * 2. [parseExifOrientation]'s byte parsing, against hand-built TIFF/IFD bytes
 *    in both byte orders - no image codec involved at all.
 * 3. One real, lossily-compressed JPEG through the public [ImageDecoder.decode]
 *    entry point, checking only that width/height end up swapped - a claim
 *    JPEG compression cannot corrupt, so it safely closes the loop that 1 and
 *    2 are actually wired together.
 */
class ImageDecoderTest {

    private val w = 5
    private val h = 3

    /**
     * Every pixel's colour encodes its own (x, y). An axis swap, an extra
     * flip, or an off-by-one at the boundary moves a specific, checkable
     * colour to the wrong place, rather than producing an image that merely
     * *looks* plausible.
     */
    private fun markerImage(width: Int = w, height: Int = h): BufferedImage {
        val img = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        for (y in 0 until height) for (x in 0 until width) {
            img.setRGB(x, y, (x shl 16) or (y shl 8) or 0x11)
        }
        return img
    }

    // ------------------------------------------------------- geometry

    /**
     * Where source pixel (sx, sy) of a [srcW]x[srcH] image lands after
     * correcting [orientation], per the EXIF spec's own textual definitions -
     * composed from primitive mirror/rotate operations, not from the matrix
     * table in [ImageDecoder.applyOrientation].
     */
    private fun expected(orientation: Int, sx: Int, sy: Int, srcW: Int, srcH: Int): Pair<Int, Int> {
        fun mirrorH(x: Int, y: Int) = (srcW - 1 - x) to y
        fun mirrorV(x: Int, y: Int) = x to (srcH - 1 - y)
        fun rotate180(x: Int, y: Int) = (srcW - 1 - x) to (srcH - 1 - y)
        fun rotate90cw(x: Int, y: Int) = (srcH - 1 - y) to x       // dims become srcH x srcW
        fun rotate270cw(x: Int, y: Int) = y to (srcW - 1 - x)      // dims become srcH x srcW

        return when (orientation) {
            1 -> sx to sy
            2 -> mirrorH(sx, sy)
            3 -> rotate180(sx, sy)
            4 -> mirrorV(sx, sy)
            // EXIF orientation 5 is "mirror horizontal, then rotate 270 CW" -
            // not 90 CW. (Orientation 7 is the 90-CW pairing, below.) Mixing
            // these up was caught here, by hand-verifying this reference
            // formula against ImageDecoder's AffineTransform matrices
            // independently before trusting either.
            5 -> { val (mx, my) = mirrorH(sx, sy); rotate270cw(mx, my) }
            6 -> rotate90cw(sx, sy)
            7 -> { val (mx, my) = mirrorH(sx, sy); rotate90cw(mx, my) }
            8 -> rotate270cw(sx, sy)
            else -> error("orientation must be 1..8")
        }
    }

    private fun expectedDims(orientation: Int, srcW: Int, srcH: Int) =
        if (orientation >= 5) srcH to srcW else srcW to srcH

    @ParameterizedTest
    @ValueSource(ints = [1, 2, 3, 4, 5, 6, 7, 8])
    fun `orientation correction moves every pixel exactly where the EXIF spec says`(orientation: Int) {
        val src = markerImage()
        val out = ImageDecoder.applyOrientation(src, orientation)

        val (expW, expH) = expectedDims(orientation, w, h)
        assertThat(out.width).isEqualTo(expW)
        assertThat(out.height).isEqualTo(expH)

        for (sy in 0 until h) for (sx in 0 until w) {
            val (dx, dy) = expected(orientation, sx, sy, w, h)
            val gotColour = out.getRGB(dx, dy) and 0xFFFFFF
            val wantColour = (sx shl 16) or (sy shl 8) or 0x11
            assertWithMessage("orientation $orientation: source ($sx,$sy) -> dest ($dx,$dy)")
                .that(gotColour)
                .isEqualTo(wantColour)
        }
    }

    @Test
    fun `orientation 1 and out-of-range values are a no-op`() {
        val src = markerImage()
        assertThat(ImageDecoder.applyOrientation(src, 1)).isSameInstanceAs(src)
        assertThat(ImageDecoder.applyOrientation(src, 0)).isSameInstanceAs(src)
        assertThat(ImageDecoder.applyOrientation(src, 9)).isSameInstanceAs(src)
    }

    // ------------------------------------------------------------ EXIF

    /** Hand-built TIFF/IFD0 bytes: one entry, tag 0x0112 (Orientation), type SHORT. */
    private fun exifPayload(orientation: Int, bigEndian: Boolean): ByteArray {
        fun u16(v: Int) = if (bigEndian) byteArrayOf((v shr 8).toByte(), v.toByte())
        else byteArrayOf(v.toByte(), (v shr 8).toByte())
        fun u32(v: Int) = if (bigEndian)
            byteArrayOf((v shr 24).toByte(), (v shr 16).toByte(), (v shr 8).toByte(), v.toByte())
        else byteArrayOf(v.toByte(), (v shr 8).toByte(), (v shr 16).toByte(), (v shr 24).toByte())

        val header = (if (bigEndian) "MM".toByteArray() else "II".toByteArray()) + u16(42) + u32(8)
        val entry = u16(0x0112) + u16(3 /* SHORT */) + u32(1) + u16(orientation) + u16(0)
        val ifd0 = u16(1 /* one entry */) + entry + u32(0 /* no next IFD */)
        return "Exif\u0000\u0000".toByteArray(Charsets.US_ASCII) + header + ifd0
    }

    @ParameterizedTest
    @ValueSource(ints = [1, 2, 3, 4, 5, 6, 7, 8])
    fun `parseExifOrientation reads the tag in both byte orders`(orientation: Int) {
        assertThat(ImageDecoder.parseExifOrientation(exifPayload(orientation, bigEndian = true)))
            .isEqualTo(orientation)
        assertThat(ImageDecoder.parseExifOrientation(exifPayload(orientation, bigEndian = false)))
            .isEqualTo(orientation)
    }

    @Test
    fun `a non-Exif APP1 payload is not mistaken for one`() {
        assertThat(ImageDecoder.parseExifOrientation("JFXX-not-exif-data".toByteArray())).isNull()
        assertThat(ImageDecoder.parseExifOrientation(ByteArray(3))).isNull()
    }

    @Test
    fun `readJpegExifOrientation walks past an unrelated marker to find APP1`(@TempDir tmp: File) {
        // A minimal but real JPEG marker stream: SOI, an unrelated APP0
        // segment the reader must skip via its declared length (not parse),
        // our APP1/Exif segment, then EOI. No pixel data is needed because
        // the reader never reaches the scan.
        val app0Body = byteArrayOf(0x00, 0x00) // contents irrelevant; must be skipped, not parsed
        val app0Len = app0Body.size + 2
        val payload = exifPayload(orientation = 6, bigEndian = true)
        val app1Len = payload.size + 2
        val bytes = byteArrayOf(0xFF.toByte(), 0xD8.toByte()) +               // SOI
            byteArrayOf(0xFF.toByte(), 0xE0.toByte()) +                       // APP0 marker
            byteArrayOf((app0Len shr 8).toByte(), app0Len.toByte()) + app0Body +
            byteArrayOf(0xFF.toByte(), 0xE1.toByte()) +                       // APP1 marker
            byteArrayOf((app1Len shr 8).toByte(), app1Len.toByte()) +         // segment length
            payload +
            byteArrayOf(0xFF.toByte(), 0xD9.toByte())                        // EOI
        val f = File(tmp, "synthetic.jpg").apply { writeBytes(bytes) }

        assertThat(ImageDecoder.readJpegExifOrientation(f)).isEqualTo(6)
    }

    @Test
    fun `a PNG has no EXIF and reads as orientation-normal`(@TempDir tmp: File) {
        val f = File(tmp, "plain.png")
        ImageIO.write(markerImage(), "png", f)
        assertThat(ImageDecoder.readJpegExifOrientation(f)).isEqualTo(1)
    }

    // ------------------------------------------------------ end to end

    private fun writeJpegWithOrientation(file: File, orientation: Int) {
        val writer = ImageIO.getImageWritersByFormatName("jpeg").next()
        val params = writer.defaultWriteParam.apply {
            compressionMode = ImageWriteParam.MODE_EXPLICIT
            compressionQuality = 0.9f
        }
        file.outputStream().use { fos ->
            ImageIO.createImageOutputStream(fos).use { ios ->
                writer.output = ios
                writer.write(null, IIOImage(markerImage(), null, null), params)
            }
        }
        writer.dispose()

        // Splice a synthetic Exif APP1 segment in right after the SOI marker,
        // exactly where a real camera places it - the same layout exercised
        // by the marker-walking test above, but now through a real, lossily
        // compressed JPEG file decoded via ImageIO.
        val original = file.readBytes()
        val payload = exifPayload(orientation, bigEndian = true)
        val segLen = payload.size + 2
        val app1 = byteArrayOf(0xFF.toByte(), 0xE1.toByte()) +
            byteArrayOf((segLen shr 8).toByte(), segLen.toByte()) + payload
        file.writeBytes(original.copyOfRange(0, 2) + app1 + original.copyOfRange(2, original.size))
    }

    @Test
    fun `decode applies EXIF rotation end to end`(@TempDir tmp: File) {
        val f = File(tmp, "rotated.jpg")
        writeJpegWithOrientation(f, orientation = 6) // 90 CW: dimensions swap

        val decoded = ImageDecoder.decode(f)

        // JPEG is lossy, so exact pixel colours are not asserted here - only
        // the claim compression cannot alter: which dimension ended up which.
        assertThat(decoded.width).isEqualTo(h)
        assertThat(decoded.height).isEqualTo(w)
    }

    @Test
    fun `decode without EXIF leaves dimensions unchanged`(@TempDir tmp: File) {
        val f = File(tmp, "plain.png")
        ImageIO.write(markerImage(), "png", f)

        val decoded = ImageDecoder.decode(f)

        assertThat(decoded.width).isEqualTo(w)
        assertThat(decoded.height).isEqualTo(h)
    }

    @Test
    fun `a corrupt file fails with a clear message, not a crash`(@TempDir tmp: File) {
        val f = File(tmp, "not-an-image.jpg").apply { writeBytes(byteArrayOf(1, 2, 3, 4, 5)) }
        val ex = org.junit.jupiter.api.assertThrows<ImageDecoder.DecodeException> {
            ImageDecoder.decode(f)
        }
        assertThat(ex.message).contains("not-an-image.jpg")
    }
}
