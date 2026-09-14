package com.fisherwiki.cli

import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import javax.imageio.ImageIO

/**
 * Decoding photographs into the ARGB pixels `:core` expects - the desktop
 * counterpart to the Android app's `ImageLoading.kt`.
 *
 * `Preprocessor.toTensor`'s own kdoc says exactly what is required: packed
 * `0xAARRGGBB` pixels, row-major, "exactly what `Bitmap.getPixels` gives on
 * Android and `BufferedImage.getRGB` gives on the JVM" - so decoding itself is
 * a one-liner. What is not a one-liner, and is easy to get silently wrong, is
 * **EXIF orientation**: phone cameras almost always write landscape sensor data
 * plus a rotation tag, training images were decoded with orientation applied
 * (PIL's `exif_transpose`), and a JPG straight off a phone that skips this step
 * feeds the model a photograph rotated 90 degrees from anything it saw in
 * training - with no error anywhere, just quietly worse accuracy. This mirrors
 * that step for the desktop JVM, which does not do it automatically.
 */
object ImageDecoder {

    data class Decoded(val pixels: IntArray, val width: Int, val height: Int)

    class DecodeException(message: String, cause: Throwable? = null) : Exception(message, cause)

    fun decode(file: File): Decoded {
        val raw = try {
            ImageIO.read(file)
        } catch (e: IOException) {
            throw DecodeException("could not read ${file.name}: ${e.message}", e)
        } ?: throw DecodeException(
            "${file.name}: not a recognised image format (or the file is corrupt)"
        )

        val orientation = readOrientation(file)
        val oriented = applyOrientation(raw, orientation)

        val w = oriented.width
        val h = oriented.height
        val pixels = IntArray(w * h)
        // getRGB always returns packed ARGB regardless of the source image's
        // internal raster (grayscale, indexed palette, etc.) - ImageIO
        // normalises that during decode, so this is correct for any JPG/PNG.
        oriented.getRGB(0, 0, w, h, pixels, 0, w)
        return Decoded(pixels, w, h)
    }

    // ----------------------------------------------------------- EXIF

    /**
     * EXIF `Orientation` tag (0x0112), or 1 (normal / no-op) when absent,
     * unreadable, or not a JPEG. PNG carries no EXIF orientation in normal use,
     * so that absence is expected, not an error - same convention as Android's
     * `ExifInterface.ORIENTATION_NORMAL` default.
     */
    private fun readOrientation(file: File): Int =
        runCatching { readJpegExifOrientation(file) }.getOrDefault(1)

    /**
     * Minimal hand-rolled EXIF reader: walks JPEG markers looking for the APP1
     * segment, then reads exactly one TIFF IFD entry (tag 0x0112) out of it.
     * Deliberately narrow - this is not a general EXIF library, it reads one
     * integer and nothing else, bounded by the segment's own declared length.
     */
    internal fun readJpegExifOrientation(file: File): Int {
        RandomAccessFile(file, "r").use { raf ->
            if (raf.length() < 4) return 1
            val soi = ByteArray(2).also { raf.readFully(it) }
            if (soi[0] != 0xFF.toByte() || soi[1] != 0xD8.toByte()) return 1 // not a JPEG

            while (raf.filePointer < raf.length() - 4) {
                val marker = raf.readUnsignedByte()
                if (marker != 0xFF) return 1 // desynced; give up quietly
                val type = raf.readUnsignedByte()
                if (type == 0xD8 || type == 0x01 || type in 0xD0..0xD7) continue // no-length markers
                if (type == 0xDA || type == 0xD9) return 1 // start-of-scan/EOI: no more markers before pixel data
                val segLen = raf.readUnsignedShort()
                if (segLen < 2) return 1
                if (type == 0xE1) { // APP1: candidate for Exif
                    val body = ByteArray(segLen - 2)
                    raf.readFully(body)
                    parseExifOrientation(body)?.let { return it }
                    continue
                }
                raf.seek(raf.filePointer + (segLen - 2))
            }
            return 1
        }
    }

    /** [app1] is the APP1 segment payload, starting right after its length field. */
    internal fun parseExifOrientation(app1: ByteArray): Int? {
        // "Exif\0\0" preamble.
        if (app1.size < 8) return null
        val preamble = String(app1, 0, 4, Charsets.US_ASCII)
        if (preamble != "Exif") return null
        val tiff = 6 // offset of the TIFF header within app1
        if (app1.size < tiff + 8) return null

        val bigEndian = when {
            app1[tiff] == 'M'.code.toByte() && app1[tiff + 1] == 'M'.code.toByte() -> true
            app1[tiff] == 'I'.code.toByte() && app1[tiff + 1] == 'I'.code.toByte() -> false
            else -> return null
        }
        fun u16(off: Int): Int {
            val a = app1[off].toInt() and 0xFF
            val b = app1[off + 1].toInt() and 0xFF
            return if (bigEndian) (a shl 8) or b else (b shl 8) or a
        }
        fun u32(off: Int): Int {
            val a = app1[off].toInt() and 0xFF
            val b = app1[off + 1].toInt() and 0xFF
            val c = app1[off + 2].toInt() and 0xFF
            val d = app1[off + 3].toInt() and 0xFF
            return if (bigEndian) (a shl 24) or (b shl 16) or (c shl 8) or d
            else (d shl 24) or (c shl 16) or (b shl 8) or a
        }

        val ifd0Offset = tiff + u32(tiff + 4)
        if (ifd0Offset < 0 || ifd0Offset + 2 > app1.size) return null
        val entryCount = u16(ifd0Offset)
        for (i in 0 until entryCount) {
            val entryOff = ifd0Offset + 2 + i * 12
            if (entryOff + 12 > app1.size) break
            val tag = u16(entryOff)
            if (tag == 0x0112) {
                // type SHORT, count 1: value sits in the first 2 bytes of the
                // 4-byte value field, at entryOff + 8.
                return u16(entryOff + 8).takeIf { it in 1..8 }
            }
        }
        return null
    }

    // ------------------------------------------------------- geometry

    /**
     * Apply an EXIF orientation code to [img], returning a new image whose
     * pixels are in display order (orientation 1 thereafter).
     *
     * The eight `AffineTransform` matrices below are the standard EXIF
     * orientation-correction table - not derived ad hoc. Orientations 5-8 swap
     * width and height, which is why the destination canvas size depends on
     * [orientation].
     */
    internal fun applyOrientation(img: BufferedImage, orientation: Int): BufferedImage {
        if (orientation <= 1 || orientation > 8) return img
        val w = img.width
        val h = img.height
        val swapped = orientation >= 5
        val out = BufferedImage(
            if (swapped) h else w, if (swapped) w else h, BufferedImage.TYPE_INT_ARGB
        )
        val t = when (orientation) {
            2 -> AffineTransform(-1.0, 0.0, 0.0, 1.0, w.toDouble(), 0.0) // flip horizontal
            3 -> AffineTransform(-1.0, 0.0, 0.0, -1.0, w.toDouble(), h.toDouble()) // 180
            4 -> AffineTransform(1.0, 0.0, 0.0, -1.0, 0.0, h.toDouble()) // flip vertical
            5 -> AffineTransform(0.0, 1.0, 1.0, 0.0, 0.0, 0.0) // transpose
            6 -> AffineTransform(0.0, 1.0, -1.0, 0.0, h.toDouble(), 0.0) // 90 CW
            7 -> AffineTransform(0.0, -1.0, -1.0, 0.0, h.toDouble(), w.toDouble()) // transverse
            8 -> AffineTransform(0.0, -1.0, 1.0, 0.0, 0.0, w.toDouble()) // 90 CCW
            else -> AffineTransform() // unreachable, orientation in 2..8 above
        }
        val g = out.createGraphics()
        try {
            g.drawImage(img, t, null)
        } finally {
            g.dispose()
        }
        return out
    }
}
