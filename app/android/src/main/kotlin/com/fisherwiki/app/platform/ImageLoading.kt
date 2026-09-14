package com.fisherwiki.app.platform

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.fisherwiki.core.engine.IdentificationEngine
import java.io.File
import java.io.InputStream
import kotlin.math.max

/**
 * Decoding photographs into the ARGB pixels `:core` expects.
 *
 * Two details here are easy to get wrong and both cause silent accuracy loss
 * rather than an error:
 *
 * **EXIF orientation.** Phone cameras almost always write landscape sensor data
 * plus a rotation tag. Training images were decoded with orientation applied
 * (PIL's `exif_transpose`), so serving must apply it too, or every portrait
 * photograph arrives at the model rotated 90 degrees from anything it saw in
 * training.
 *
 * **Downsampling before the model's own resize.** We decode with `inSampleSize`
 * to roughly twice the model's input, not to full resolution: a 12 MP bitmap is
 * ~48 MB of heap for no benefit, since the next step shrinks it to 256px
 * anyway. Decoding to *exactly* the input size would be worse, because
 * `inSampleSize` only halves and the resulting aliasing differs from the
 * bilinear path used in training.
 */
object ImageLoading {

    /** Target short edge before `Preprocessor` does its own resize + crop. */
    private const val DECODE_TARGET = 640

    fun fromUri(context: Context, uri: Uri, targetShortEdge: Int = DECODE_TARGET): Decoded? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        } ?: return null
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        val opts = BitmapFactory.Options().apply {
            inSampleSize = sampleSizeFor(bounds.outWidth, bounds.outHeight, targetShortEdge)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        } ?: return null

        val orientation = context.contentResolver.openInputStream(uri)?.use {
            readOrientation(it)
        } ?: ExifInterface.ORIENTATION_NORMAL

        return decoded(applyOrientation(bitmap, orientation))
    }

    fun fromFile(file: File, targetShortEdge: Int = DECODE_TARGET): Decoded? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        val opts = BitmapFactory.Options().apply {
            inSampleSize = sampleSizeFor(bounds.outWidth, bounds.outHeight, targetShortEdge)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = BitmapFactory.decodeFile(file.absolutePath, opts) ?: return null
        val orientation = file.inputStream().use { readOrientation(it) }
        return decoded(applyOrientation(bitmap, orientation))
    }

    fun fromBitmap(bitmap: Bitmap): Decoded = decoded(bitmap)

    data class Decoded(val pixels: IntArray, val width: Int, val height: Int) {
        fun asPhoto() = IdentificationEngine.Photo(pixels, width, height)

        override fun equals(other: Any?): Boolean =
            other is Decoded && width == other.width && height == other.height &&
                pixels.contentEquals(other.pixels)

        override fun hashCode(): Int =
            31 * (31 * pixels.contentHashCode() + width) + height
    }

    private fun decoded(bitmap: Bitmap): Decoded {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        return Decoded(pixels, w, h)
    }

    private fun readOrientation(input: InputStream): Int =
        runCatching {
            ExifInterface(input).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
            )
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)

    private fun sampleSizeFor(width: Int, height: Int, targetShortEdge: Int): Int {
        val short = minOf(width, height)
        var sample = 1
        while (short / (sample * 2) >= targetShortEdge) sample *= 2
        return max(1, sample)
    }

    private fun applyOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                m.postRotate(90f); m.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                m.postRotate(270f); m.postScale(-1f, 1f)
            }
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }
}
