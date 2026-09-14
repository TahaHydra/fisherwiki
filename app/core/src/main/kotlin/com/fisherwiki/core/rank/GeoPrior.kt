package com.fisherwiki.core.rank

import java.io.DataInputStream
import java.io.File
import java.io.IOException
import kotlin.math.ln

/**
 * Offline geographic plausibility prior.
 *
 * ## What this is for
 * Location genuinely disambiguates fish: a *Micropterus salmoides* and a
 * *Micropterus dolomieu* look similar, but a photo taken in a Finnish lake is
 * neither. Used carefully, a prior converts an ambiguous visual result into a
 * confident one.
 *
 * ## What this must never do
 * Override the eyes. A rare vagrant, an escapee, an introduced population and a
 * stocked pond all exist, and an angler who has actually caught something
 * unusual is exactly the user who most needs the app to be right. So:
 *
 * - the prior is applied as a **bounded multiplier**, clamped to
 *   [MIN_FACTOR]..[MAX_FACTOR], never as a hard filter;
 * - a species with **no occurrence data at all** in a cell gets
 *   [UNKNOWN_FACTOR] (slightly below neutral), not zero - absence of evidence
 *   in a citizen-science dataset is weak evidence of absence;
 * - if the visual model is highly confident the prior is attenuated further,
 *   via [confidenceAttenuation];
 * - everything works with no location at all, in which case every factor is
 *   exactly 1.0 and ranking is purely visual.
 *
 * ## Data structure
 * A quantised occurrence histogram over equal-angle cells, defaulting to 2°.
 * Bounding boxes were rejected: a box around Europe also contains the Sahara,
 * and would tell a user that a brown trout is plausible there. Cells are built
 * from the same observation coordinates that produced the training corpus, so
 * the prior describes where the species is *photographed*, which is the right
 * distribution for this task.
 *
 * The on-disk format is a compact binary blob; see `docs/OFFLINE_PACK_FORMAT.md`.
 */
class GeoPrior private constructor(
    private val cellDegrees: Double,
    private val numClasses: Int,
    /** classIndex -> (packedCell -> quantised log-frequency 0..255). */
    private val cells: Array<Map<Int, Byte>>,
    /** Per-class total observation count, for confidence weighting. */
    private val classTotals: IntArray,
) {

    companion object {
        const val MAGIC = 0x46574750  // "FWGP"
        const val VERSION = 1

        /** Strongest boost the prior may apply to a species. */
        const val MAX_FACTOR = 3.0f

        /** Strongest penalty. Deliberately not near zero. */
        const val MIN_FACTOR = 0.12f

        /** Applied when we have no occurrence data for this class anywhere near. */
        const val UNKNOWN_FACTOR = 0.75f

        /** Neighbourhood radius in cells; smooths hard cell edges. */
        const val NEIGHBOURHOOD = 1

        /** An empty prior: every factor is 1.0. Used when a pack ships none. */
        fun neutral(numClasses: Int): GeoPrior =
            GeoPrior(2.0, numClasses, Array(numClasses) { emptyMap() }, IntArray(numClasses))

        /**
         * Read the binary prior.
         *
         * Format is **big-endian**, matching [DataInputStream]'s network byte
         * order. The Python writer in `tools/fwdata/geoprior.py` uses struct
         * '>' for the same reason; a mismatch here would parse without error
         * and yield nonsense, so a round-trip test pins the two together.
         *
         * ```
         * u32 magic, u32 version, f32 cellDegrees, u32 numClasses
         * repeat numClasses:
         *   u32 classTotal, u32 cellCount, repeat cellCount: i32 packedCell, u8 value
         * ```
         *
         * A cell absent from the file means *unknown*, not *absent*; the
         * reader maps it to [UNKNOWN_FACTOR] rather than zero.
         */
        fun read(file: File, expectedClasses: Int): GeoPrior {
            DataInputStream(file.inputStream().buffered()).use { input ->
                val magic = input.readInt()
                if (magic != MAGIC) throw IOException("not a geo prior (magic $magic)")
                val version = input.readInt()
                if (version != VERSION) throw IOException("geo prior version $version unsupported")
                val cellDeg = input.readFloat().toDouble()
                if (cellDeg <= 0.0 || cellDeg > 90.0) {
                    throw IOException("implausible cell size $cellDeg")
                }
                val n = input.readInt()
                if (n != expectedClasses) {
                    throw IOException("geo prior has $n classes, model has $expectedClasses")
                }
                val totals = IntArray(n)
                val maps = Array<Map<Int, Byte>>(n) { emptyMap() }
                for (c in 0 until n) {
                    totals[c] = input.readInt()
                    val count = input.readInt()
                    if (count < 0 || count > 5_000_000) {
                        throw IOException("implausible cell count $count for class $c")
                    }
                    if (count == 0) continue
                    val m = HashMap<Int, Byte>(count * 2)
                    repeat(count) {
                        val cell = input.readInt()
                        val v = input.readByte()
                        m[cell] = v
                    }
                    maps[c] = m
                }
                return GeoPrior(cellDeg, n, maps, totals)
            }
        }
    }

    /** Pack a lat/lon into a single int cell key. */
    fun cellOf(latitude: Double, longitude: Double): Int {
        val lat = latitude.coerceIn(-90.0, 90.0)
        var lon = longitude
        // Normalise longitude into [-180, 180) so the antimeridian does not
        // produce two disjoint cell families for one place.
        while (lon < -180.0) lon += 360.0
        while (lon >= 180.0) lon -= 360.0
        val latIdx = ((lat + 90.0) / cellDegrees).toInt()
        val lonIdx = ((lon + 180.0) / cellDegrees).toInt()
        return (latIdx shl 16) or (lonIdx and 0xFFFF)
    }

    /**
     * Multiplier for [classIndex] at a location, or 1.0 when location is absent.
     *
     * @param visualConfidence calibrated top-1 probability from the model; a
     *   very confident visual result attenuates the prior so that a genuinely
     *   unusual catch is not argued away.
     */
    fun factor(
        classIndex: Int,
        latitude: Double?,
        longitude: Double?,
        visualConfidence: Float = 0f,
    ): Float {
        if (latitude == null || longitude == null) return 1f
        if (classIndex !in 0 until numClasses) return 1f
        val map = cells[classIndex]
        if (map.isEmpty()) return attenuate(UNKNOWN_FACTOR, visualConfidence)

        val latIdx = ((latitude.coerceIn(-90.0, 90.0) + 90.0) / cellDegrees).toInt()
        var lon = longitude
        while (lon < -180.0) lon += 360.0
        while (lon >= 180.0) lon -= 360.0
        val lonIdx = ((lon + 180.0) / cellDegrees).toInt()
        val lonCells = (360.0 / cellDegrees).toInt()

        // Take the best value in a small neighbourhood so a coordinate that
        // lands just across a cell boundary from a dense cell is not penalised.
        var bestValue = 0
        for (dy in -NEIGHBOURHOOD..NEIGHBOURHOOD) {
            for (dx in -NEIGHBOURHOOD..NEIGHBOURHOOD) {
                val ly = latIdx + dy
                if (ly < 0) continue
                val lx = ((lonIdx + dx) % lonCells + lonCells) % lonCells
                val key = (ly shl 16) or (lx and 0xFFFF)
                val v = map[key]?.toInt()?.and(0xFF) ?: continue
                // Neighbouring cells count for less than the exact one.
                val discounted = if (dx == 0 && dy == 0) v else (v * 2) / 3
                if (discounted > bestValue) bestValue = discounted
            }
        }

        if (bestValue == 0) return attenuate(UNKNOWN_FACTOR, visualConfidence)

        // Stored value is a quantised log-frequency in 0..255. Map it onto the
        // allowed multiplier range on a log scale so that the difference
        // between "very common here" and "common here" is small, while the
        // difference between "present" and "absent" is meaningful.
        val t = bestValue / 255f
        val raw = MIN_FACTOR * Math.pow(
            (MAX_FACTOR / MIN_FACTOR).toDouble(), t.toDouble()
        ).toFloat()
        return attenuate(raw.coerceIn(MIN_FACTOR, MAX_FACTOR), visualConfidence)
    }

    /**
     * Shrink the prior towards 1.0 as the visual model becomes more confident.
     *
     * At 0.95 visual confidence the prior retains only ~25% of its effect in
     * log space, so a clear photograph of an out-of-range fish still wins.
     */
    private fun attenuate(factor: Float, visualConfidence: Float): Float {
        if (visualConfidence <= 0.5f) return factor
        val strength = (1f - (visualConfidence - 0.5f) / 0.5f).coerceIn(0f, 1f)
        val logF = ln(factor.toDouble())
        return Math.exp(logF * strength).toFloat()
    }

    /** True when we hold no occurrence data at all for this class. */
    fun isUnknown(classIndex: Int): Boolean =
        classIndex !in 0 until numClasses || cells[classIndex].isEmpty()

    fun observationCount(classIndex: Int): Int =
        if (classIndex in 0 until numClasses) classTotals[classIndex] else 0
}
