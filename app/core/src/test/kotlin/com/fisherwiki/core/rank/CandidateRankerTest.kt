package com.fisherwiki.core.rank

import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.model.Rank
import com.fisherwiki.core.model.Taxon
import com.fisherwiki.core.pack.CalibrationSpec
import com.google.common.truth.Truth.assertThat
import kotlin.math.ln
import org.junit.jupiter.api.Test

/**
 * These tests encode the product's central promise:
 *
 * > The dangerous failure is not "I don't know this fish". The dangerous
 * > failure is "this is definitely species X" when the model does not know.
 *
 * So most of what is asserted here is about *refusing* to answer, and about the
 * geographic prior never being allowed to overrule clear visual evidence.
 */
class CandidateRankerTest {

    private val spec = CalibrationSpec(
        temperature = 1f,
        unknownThreshold = 0.35f,
        marginThreshold = 0.08f,
        entropyThreshold = 0.85f,
    )

    /** Four classes: two Perca, one Sander, one Scorpaena. */
    private val names = listOf(
        "Perca fluviatilis", "Perca flavescens", "Sander lucioperca", "Scorpaena porcus"
    )
    private val genusOfClass = intArrayOf(0, 0, 1, 2)
    private val genusNames = arrayOf("Perca", "Sander", "Scorpaena")

    private fun taxon(i: Int) = Taxon(
        id = 1000L + i,
        scientificName = names[i],
        rank = Rank.SPECIES,
        genus = names[i].substringBefore(' '),
        family = if (i < 3) "Percidae" else "Scorpaenidae",
    )

    private fun ranker(geo: GeoPrior? = null) = CandidateRanker(
        calibration = spec,
        geoPrior = geo,
        taxonResolver = { i -> if (i in names.indices) taxon(i) else null },
        genusOfClass = genusOfClass,
        genusName = { g -> genusNames.getOrNull(g) },
    )

    // ------------------------------------------------------------ confident

    @Test
    fun `clear winner is reported confidently`() {
        val id = ranker().rank(floatArrayOf(9f, 1f, 0f, -1f))
        assertThat(id.certainty).isEqualTo(Certainty.CONFIDENT)
        assertThat(id.best).isNotNull()
        assertThat(id.best!!.taxon.scientificName).isEqualTo("Perca fluviatilis")
        assertThat(id.best.probability).isGreaterThan(0.9f)
        assertThat(id.isUncertain).isFalse()
    }

    @Test
    fun `alternatives are ranked below the best and included`() {
        val id = ranker().rank(floatArrayOf(9f, 5f, 2f, 0f), topK = 3)
        assertThat(id.alternatives).hasSize(2)
        assertThat(id.alternatives[0].probability).isLessThan(id.best!!.probability)
        assertThat(id.alternatives[0].probability)
            .isAtLeast(id.alternatives[1].probability)
    }

    // ------------------------------------------------------------ uncertain

    @Test
    fun `two near identical species produce a genus level answer not a guess`() {
        // The classic confusable pair: the model is sure it is a Perca and has
        // no idea which one. Naming either would be the dangerous failure.
        val id = ranker().rank(floatArrayOf(5.0f, 4.95f, -2f, -3f))
        assertThat(id.best).isNull()
        assertThat(id.certainty).isEqualTo(Certainty.COARSE_ONLY)
        assertThat(id.coarseFallback).isNotNull()
        assertThat(id.coarseFallback!!.taxon.scientificName).isEqualTo("Perca")
        assertThat(id.coarseFallback.taxon.rank).isEqualTo(Rank.GENUS)
        assertThat(id.coarseFallback.probability).isGreaterThan(0.9f)
        assertThat(id.isUncertain).isTrue()
        // The species candidates are still offered for a human to compare.
        assertThat(id.alternatives.map { it.taxon.scientificName })
            .containsAtLeast("Perca fluviatilis", "Perca flavescens")
    }

    @Test
    fun `mass spread across unrelated genera yields unknown not a coarse guess`() {
        // Nothing coherent to say: the model is mildly attracted to everything.
        val id = ranker().rank(floatArrayOf(1.0f, 0.9f, 1.05f, 0.95f))
        assertThat(id.certainty).isEqualTo(Certainty.UNKNOWN)
        assertThat(id.best).isNull()
        assertThat(id.coarseFallback).isNull()
    }

    @Test
    fun `low confidence across many classes is unknown`() {
        val id = ranker().rank(FloatArray(4) { 0.01f * it })
        assertThat(id.isUncertain).isTrue()
    }

    @Test
    fun `unknown identification exposes no species to render`() {
        // The type makes the mistake impossible: there is no `best` to show.
        val id = ranker().rank(floatArrayOf(1f, 1f, 1f, 1f))
        assertThat(id.best).isNull()
        assertThat(id.certainty).isEqualTo(Certainty.UNKNOWN)
    }

    // ----------------------------------------------------------- geo prior

    // GeoPrior's constructor is private and it is loaded from a pack file, so
    // the behaviour of a *populated* prior is covered by GeoPriorRoundTripTest,
    // which writes a real file with the Python format and reads it back here.
    // The tests below cover the invariants that must hold for any prior.

    @Test
    fun `geo factor is neutral when no location is supplied`() {
        val id = ranker().rank(floatArrayOf(9f, 1f, 0f, -1f), location = null)
        assertThat(id.geoApplied).isFalse()
        assertThat(id.ranked.all { it.geoFactor == 1f }).isTrue()
    }

    @Test
    fun `geo prior never eliminates a species`() {
        // Documented invariant: the bounded multiplier cannot reach zero, so a
        // vagrant or an escapee stays identifiable.
        val neutral = GeoPrior.neutral(4)
        for (c in 0 until 4) {
            val f = neutral.factor(c, 50.0, 5.0, 0f)
            assertThat(f).isGreaterThan(0f)
        }
        assertThat(GeoPrior.MIN_FACTOR).isGreaterThan(0f)
        assertThat(GeoPrior.UNKNOWN_FACTOR).isGreaterThan(0f)
        assertThat(GeoPrior.UNKNOWN_FACTOR).isLessThan(1f)
    }

    @Test
    fun `empty prior treats every class as unknown rather than absent`() {
        val neutral = GeoPrior.neutral(4)
        assertThat(neutral.isUnknown(0)).isTrue()
        // Unknown must be a mild penalty, not exclusion.
        val f = neutral.factor(0, 50.0, 5.0, 0f)
        assertThat(f).isWithin(1e-4f).of(GeoPrior.UNKNOWN_FACTOR)
    }

    @Test
    fun `high visual confidence attenuates the prior towards neutral`() {
        val neutral = GeoPrior.neutral(4)
        val weakEvidence = neutral.factor(0, 50.0, 5.0, visualConfidence = 0.0f)
        val strongEvidence = neutral.factor(0, 50.0, 5.0, visualConfidence = 0.97f)
        // A confident look at the fish should pull the factor back toward 1.0.
        assertThat(kotlin.math.abs(1f - strongEvidence))
            .isLessThan(kotlin.math.abs(1f - weakEvidence))
    }

    @Test
    fun `out of range coordinates do not throw`() {
        val neutral = GeoPrior.neutral(4)
        for (lon in listOf(-400.0, 400.0, 179.99, -180.0)) {
            assertThat(neutral.factor(0, 45.0, lon, 0f)).isGreaterThan(0f)
        }
        assertThat(neutral.factor(0, 95.0, 0.0, 0f)).isGreaterThan(0f)
    }

    // ------------------------------------------------------------- fusion

    @Test
    fun `fusing identical photographs matches the single photo result`() {
        val logits = floatArrayOf(8f, 2f, 1f, 0f)
        val single = ranker().rank(logits)
        val fused = ranker().fuse(listOf(logits, logits, logits))
        assertThat(fused.best!!.taxon.scientificName)
            .isEqualTo(single.best!!.taxon.scientificName)
        assertThat(fused.photoCount).isEqualTo(3)
    }

    @Test
    fun `one confidently wrong frame does not overturn two agreeing frames`() {
        // This is why fusion averages in log space: with an arithmetic mean of
        // probabilities a single 0.999 outlier wins outright.
        val good = floatArrayOf(6f, 1f, 0f, 0f)      // says class 0
        val outlier = floatArrayOf(0f, 12f, 0f, 0f)  // says class 1, very loudly
        val fused = ranker().fuse(listOf(good, good, outlier))
        val winner = fused.best?.taxon?.scientificName
            ?: fused.coarseFallback?.taxon?.scientificName
        assertThat(winner).isAnyOf("Perca fluviatilis", "Perca")
    }

    @Test
    fun `uninformative frames are down weighted but not discarded`() {
        val informative = floatArrayOf(7f, 1f, 0f, 0f)
        val uniform = floatArrayOf(0f, 0f, 0f, 0f)
        val fused = ranker().fuse(listOf(informative, uniform))
        // The informative frame should still dominate.
        assertThat(fused.best).isNotNull()
        assertThat(fused.best!!.taxon.scientificName).isEqualTo("Perca fluviatilis")
    }

    @Test
    fun `fusing a single photograph is identical to ranking it`() {
        val logits = floatArrayOf(5f, 2f, 1f, 0f)
        val a = ranker().rank(logits)
        val b = ranker().fuse(listOf(logits))
        assertThat(b.best!!.probability).isWithin(1e-5f).of(a.best!!.probability)
    }

    @Test
    fun `fusion requires consistent class counts`() {
        val e = runCatching {
            ranker().fuse(listOf(floatArrayOf(1f, 2f, 3f, 4f), floatArrayOf(1f, 2f)))
        }.exceptionOrNull()
        assertThat(e).isInstanceOf(IllegalArgumentException::class.java)
    }

    // -------------------------------------------------------------- misc

    @Test
    fun `pack identity is carried through to the result`() {
        val id = ranker().rank(
            floatArrayOf(9f, 1f, 0f, -1f),
            packId = "europe_freshwater",
            packVersion = 3,
            inferenceMillis = 42,
        )
        assertThat(id.packId).isEqualTo("europe_freshwater")
        assertThat(id.packVersion).isEqualTo(3)
        assertThat(id.inferenceMillis).isEqualTo(42)
    }

    @Test
    fun `unresolvable classes are skipped rather than rendered blank`() {
        val r = CandidateRanker(
            calibration = spec,
            taxonResolver = { null },          // pack database has no rows
            genusOfClass = genusOfClass,
        )
        val id = r.rank(floatArrayOf(9f, 1f, 0f, -1f))
        assertThat(id.certainty).isEqualTo(Certainty.UNKNOWN)
        assertThat(id.best).isNull()
    }

    @Test
    fun `percent rendering rounds the calibrated probability`() {
        val id = ranker().rank(floatArrayOf(ln(9f), 0f, 0f, 0f))
        // 9 / (9+1+1+1) = 0.75
        assertThat(id.best!!.percent).isEqualTo(75)
    }
}
