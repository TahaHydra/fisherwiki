package com.fisherwiki.cli

import com.fisherwiki.core.model.Candidate
import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.model.Identification
import com.fisherwiki.core.model.Rank
import com.fisherwiki.core.model.Taxon
import com.fisherwiki.core.pack.CalibrationSpec
import com.google.common.truth.Truth.assertThat
import com.google.common.truth.Truth.assertWithMessage
import org.junit.jupiter.api.Test

/**
 * `Report.explain` builds its sentences with `"a " + "b".format(x)`, which
 * Kotlin parses as `"a " + ("b".format(x))` - `.format` binds to the trailing
 * literal, not the whole concatenation. Get that wrong and a threshold value
 * silently prints as the literal text `%.4f` instead of a number: still valid
 * Kotlin, still runs, wrong forever. This file caught five instances of
 * exactly that while it was being written. [assertClean], called from every
 * test below, is the regression guard.
 */
class ReportTest {

    private val species = Taxon(1, "Perca fluviatilis", Rank.SPECIES, commonName = "European perch")
    private val runnerUp = Taxon(2, "Perca flavescens", Rank.SPECIES, commonName = "Yellow perch")
    private val genus = Taxon(-2, "Perca", Rank.GENUS, genus = "Perca")

    private fun candidate(taxon: Taxon, p: Float, classIndex: Int = 0) =
        Candidate(taxon = taxon, probability = p, visualProbability = p, classIndex = classIndex)

    /** A leftover `%.4f`, `%s`, `%.1f%%` etc. that survived unconsumed. */
    private val unsubstitutedPlaceholder = Regex("""%[-.0-9]*[sfd]""")

    private fun assertClean(text: String) {
        assertWithMessage("'$text' contains an unsubstituted format placeholder")
            .that(unsubstitutedPlaceholder.find(text))
            .isNull()
    }

    @Test
    fun `confident result explains as none, cleanly`() {
        val id = Identification(
            certainty = Certainty.CONFIDENT,
            best = candidate(species, 0.95f),
            alternatives = emptyList(),
            margin = 0.9f,
            normalizedEntropy = 0.05f,
        )
        val text = Report.explain(id, CalibrationSpec())
        assertClean(text)
        assertThat(text).startsWith("none")
    }

    @Test
    fun `ambiguous result cites the actual margin and threshold values`() {
        val cal = CalibrationSpec(marginThreshold = 0.08f)
        val id = Identification(
            certainty = Certainty.AMBIGUOUS,
            best = candidate(species, 0.6f),
            alternatives = listOf(candidate(runnerUp, 0.55f, classIndex = 1)),
            margin = 0.05f,
            normalizedEntropy = 0.4f,
        )
        val text = Report.explain(id, cal)
        assertClean(text)
        assertThat(text).contains("0.0500")   // %.4f of the actual margin
        assertThat(text).contains("0.2000")   // %.4f of 0.08 * 2.5
    }

    @Test
    fun `low confidence rejection cites the species name and both numbers`() {
        val cal = CalibrationSpec(unknownThreshold = 0.35f)
        val id = Identification(
            certainty = Certainty.UNKNOWN,
            best = null,
            alternatives = listOf(candidate(species, 0.20f)),
            margin = 0.5f,
            normalizedEntropy = 0.3f,
        )
        val text = Report.explain(id, cal)
        assertClean(text)
        assertThat(text).contains("LOW_CONFIDENCE")
        assertThat(text).contains("Perca fluviatilis")
        assertThat(text).contains("20.0%")
        assertThat(text).contains("35.0%")
    }

    @Test
    fun `narrow margin rejection cites both margin numbers`() {
        val cal = CalibrationSpec(unknownThreshold = 0.1f, marginThreshold = 0.08f)
        val id = Identification(
            certainty = Certainty.UNKNOWN,
            best = null,
            alternatives = listOf(candidate(species, 0.5f), candidate(runnerUp, 0.48f, classIndex = 1)),
            margin = 0.02f,
            normalizedEntropy = 0.3f,
        )
        val text = Report.explain(id, cal)
        assertClean(text)
        assertThat(text).contains("NARROW_MARGIN")
        assertThat(text).contains("0.0200")
        assertThat(text).contains("0.0800")
    }

    @Test
    fun `high entropy rejection cites both entropy numbers`() {
        val cal = CalibrationSpec(unknownThreshold = 0.1f, marginThreshold = 0.01f, entropyThreshold = 0.85f)
        val id = Identification(
            certainty = Certainty.UNKNOWN,
            best = null,
            alternatives = listOf(candidate(species, 0.3f)),
            margin = 0.2f,
            normalizedEntropy = 0.95f,
        )
        val text = Report.explain(id, cal)
        assertClean(text)
        assertThat(text).contains("HIGH_ENTROPY")
        assertThat(text).contains("0.9500")
        assertThat(text).contains("0.8500")
    }

    @Test
    fun `below class threshold rejection cites the species and its own threshold`() {
        val cal = CalibrationSpec(
            unknownThreshold = 0.1f,
            perClassThreshold = mapOf("0" to 0.9f),
        )
        val id = Identification(
            certainty = Certainty.UNKNOWN,
            best = null,
            alternatives = listOf(candidate(species, 0.5f, classIndex = 0)),
            margin = 0.3f,
            normalizedEntropy = 0.2f,
        )
        val text = Report.explain(id, cal)
        assertClean(text)
        assertThat(text).contains("BELOW_CLASS_THRESHOLD")
        assertThat(text).contains("Perca fluviatilis")
    }

    @Test
    fun `genus fallback explains both the species rejection and the genus success`() {
        val cal = CalibrationSpec(unknownThreshold = 0.5f)
        val id = Identification(
            certainty = Certainty.COARSE_ONLY,
            best = null,
            alternatives = listOf(candidate(species, 0.3f)),
            coarseFallback = candidate(genus, 0.7f, classIndex = -1),
            margin = 0.3f,
            normalizedEntropy = 0.4f,
        )
        val text = Report.explain(id, cal)
        assertClean(text)
        assertThat(text).contains("LOW_CONFIDENCE")
        assertThat(text).contains("70.0%")   // genus mass
        assertThat(text).contains("55%")     // COARSE_THRESHOLD, shown as %.0f
        assertThat(text).contains("Perca")
    }

    @Test
    fun `unknown result with no candidates at all is explained without a crash`() {
        val id = Identification.unknown(alternatives = emptyList())
        val text = Report.explain(id, CalibrationSpec())
        assertClean(text)
        assertThat(text).contains("NO_CANDIDATE")
    }

    @Test
    fun `candidateLine renders without leftover placeholders`() {
        val line = Report.candidateLine(1, candidate(species, 0.734f))
        assertClean(line)
        assertThat(line).contains("Perca fluviatilis")
        assertThat(line).contains("73.40%")
    }
}
