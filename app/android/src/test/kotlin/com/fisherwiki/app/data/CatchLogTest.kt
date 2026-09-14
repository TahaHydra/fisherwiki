package com.fisherwiki.app.data

import com.fisherwiki.core.model.Candidate
import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.model.Identification
import com.fisherwiki.core.model.Rank
import com.fisherwiki.core.model.Taxon
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

/**
 * [CatchLog.from] with a user correction, and the round trip through
 * [CatchLog.insert]/[CatchLog.all] - the data-layer half of "Not right?".
 * The other half, [IdentifyViewModel.save] and [ui.CorrectionSheet], is UI
 * wiring with no independent logic of its own to test on the JVM.
 */
@RunWith(RobolectricTestRunner::class)
class CatchLogTest {

    private val catchLog get() = CatchLog(RuntimeEnvironment.getApplication())

    private fun candidate(name: String, id: Long, p: Float) = Candidate(
        taxon = Taxon(id, name, Rank.SPECIES),
        probability = p, visualProbability = p, classIndex = 0,
    )

    @Test
    fun `a correction preserves the model's own guess alongside it`() {
        val modelGuess = candidate("Mullus barbatus", 100, 0.97f)
        val identification = Identification(
            certainty = Certainty.CONFIDENT,
            best = modelGuess,
            alternatives = emptyList(),
            rejectionReason = com.fisherwiki.core.model.RejectionReason.NONE,
        )

        val record = CatchLog.from(
            identification, photoPaths = emptyList(),
            correction = 200L to "Trachinus draco",
        )

        // The model's answer is not overwritten by the correction - that is
        // the whole point: a correction is only useful training signal if it
        // is honest about what the model actually said.
        assertThat(record.identifiedTaxonId).isEqualTo(100L)
        assertThat(record.identifiedName).isEqualTo("Mullus barbatus")
        assertThat(record.correctedTaxonId).isEqualTo(200L)
        assertThat(record.correctedName).isEqualTo("Trachinus draco")
        assertThat(record.wasCorrected).isTrue()
        // The corrected name wins for display - a user should see their own
        // answer, not the one they said was wrong.
        assertThat(record.displayName).isEqualTo("Trachinus draco")
    }

    @Test
    fun `no correction leaves the record exactly as the model produced it`() {
        val identification = Identification(
            certainty = Certainty.CONFIDENT,
            best = candidate("Perca fluviatilis", 1, 0.9f),
            alternatives = emptyList(),
            rejectionReason = com.fisherwiki.core.model.RejectionReason.NONE,
        )
        val record = CatchLog.from(identification, photoPaths = emptyList())

        assertThat(record.correctedTaxonId).isNull()
        assertThat(record.wasCorrected).isFalse()
        assertThat(record.displayName).isEqualTo("Perca fluviatilis")
    }

    @Test
    fun `a genus-level fallback is recorded without a fake species id`() {
        // CandidateRanker.coarseOrUnknown mints the fallback's taxon id as
        // `-1L - classIndex` - real, but not a row in the species database.
        val genusFallback = Candidate(
            taxon = Taxon(id = -2L, scientificName = "Sebastes", rank = Rank.GENUS, genus = "Sebastes"),
            probability = 0.73f, visualProbability = 0.73f, classIndex = -1,
        )
        val identification = Identification(
            certainty = Certainty.COARSE_ONLY,
            best = null,
            alternatives = emptyList(),
            coarseFallback = genusFallback,
            rejectionReason = com.fisherwiki.core.model.RejectionReason.NARROW_MARGIN,
        )

        val record = CatchLog.from(identification, photoPaths = emptyList())

        // The genus name is real and worth keeping; the synthetic id is not
        // a species and must not be stored in a column named identifiedTaxonId.
        assertThat(record.identifiedTaxonId).isNull()
        assertThat(record.identifiedName).isEqualTo("Sebastes")
        assertThat(record.displayName).isEqualTo("Sebastes")
    }

    @Test
    fun `a corrected record round-trips through insert and all`() {
        val identification = Identification(
            certainty = Certainty.CONFIDENT,
            best = candidate("Mullus barbatus", 100, 0.97f),
            alternatives = emptyList(),
            rejectionReason = com.fisherwiki.core.model.RejectionReason.NONE,
        )
        val record = CatchLog.from(
            identification, photoPaths = listOf("/data/catches/1.jpg"),
            correction = 200L to "Trachinus draco",
        )

        val id = catchLog.insert(record)
        val stored = catchLog.all().single { it.id == id }

        assertThat(stored.identifiedName).isEqualTo("Mullus barbatus")
        assertThat(stored.correctedName).isEqualTo("Trachinus draco")
        assertThat(stored.wasCorrected).isTrue()
        assertThat(stored.photoPaths).containsExactly("/data/catches/1.jpg")
    }

    @Test
    fun `correcting an already-saved catch via correct() also sticks`() {
        val identification = Identification(
            certainty = Certainty.CONFIDENT,
            best = candidate("Mullus barbatus", 100, 0.97f),
            alternatives = emptyList(),
            rejectionReason = com.fisherwiki.core.model.RejectionReason.NONE,
        )
        val id = catchLog.insert(CatchLog.from(identification, photoPaths = emptyList()))
        assertThat(catchLog.all().single { it.id == id }.wasCorrected).isFalse()

        catchLog.correct(id, 200L, "Trachinus draco")

        val stored = catchLog.all().single { it.id == id }
        assertThat(stored.wasCorrected).isTrue()
        assertThat(stored.correctedName).isEqualTo("Trachinus draco")
        // The original prediction the audit's finding #12 was about
        // preserving is, in fact, preserved by this path too.
        assertThat(stored.identifiedName).isEqualTo("Mullus barbatus")
    }
}
