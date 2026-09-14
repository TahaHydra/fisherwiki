package com.fisherwiki.core.rank

import com.google.common.truth.Truth.assertThat
import java.io.File
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir

/**
 * Cross-language round trip for the geographic prior.
 *
 * The fixture at `resources/fixtures/geoprior_fixture.bin` was produced by the
 * **shipping Python writer** (`tools/fwdata/geoprior.py`), not hand-assembled
 * here. That is the point: the format is big-endian to match
 * `DataInputStream`, and a byte-order or packing mismatch between the two
 * implementations would parse perfectly and yield silent nonsense — the app
 * would simply rank fish by a geography that does not exist.
 *
 * Fixture contents, all at 2-degree cells:
 *
 * | class | data |
 * |---|---|
 * | 0 | dense at Paris (48.85N, 2.35E, 500 obs), sparse at Oslo (60N, 10E, 3 obs) |
 * | 1 | dense in Florida (27N, 81W, 1200 obs) |
 * | 2 | no data at all |
 * | 3 | straddles the antimeridian: Fiji at 178E and at 178W |
 */
class GeoPriorRoundTripTest {

    private fun fixture(@TempDir tmp: File): File {
        val bytes = javaClass.getResourceAsStream("/fixtures/geoprior_fixture.bin")
            ?.readBytes()
            ?: error("geoprior_fixture.bin missing from test resources")
        return File(tmp, "geoprior.bin").apply { writeBytes(bytes) }
    }

    @Test
    fun `reads a file written by the Python writer`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)

        // If the byte order were wrong, the header would not have validated
        // and read() would have thrown. Getting here already proves agreement
        // on magic, version, cell size and class count.
        assertThat(prior.observationCount(0)).isEqualTo(503)
        assertThat(prior.observationCount(1)).isEqualTo(1200)
        assertThat(prior.observationCount(2)).isEqualTo(0)
        assertThat(prior.observationCount(3)).isEqualTo(75)
    }

    @Test
    fun `cell packing agrees with the Python implementation`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)
        // Values computed by tools/fwdata/geoprior.pack_cell at 2-degree cells.
        assertThat(prior.cellOf(48.85, 2.35)).isEqualTo(4522075)   // Paris
        assertThat(prior.cellOf(60.0, 10.0)).isEqualTo(4915295)    // Oslo
        assertThat(prior.cellOf(27.0, -81.0)).isEqualTo(3801137)   // Florida
        assertThat(prior.cellOf(-17.8, 178.0)).isEqualTo(2359475)  // Fiji east
        assertThat(prior.cellOf(-17.8, -178.0)).isEqualTo(2359297) // Fiji west
    }

    @Test
    fun `a species is boosted where it is common`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)
        val atParis = prior.factor(0, 48.85, 2.35, visualConfidence = 0f)
        assertThat(atParis).isGreaterThan(1.5f)
        assertThat(atParis).isAtMost(GeoPrior.MAX_FACTOR)
    }

    @Test
    fun `a species is penalised where it is not recorded`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)
        // Class 1 (Florida) has no data anywhere near Paris.
        val floridaFishInParis = prior.factor(1, 48.85, 2.35, visualConfidence = 0f)
        assertThat(floridaFishInParis).isLessThan(1f)
        // But never eliminated: an escapee or a vagrant must stay identifiable.
        assertThat(floridaFishInParis).isAtLeast(GeoPrior.MIN_FACTOR)
        assertThat(floridaFishInParis).isGreaterThan(0f)
    }

    @Test
    fun `sparse presence scores lower than dense presence`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)
        val dense = prior.factor(0, 48.85, 2.35, 0f)   // 500 observations
        val sparse = prior.factor(0, 60.0, 10.0, 0f)   // 3 observations
        assertThat(sparse).isLessThan(dense)
        // The log scale must keep them distinguishable rather than collapsing
        // the rare cell onto "absent".
        assertThat(sparse).isGreaterThan(GeoPrior.MIN_FACTOR)
    }

    @Test
    fun `a class with no data anywhere is unknown not absent`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)
        assertThat(prior.isUnknown(2)).isTrue()
        val f = prior.factor(2, 48.85, 2.35, 0f)
        assertThat(f).isWithin(1e-4f).of(GeoPrior.UNKNOWN_FACTOR)
        // Absence of evidence in a citizen-science dataset is weak evidence of
        // absence, so this must be a mild penalty rather than exclusion.
        assertThat(f).isLessThan(1f)
        assertThat(f).isGreaterThan(0.5f)
    }

    @Test
    fun `antimeridian is handled without splitting a range in two`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)
        // Class 3 has cells on both sides of 180 degrees. Both must score high,
        // and a longitude expressed either way must land in the same place.
        val east = prior.factor(3, -17.8, 178.0, 0f)
        val west = prior.factor(3, -17.8, -178.0, 0f)
        assertThat(east).isGreaterThan(1f)
        assertThat(west).isGreaterThan(1f)

        // 182E is the same meridian as 178W.
        assertThat(prior.cellOf(-17.8, 182.0)).isEqualTo(prior.cellOf(-17.8, -178.0))
    }

    @Test
    fun `no location means a neutral factor`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)
        assertThat(prior.factor(0, null, null, 0f)).isEqualTo(1f)
        assertThat(prior.factor(1, 48.85, null, 0f)).isEqualTo(1f)
    }

    @Test
    fun `visual confidence attenuates the prior`(@TempDir tmp: File) {
        val prior = GeoPrior.read(fixture(tmp), expectedClasses = 4)
        // A Florida fish photographed in Paris: penalised when the model is
        // unsure, but much less so when the model is confident about what it
        // is looking at.
        val unsure = prior.factor(1, 48.85, 2.35, visualConfidence = 0.2f)
        val confident = prior.factor(1, 48.85, 2.35, visualConfidence = 0.97f)
        assertThat(confident).isGreaterThan(unsure)
        assertThat(confident).isGreaterThan(0.9f)
    }

    @Test
    fun `a class count mismatch is rejected`(@TempDir tmp: File) {
        // A prior built for a different model must never be silently accepted:
        // its class indices would mean different species.
        assertThrows<java.io.IOException> {
            GeoPrior.read(fixture(tmp), expectedClasses = 99)
        }
    }

    @Test
    fun `a corrupt header is rejected`(@TempDir tmp: File) {
        val f = fixture(tmp)
        val bytes = f.readBytes()
        bytes[0] = 0x00
        val bad = File(tmp, "bad.bin").apply { writeBytes(bytes) }
        assertThrows<java.io.IOException> { GeoPrior.read(bad, expectedClasses = 4) }
    }

    @Test
    fun `neutral prior leaves ranking purely visual`() {
        val prior = GeoPrior.neutral(10)
        for (c in 0 until 10) {
            assertThat(prior.factor(c, null, null, 0f)).isEqualTo(1f)
        }
    }
}
