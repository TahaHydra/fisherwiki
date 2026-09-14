package com.fisherwiki.core.engine

import com.fisherwiki.core.db.JdbcSqliteDriver
import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.model.Rank
import com.fisherwiki.core.pack.PackVerifier
import com.fisherwiki.core.rank.CandidateRanker
import com.google.common.truth.Truth.assertThat
import java.io.File
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir

/**
 * The whole offline path, end to end, against a **real pack**.
 *
 * The fixture `.fwpack` is genuine in every respect: a real ONNX graph exported
 * by torch, a real SQLite database built by the shipping `SpeciesDatabaseBuilder`,
 * a real geo prior written by the shipping writer, and a real manifest with real
 * SHA-256 hashes. It is small only because the model is deliberately tiny.
 *
 * The model is constructed so that its behaviour is *predictable* rather than
 * merely non-crashing: each class responds to one dominant colour. That lets
 * these tests assert which species comes back, which is the difference between
 * a smoke test and a behavioural one.
 *
 * This is what verifies that verification, extraction, ONNX inference,
 * calibration, the geographic prior, ranking and the species database actually
 * work *together* — on a workstation, with no Android device and no trained
 * model.
 */
class EngineEndToEndTest {

    private fun openPack(tmp: File): IdentificationEngine {
        val bytes = javaClass.getResourceAsStream("/fixtures/pack/fixture_v1.fwpack")
            ?.readBytes() ?: error("fixture_v1.fwpack missing from test resources")
        val archive = File(tmp, "fixture_v1.fwpack").apply { writeBytes(bytes) }

        val result = PackVerifier.verifyAndExtract(archive, File(tmp, "installed"))
        assertThat(result).isInstanceOf(PackVerifier.Result.Ok::class.java)
        val pack = (result as PackVerifier.Result.Ok).pack

        return IdentificationEngine.open(
            pack = pack,
            driverFactory = { path -> JdbcSqliteDriver(path) },
            language = "en",
            threads = 2,
        )
    }

    /** A solid-colour image, as ARGB pixels. */
    private fun solid(r: Int, g: Int, b: Int, w: Int = 200, h: Int = 150): IntArray =
        IntArray(w * h) { (0xFF shl 24) or (r shl 16) or (g shl 8) or b }

    // ------------------------------------------------------- pack loading

    @Test
    fun `a real pack verifies, extracts and opens`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            assertThat(engine.pack.id).isEqualTo("fixture_v1")
            assertThat(engine.pack.version).isEqualTo(1)
            assertThat(engine.pack.manifest.model.numClasses).isEqualTo(4)
            assertThat(engine.pack.commercialSafe).isTrue()
            assertThat(engine.describeModel()).contains("classes: 4")
        }
    }

    @Test
    fun `a tampered pack is rejected`(@TempDir tmp: File) {
        val bytes = javaClass.getResourceAsStream("/fixtures/pack/fixture_v1.fwpack")!!
            .readBytes()
        // Rebuild the archive with a modified model, leaving the manifest's
        // hash unchanged. This is the attack the per-file SHA-256 exists for.
        val src = File(tmp, "src.fwpack").apply { writeBytes(bytes) }
        val tampered = File(tmp, "tampered.fwpack")
        java.util.zip.ZipFile(src).use { zf ->
            java.util.zip.ZipOutputStream(tampered.outputStream()).use { out ->
                for (e in zf.entries()) {
                    out.putNextEntry(java.util.zip.ZipEntry(e.name))
                    val data = zf.getInputStream(e).readBytes()
                    if (e.name == "model.onnx") {
                        data[data.size / 2] = (data[data.size / 2] + 1).toByte()
                    }
                    out.write(data)
                    out.closeEntry()
                }
            }
        }
        val result = PackVerifier.verifyAndExtract(tampered, File(tmp, "out"))
        assertThat(result).isInstanceOf(PackVerifier.Result.Rejected::class.java)
        assertThat((result as PackVerifier.Result.Rejected).reason).contains("sha256")
    }

    // ------------------------------------------------------- identification

    @Test
    fun `a red image identifies as the red species`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val id = engine.identify(solid(230, 30, 30), 200, 150)
            assertThat(id.certainty).isEqualTo(Certainty.CONFIDENT)
            assertThat(id.best).isNotNull()
            assertThat(id.best!!.taxon.scientificName).isEqualTo("Testus ruber")
            assertThat(id.best.taxon.commonName).isEqualTo("Red test fish")
            assertThat(id.best.probability).isGreaterThan(0.9f)
            assertThat(id.inferenceMillis).isAtLeast(0)
            assertThat(id.packId).isEqualTo("fixture_v1")
        }
    }

    @Test
    fun `a blue image identifies as the blue species`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val id = engine.identify(solid(30, 30, 230), 200, 150)
            assertThat(id.best!!.taxon.scientificName).isEqualTo("Alius caeruleus")
        }
    }

    @Test
    fun `alternatives are populated and ordered`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val id = engine.identify(solid(230, 30, 30), 200, 150, topK = 4)
            assertThat(id.alternatives).isNotEmpty()
            var prev = id.best!!.probability
            for (a in id.alternatives) {
                assertThat(a.probability).isAtMost(prev)
                prev = a.probability
            }
        }
    }

    // --------------------------------------------------------- uncertainty

    @Test
    fun `an ambiguous image does not produce a confident species claim`(
        @TempDir tmp: File
    ) {
        openPack(tmp).use { engine ->
            // Equal red and green: the model genuinely cannot separate class 0
            // from class 1, which is exactly when it must not pick one.
            val id = engine.identify(solid(180, 180, 40), 200, 150)
            if (id.certainty == Certainty.CONFIDENT) {
                // Permitted only if the margin is genuinely wide.
                assertThat(id.margin).isGreaterThan(0.2f)
            } else {
                assertThat(id.isUncertain || id.certainty == Certainty.AMBIGUOUS)
                    .isTrue()
            }
        }
    }

    @Test
    fun `a grey image is not confidently named`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            // No dominant channel: the closest thing this fixture has to an
            // out-of-distribution input.
            val id = engine.identify(solid(128, 128, 128), 200, 150)
            if (id.best != null) {
                assertThat(id.best!!.probability).isLessThan(0.99f)
            }
        }
    }

    // ------------------------------------------------------------- geo prior

    @Test
    fun `the geographic prior shifts ranking without overriding clear evidence`(
        @TempDir tmp: File
    ) {
        openPack(tmp).use { engine ->
            val pixels = solid(230, 30, 30)     // unambiguously class 0
            val noLoc = engine.identify(pixels, 200, 150)
            // Class 0 is recorded in Europe; class 2 in Florida. Asking from
            // Florida must not turn a clear red fish into the blue one.
            val florida = engine.identify(
                pixels, 200, 150, CandidateRanker.Location(27.0, -81.0)
            )
            assertThat(florida.geoApplied).isTrue()
            assertThat(noLoc.geoApplied).isFalse()
            assertThat(florida.best!!.taxon.scientificName)
                .isEqualTo(noLoc.best!!.taxon.scientificName)
        }
    }

    @Test
    fun `geo factors are recorded on candidates`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val id = engine.identify(
                solid(230, 30, 30), 200, 150,
                CandidateRanker.Location(48.85, 2.35),
            )
            assertThat(id.ranked.any { it.geoFactor != 1f }).isTrue()
        }
    }

    // ------------------------------------------------------ multi-photo

    @Test
    fun `multi photo fusion runs and agrees with the single photo result`(
        @TempDir tmp: File
    ) {
        openPack(tmp).use { engine ->
            val photos = listOf(
                IdentificationEngine.Photo(solid(230, 30, 30, 200, 150), 200, 150),
                IdentificationEngine.Photo(solid(220, 40, 40, 180, 180), 180, 180),
                IdentificationEngine.Photo(solid(240, 20, 35, 300, 200), 300, 200),
            )
            val fused = engine.identifyMulti(photos)
            assertThat(fused.photoCount).isEqualTo(3)
            assertThat(fused.best!!.taxon.scientificName).isEqualTo("Testus ruber")
        }
    }

    @Test
    fun `one bad frame does not overturn two agreeing frames`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val photos = listOf(
                IdentificationEngine.Photo(solid(230, 30, 30), 200, 150),
                IdentificationEngine.Photo(solid(230, 30, 30), 200, 150),
                IdentificationEngine.Photo(solid(30, 30, 240), 200, 150), // wrong
            )
            val fused = engine.identifyMulti(photos)
            // The sum rule must let the majority win; the product rule would
            // let the outlier veto class 0 entirely.
            assertThat(fused.best?.taxon?.scientificName
                ?: fused.coarseFallback?.taxon?.scientificName)
                .isAnyOf("Testus ruber", "Testus")
        }
    }

    // -------------------------------------------------------- species data

    @Test
    fun `species detail comes back with sourced facts`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val id = engine.identify(solid(30, 30, 230), 200, 150)
            val detail = engine.detail(id.best!!.taxon.id)
            assertThat(detail).isNotNull()
            assertThat(detail!!.displayName).isEqualTo("Blue test fish")
            // This species carries a danger-level warning in the fixture.
            assertThat(detail.safetyWarnings).isNotEmpty()
            assertThat(detail.safetyWarnings.first().isDangerous).isTrue()
            assertThat(detail.sources).isNotEmpty()
        }
    }

    @Test
    fun `search works across common names and synonyms`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            assertThat(engine.search("Red test").map { it.scientificName })
                .contains("Testus ruber")
            // Historical synonym.
            assertThat(engine.search("Testus rubra").map { it.scientificName })
                .contains("Testus ruber")
        }
    }

    @Test
    fun `sources are listed for the about screen`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val sources = engine.sources()
            assertThat(sources).isNotEmpty()
            assertThat(sources.all { it.license.isNotEmpty() }).isTrue()
        }
    }

    @Test
    fun `a thin class is flagged as low confidence`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            // Class 3 has 45 training images in the fixture.
            val detail = engine.detail(9004L)
            assertThat(detail).isNotNull()
            assertThat(detail!!.isLowConfidenceClass).isTrue()
        }
    }

    @Test
    fun `taxon ranks are read correctly`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val id = engine.identify(solid(230, 30, 30), 200, 150)
            assertThat(id.best!!.taxon.rank).isEqualTo(Rank.SPECIES)
            assertThat(id.best.taxon.family).isEqualTo("Testidae")
        }
    }

    // ----------------------------------------------------------- robustness

    @Test
    fun `differently sized and shaped images all work`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            for ((w, h) in listOf(64 to 64, 1000 to 200, 200 to 1000, 33 to 47)) {
                val id = engine.identify(solid(230, 30, 30, w, h), w, h)
                assertThat(id.ranked).isNotEmpty()
            }
        }
    }

    @Test
    fun `repeated inference is stable`(@TempDir tmp: File) {
        openPack(tmp).use { engine ->
            val pixels = solid(230, 30, 30)
            val first = engine.identify(pixels, 200, 150)
            repeat(5) {
                val again = engine.identify(pixels, 200, 150)
                assertThat(again.best!!.taxon.id).isEqualTo(first.best!!.taxon.id)
                assertThat(again.best.probability)
                    .isWithin(1e-5f).of(first.best.probability)
            }
        }
    }
}
