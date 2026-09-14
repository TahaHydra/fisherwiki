package com.fisherwiki.core.engine

import com.fisherwiki.core.db.SpeciesRepository
import com.fisherwiki.core.db.SqliteDriver
import com.fisherwiki.core.infer.OnnxClassifier
import com.fisherwiki.core.infer.Preprocessor
import com.fisherwiki.core.model.Identification
import com.fisherwiki.core.pack.InstalledPack
import com.fisherwiki.core.rank.CandidateRanker
import com.fisherwiki.core.rank.GeoPrior
import java.io.Closeable

/**
 * The whole offline identification path, behind one object.
 *
 * Owns a verified pack's model, species database and geographic prior, and
 * turns pixels into an [Identification]. The Android layer supplies only two
 * things it cannot: a [SqliteDriver] and decoded ARGB pixels.
 *
 * **Nothing in this class or anything it calls opens a network connection.**
 * That is the product's core guarantee and it is structural, not a policy: the
 * model file, the database and the prior all come from a hash-verified pack on
 * local storage, and the only inputs are an int array of pixels and an optional
 * latitude/longitude.
 */
class IdentificationEngine private constructor(
    val pack: InstalledPack,
    private val classifier: OnnxClassifier,
    private val repository: SpeciesRepository,
    private val ranker: CandidateRanker,
) : Closeable {

    companion object {
        /**
         * Open an engine over a verified pack.
         *
         * @param driverFactory builds a read-only SQLite driver for a file path;
         *   supplied by the platform because Android and the JVM differ here.
         * @param threads inference threads. See [OnnxClassifier.open] for why
         *   this is not "all of them".
         */
        fun open(
            pack: InstalledPack,
            driverFactory: (String) -> SqliteDriver,
            language: String = "en",
            threads: Int = 4,
            useNnapi: Boolean = false,
        ): IdentificationEngine {
            val spec = pack.manifest.model
            val classifier = OnnxClassifier.open(
                pack.modelFile, spec, threads = threads, useNnapi = useNnapi
            )
            // Everything from here on can throw (consistency checks below,
            // GeoPrior, taxon resolution), and unlike the happy path - where
            // the returned IdentificationEngine owns classifier/repository and
            // callers close() it - nothing else will ever close *these*
            // instances if we throw out of this function. Close on any
            // failure so a pack that fails a late check does not leak an ONNX
            // session or a SQLite connection; the latter is what a JUnit
            // @TempDir failing to clean up on Windows was actually reporting.
            try {
                val repository = SpeciesRepository(
                    driverFactory(pack.databaseFile.absolutePath), language
                )
                try {
                    val (classCount, minIndex, maxIndex) = repository.classIndexDensity()
                    require(classCount == spec.numClasses) {
                        "pack is internally inconsistent: manifest declares " +
                            "${spec.numClasses} classes but the database has $classCount"
                    }
                    // Count alone does not prove density: a table with indices
                    // 0, 1, ..., N-2, 5000 has exactly N rows too, and every
                    // class beyond the gap would silently resolve to the wrong
                    // taxon. See SpeciesRepository.classIndexDensity for why
                    // this check exists.
                    require(classCount == 0 || (minIndex == 0 && maxIndex == classCount - 1)) {
                        "pack is internally inconsistent: $classCount model classes are " +
                            "not densely indexed from 0 (min=$minIndex, max=$maxIndex) - a " +
                            "gap would silently shift the meaning of every class index after it"
                    }

                    val geo = pack.geoPriorFile?.let { f ->
                        runCatching { GeoPrior.read(f, spec.numClasses) }.getOrElse {
                            // A corrupt prior must degrade to purely visual ranking,
                            // never to a crash or to silently wrong geography.
                            GeoPrior.neutral(spec.numClasses)
                        }
                    } ?: GeoPrior.neutral(spec.numClasses)

                    val (genusOfClass, genusNames) = repository.genusIndexByClass()

                    // Resolve every class's taxon once, up front. ~2,000 rows is
                    // a few hundred KB and it takes a database round-trip off
                    // the path between shutter press and result.
                    val taxa = repository.taxaForClasses((0 until spec.numClasses).toList())

                    val ranker = CandidateRanker(
                        calibration = spec.calibration,
                        geoPrior = geo,
                        taxonResolver = { idx -> taxa[idx] },
                        genusOfClass = genusOfClass,
                        genusName = { g -> genusNames.getOrNull(g) },
                    )
                    return IdentificationEngine(pack, classifier, repository, ranker)
                } catch (t: Throwable) {
                    runCatching { repository.close() }
                    throw t
                }
            } catch (t: Throwable) {
                runCatching { classifier.close() }
                throw t
            }
        }
    }

    /** Quick ID: one photograph. */
    fun identify(
        pixels: IntArray,
        width: Int,
        height: Int,
        location: CandidateRanker.Location? = null,
        topK: Int = 5,
    ): Identification {
        val spec = pack.manifest.model
        val tensor = Preprocessor.toTensor(
            pixels, width, height, spec.inputSize,
            spec.inputMean.toFloatArray(), spec.inputStd.toFloatArray(),
        )
        val out = classifier.run(tensor)
        return ranker.rank(
            out.logits, location, topK,
            packId = pack.id, packVersion = pack.version,
            inferenceMillis = out.millis,
        )
    }

    /**
     * Expert ID: several photographs of the same fish.
     *
     * Batched into one session call, which matters on a phone where per-call
     * overhead dominates for a small model.
     */
    fun identifyMulti(
        photos: List<Photo>,
        location: CandidateRanker.Location? = null,
        topK: Int = 5,
    ): Identification {
        require(photos.isNotEmpty()) { "no photographs supplied" }
        val spec = pack.manifest.model
        val mean = spec.inputMean.toFloatArray()
        val std = spec.inputStd.toFloatArray()
        val tensors = photos.map {
            Preprocessor.toTensor(
                it.pixels, it.width, it.height, spec.inputSize, mean, std
            )
        }
        val outs = classifier.runBatch(tensors)
        return ranker.fuse(
            outs.map { it.logits }, location, topK,
            packId = pack.id, packVersion = pack.version,
            inferenceMillis = outs.sumOf { it.millis },
        )
    }

    data class Photo(val pixels: IntArray, val width: Int, val height: Int) {
        override fun equals(other: Any?): Boolean =
            other is Photo && width == other.width && height == other.height &&
                pixels.contentEquals(other.pixels)

        override fun hashCode(): Int =
            31 * (31 * pixels.contentHashCode() + width) + height
    }

    /** Species detail for the result and detail screens. */
    fun detail(fwTaxonId: Long): SpeciesRepository.SpeciesDetail? =
        repository.detail(fwTaxonId)

    /**
     * Safety warnings for everything the result puts on screen.
     *
     * Takes the whole [Identification] rather than a taxon id, because the
     * warning a user needs may belong to a candidate that did *not* win. See
     * [SpeciesRepository.hazardsForCandidates].
     */
    fun hazards(identification: Identification): List<SpeciesRepository.CandidateHazard> {
        val ids = (identification.ranked.map { it.taxon.id } +
            listOfNotNull(identification.coarseFallback?.taxon?.id))
            .filter { it > 0 }
        return repository.hazardsForCandidates(ids)
    }

    fun search(query: String, limit: Int = 50) = repository.search(query, limit)

    fun sources() = repository.allSources()

    fun describeModel(): String = classifier.describe()

    override fun close() {
        runCatching { classifier.close() }
        runCatching { repository.close() }
    }
}
