package com.fisherwiki.core.db

import com.fisherwiki.core.model.Rank
import com.fisherwiki.core.model.Taxon

/**
 * Read-only access to a pack's species database.
 *
 * Everything the result screen shows comes from here, and every factual row
 * carries a source. Fields with no source are **absent**, and the repository
 * returns them as null rather than substituting a plausible default - see
 * `docs/SPECIES_DATA.md` for why that is the whole point.
 */
class SpeciesRepository(
    private val driver: SqliteDriver,
    /** Language tag for common names, e.g. "en" or "fr". */
    private val language: String = "en",
) : AutoCloseable {

    data class CommonName(val lang: String, val name: String, val preferred: Boolean)

    data class SourceRef(
        val sourceId: String,
        val title: String,
        val url: String?,
        val license: String,
        val licenseUrl: String?,
        val citation: String,
        val retrievedOn: String,
    )

    data class Trait(
        val key: String,
        val numeric: Double?,
        val text: String?,
        val unit: String?,
        val sourceId: String,
    )

    data class DiagnosticFeature(
        val ordinal: Int,
        val feature: String,
        val bodyPart: String?,
        val sourceId: String,
    )

    data class SimilarSpecies(
        val taxon: Taxon,
        val difference: String,
        /** Measured on our held-out test set; null when not yet measured. */
        val confusionRate: Double?,
        val sourceId: String,
    )

    data class SafetyWarning(
        val kind: String,
        val severity: String,
        val summary: String,
        val detail: String?,
        val appliesTo: String,
        val sourceId: String,
    ) {
        val isDangerous: Boolean get() = severity == "danger"
    }

    /**
     * A [SafetyWarning] tagged with the candidate it belongs to.
     *
     * Warnings must be reachable for *every* candidate on screen, not only the
     * top one. The model's most frequent confusions are within genus, which is
     * usually harmless, but not always: it confuses *Trachinus draco* -- a
     * weeverfish with venomous dorsal spines -- with the harmless *Mullus
     * barbatus* in 30.8% of test images. If warnings were attached only to the
     * winning candidate, being wrong in that direction would mean showing the
     * user no warning at all about a fish that can put them in hospital.
     */
    /** Why a hazard is being shown for a fish the user may not have caught. */
    enum class HazardReason {
        /** The warning belongs to a candidate the model actually proposed. */
        ALTERNATIVE_CANDIDATE,

        /**
         * The warning belongs to a species this model is *measured* to confuse
         * with the one on screen, whether or not it appeared in this result.
         *
         * This exists because ranking is not enough. On the 19 test images
         * where the model misidentified the venomous *Trachinus draco*, the
         * weever was still in the top five only 32% of the time -- once it was
         * called *Mullus barbatus* at 0.974 confidence with the weever at rank
         * 14. A static cross-reference fires in those cases; a candidate scan
         * does not.
         */
        KNOWN_LOOKALIKE,
    }

    data class CandidateHazard(
        val fwTaxonId: Long,
        val scientificName: String,
        val commonName: String?,
        val warning: SafetyWarning,
        /** 0 for the winning candidate, 1+ for alternatives. */
        val candidateRank: Int,
        val reason: HazardReason = HazardReason.ALTERNATIVE_CANDIDATE,
        /** Measured rate this model confuses the two; null when unmeasured. */
        val confusionRate: Double? = null,
    ) {
        val isForBestCandidate: Boolean
            get() = candidateRank == 0 && reason == HazardReason.ALTERNATIVE_CANDIDATE
    }

    data class RegionPresence(
        val regionId: String,
        val regionName: String,
        val observations: Int,
        val share: Double,
    )

    data class SpeciesDetail(
        val taxon: Taxon,
        val commonNames: List<CommonName>,
        val synonyms: List<String>,
        val habitats: List<Pair<String, String?>>,
        val traits: List<Trait>,
        val diagnosticFeatures: List<DiagnosticFeature>,
        val similarSpecies: List<SimilarSpecies>,
        val safetyWarnings: List<SafetyWarning>,
        val regions: List<RegionPresence>,
        val sources: List<SourceRef>,
        val trainImages: Int,
        val trainObservations: Int,
        val testPrecision: Double?,
        val testRecall: Double?,
        /** Language this detail was requested in, e.g. "en" or "fr". */
        val language: String = "en",
    ) {
        /**
         * Best available common name, preferring [language], then English.
         *
         * Falling back to English is more useful than showing only a Latin
         * binomial - but the fallback must be *visible*. [displayNameLanguage]
         * reports which language was actually used so the UI can mark it,
         * rather than presenting an English name as though it were the user's
         * own. Wikidata gives us English for 1,289 of 1,963 species and French
         * for 599, so this path is common, not exceptional.
         */
        val displayName: String
            get() = bestCommonName()?.name ?: taxon.scientificName

        /**
         * Language of [displayName]: the requested language when available,
         * `"en"` when we fell back, or null when no common name exists at all
         * and the scientific name is being shown.
         */
        val displayNameLanguage: String?
            get() = bestCommonName()?.lang

        /** True when [displayName] is not in the language that was asked for. */
        val displayNameIsFallback: Boolean
            get() {
                val used = displayNameLanguage ?: return false
                return used.substringBefore('-') != language.substringBefore('-')
            }

        private fun bestCommonName(): CommonName? {
            val base = language.substringBefore('-')
            return commonNames.firstOrNull {
                it.lang.substringBefore('-') == base && it.preferred
            }
                ?: commonNames.firstOrNull { it.lang.substringBefore('-') == base }
                ?: commonNames.firstOrNull {
                    it.lang.substringBefore('-') == "en" && it.preferred
                }
                ?: commonNames.firstOrNull { it.lang.substringBefore('-') == "en" }
        }

        /** True when this class had thin training support and should say so. */
        val isLowConfidenceClass: Boolean
            get() = trainImages < 80 || trainObservations < 40
    }

    // ------------------------------------------------------------------
    /** Class index -> taxon. The hot path immediately after inference. */
    fun taxonForClass(classIndex: Int): Taxon? = driver.queryOne(
        """
        SELECT t.fw_taxon_id, t.scientific_name, t.rank, t.genus, t.family,
               t.taxon_order, t.class,
               (SELECT name FROM common_names c
                 WHERE c.fw_taxon_id = t.fw_taxon_id AND c.lang = ?
                 ORDER BY c.preferred DESC LIMIT 1)
        FROM model_classes m JOIN taxa t ON t.fw_taxon_id = m.fw_taxon_id
        WHERE m.class_index = ?
        """,
        listOf(language, classIndex),
    ) { r -> mapTaxon(r) }

    /** Bulk variant, so ranking N candidates is one query rather than N. */
    fun taxaForClasses(classIndices: List<Int>): Map<Int, Taxon> {
        if (classIndices.isEmpty()) return emptyMap()
        val placeholders = classIndices.joinToString(",") { "?" }
        val out = HashMap<Int, Taxon>(classIndices.size * 2)
        driver.query(
            """
            SELECT m.class_index, t.fw_taxon_id, t.scientific_name, t.rank,
                   t.genus, t.family, t.taxon_order, t.class,
                   (SELECT name FROM common_names c
                     WHERE c.fw_taxon_id = t.fw_taxon_id AND c.lang = ?
                     ORDER BY c.preferred DESC LIMIT 1)
            FROM model_classes m JOIN taxa t ON t.fw_taxon_id = m.fw_taxon_id
            WHERE m.class_index IN ($placeholders)
            """,
            listOf<Any?>(language) + classIndices,
        ) { r ->
            val idx = r.getInt(0) ?: return@query
            out[idx] = Taxon(
                id = r.getLong(1) ?: 0L,
                scientificName = r.getString(2) ?: "",
                rank = parseRank(r.getString(3)),
                genus = r.getString(4),
                family = r.getString(5),
                order = r.getString(6),
                klass = r.getString(7),
                commonName = r.getString(8),
            )
        }
        return out
    }

    /**
     * Safety warnings across a ranked list of candidate taxa.
     *
     * [fwTaxonIds] is in candidate order, best first; the returned list is
     * ordered by severity and then by that rank, because a `danger` on the
     * third candidate matters more than a `caution` on the first.
     *
     * Duplicates are collapsed on (kind, summary): most warnings in the pack
     * apply at family or order rank, so three congeners in the candidate list
     * would otherwise produce the same "venomous dorsal spines" card three
     * times and bury everything else.
     */
    fun hazardsForCandidates(fwTaxonIds: List<Long>): List<CandidateHazard> {
        if (fwTaxonIds.isEmpty()) return emptyList()
        val rankOf = HashMap<Long, Int>(fwTaxonIds.size * 2)
        fwTaxonIds.forEachIndexed { i, id -> rankOf.putIfAbsent(id, i) }
        val ids = rankOf.keys.toList()
        val placeholders = ids.joinToString(",") { "?" }

        val rows = driver.query(
            """
            SELECT w.fw_taxon_id, t.scientific_name,
                   (SELECT name FROM common_names c
                     WHERE c.fw_taxon_id = t.fw_taxon_id AND c.lang = ?
                     ORDER BY c.preferred DESC LIMIT 1),
                   w.kind, w.severity, w.summary, w.detail, w.applies_to,
                   w.source_id
            FROM safety_warnings w
            JOIN taxa t ON t.fw_taxon_id = w.fw_taxon_id
            WHERE w.fw_taxon_id IN ($placeholders)
            """,
            listOf<Any?>(language) + ids,
        ) { r ->
            val id = r.getLong(0) ?: 0L
            CandidateHazard(
                fwTaxonId = id,
                scientificName = r.getString(1) ?: "",
                commonName = r.getString(2),
                warning = SafetyWarning(
                    kind = r.getString(3) ?: "",
                    severity = r.getString(4) ?: "info",
                    summary = r.getString(5) ?: "",
                    detail = r.getString(6),
                    appliesTo = r.getString(7) ?: "species",
                    sourceId = r.getString(8) ?: "",
                ),
                candidateRank = rankOf[id] ?: Int.MAX_VALUE,
            )
        }

        val severityOrder = { s: String ->
            when (s) { "danger" -> 0; "caution" -> 1; else -> 2 }
        }
        return (rows + dangerousLookalikes(ids, rankOf))
            .sortedWith(
                compareBy(
                    { severityOrder(it.warning.severity) },
                    // A species the model actually proposed beats a static
                    // look-alike carrying the same warning: "this photograph
                    // might be a weeverfish" is a stronger and more useful
                    // claim than "the species named above resembles one".
                    // Without this the look-alike wins, because it inherits
                    // the *displayed* species' rank, which is 0.
                    { it.reason.ordinal },
                    { it.candidateRank },
                ),
            )
            .distinctBy { it.warning.kind to it.warning.summary }
    }

    /**
     * Dangerous species this model is measured to confuse with [ids].
     *
     * Restricted to `danger`: a `caution` look-alike for every candidate would
     * be noise, and noise is what stops people reading warnings. Species that
     * already carry the same warning kind themselves are excluded -- telling
     * someone holding a scorpionfish that it resembles a scorpionfish adds
     * nothing to the warning already on screen.
     */
    private fun dangerousLookalikes(
        ids: List<Long>,
        rankOf: Map<Long, Int>,
    ): List<CandidateHazard> {
        if (ids.isEmpty()) return emptyList()
        val placeholders = ids.joinToString(",") { "?" }
        return driver.query(
            """
            SELECT s.fw_taxon_id, o.fw_taxon_id, o.scientific_name,
                   (SELECT name FROM common_names c
                     WHERE c.fw_taxon_id = o.fw_taxon_id AND c.lang = ?
                     ORDER BY c.preferred DESC LIMIT 1),
                   w.kind, w.severity, w.summary, w.detail, w.applies_to,
                   w.source_id, s.confusion_rate
            FROM similar_species s
            JOIN taxa o ON o.fw_taxon_id = s.other_fw_taxon_id
            JOIN safety_warnings w ON w.fw_taxon_id = s.other_fw_taxon_id
            WHERE s.fw_taxon_id IN ($placeholders)
              AND w.severity = 'danger'
              AND NOT EXISTS (
                  SELECT 1 FROM safety_warnings own
                  WHERE own.fw_taxon_id = s.fw_taxon_id AND own.kind = w.kind
              )
            ORDER BY s.confusion_rate DESC
            """,
            listOf<Any?>(language) + ids,
        ) { r ->
            CandidateHazard(
                fwTaxonId = r.getLong(1) ?: 0L,
                scientificName = r.getString(2) ?: "",
                commonName = r.getString(3),
                warning = SafetyWarning(
                    kind = r.getString(4) ?: "",
                    severity = r.getString(5) ?: "info",
                    summary = r.getString(6) ?: "",
                    detail = r.getString(7),
                    appliesTo = r.getString(8) ?: "species",
                    sourceId = r.getString(9) ?: "",
                ),
                // The rank of the *displayed* species this is a look-alike
                // of, not of the look-alike itself, which is typically not a
                // candidate at all. Used only to order look-alikes among
                // themselves; `reason` keeps them behind real candidates.
                candidateRank = rankOf[r.getLong(0) ?: 0L] ?: Int.MAX_VALUE,
                reason = HazardReason.KNOWN_LOOKALIKE,
                confusionRate = r.getDouble(10),
            )
        }
    }

    fun detail(fwTaxonId: Long): SpeciesDetail? {
        val taxon = driver.queryOne(
            """
            SELECT fw_taxon_id, scientific_name, rank, genus, family,
                   taxon_order, class,
                   (SELECT name FROM common_names c
                     WHERE c.fw_taxon_id = taxa.fw_taxon_id AND c.lang = ?
                     ORDER BY c.preferred DESC LIMIT 1)
            FROM taxa WHERE fw_taxon_id = ?
            """,
            listOf(language, fwTaxonId),
        ) { mapTaxon(it) } ?: return null

        val commonNames = driver.query(
            "SELECT lang, name, preferred FROM common_names WHERE fw_taxon_id = ? " +
                "ORDER BY (lang = ?) DESC, preferred DESC, name",
            listOf(fwTaxonId, language),
        ) { r ->
            CommonName(r.getString(0) ?: "", r.getString(1) ?: "", (r.getInt(2) ?: 0) == 1)
        }

        val synonyms = driver.query(
            "SELECT name FROM taxon_synonyms WHERE fw_taxon_id = ? ORDER BY name",
            listOf(fwTaxonId),
        ) { it.getString(0) ?: "" }

        val habitats = driver.query(
            "SELECT water, detail FROM habitats WHERE fw_taxon_id = ?",
            listOf(fwTaxonId),
        ) { r -> (r.getString(0) ?: "") to r.getString(1) }

        val traits = driver.query(
            "SELECT key, value_num, value_text, unit, source_id FROM traits " +
                "WHERE fw_taxon_id = ? ORDER BY key",
            listOf(fwTaxonId),
        ) { r ->
            Trait(r.getString(0) ?: "", r.getDouble(1), r.getString(2),
                r.getString(3), r.getString(4) ?: "")
        }

        val features = driver.query(
            "SELECT ordinal, feature, body_part, source_id FROM diagnostic_features " +
                "WHERE fw_taxon_id = ? ORDER BY ordinal",
            listOf(fwTaxonId),
        ) { r ->
            DiagnosticFeature(r.getInt(0) ?: 0, r.getString(1) ?: "",
                r.getString(2), r.getString(3) ?: "")
        }

        val similar = driver.query(
            """
            SELECT s.other_fw_taxon_id, t.scientific_name, t.rank, t.genus, t.family,
                   t.taxon_order, t.class,
                   (SELECT name FROM common_names c
                     WHERE c.fw_taxon_id = t.fw_taxon_id AND c.lang = ?
                     ORDER BY c.preferred DESC LIMIT 1),
                   s.difference, s.confusion_rate, s.source_id
            FROM similar_species s JOIN taxa t ON t.fw_taxon_id = s.other_fw_taxon_id
            WHERE s.fw_taxon_id = ?
            ORDER BY s.confusion_rate DESC NULLS LAST
            """,
            listOf(language, fwTaxonId),
        ) { r ->
            SimilarSpecies(
                taxon = Taxon(
                    id = r.getLong(0) ?: 0L,
                    scientificName = r.getString(1) ?: "",
                    rank = parseRank(r.getString(2)),
                    genus = r.getString(3), family = r.getString(4),
                    order = r.getString(5), klass = r.getString(6),
                    commonName = r.getString(7),
                ),
                difference = r.getString(8) ?: "",
                confusionRate = r.getDouble(9),
                sourceId = r.getString(10) ?: "",
            )
        }

        val warnings = driver.query(
            "SELECT kind, severity, summary, detail, applies_to, source_id " +
                "FROM safety_warnings WHERE fw_taxon_id = ? " +
                // Show the most serious first: a user skims.
                "ORDER BY CASE severity WHEN 'danger' THEN 0 WHEN 'caution' THEN 1 " +
                "ELSE 2 END",
            listOf(fwTaxonId),
        ) { r ->
            SafetyWarning(r.getString(0) ?: "", r.getString(1) ?: "info",
                r.getString(2) ?: "", r.getString(3),
                r.getString(4) ?: "species", r.getString(5) ?: "")
        }

        val regionRows = driver.query(
            "SELECT tr.region_id, r.name, tr.observations, tr.share " +
                "FROM taxon_regions tr JOIN regions r USING (region_id) " +
                "WHERE tr.fw_taxon_id = ? ORDER BY tr.observations DESC",
            listOf(fwTaxonId),
        ) { r ->
            RegionPresence(r.getString(0) ?: "", r.getString(1) ?: "",
                r.getInt(2) ?: 0, r.getDouble(3) ?: 0.0)
        }

        val sourceIds = (traits.map { it.sourceId } + features.map { it.sourceId } +
            similar.map { it.sourceId } + warnings.map { it.sourceId })
            .filter { it.isNotEmpty() }.distinct()
        val sources = if (sourceIds.isEmpty()) allSources() else sourcesById(sourceIds)

        val support = driver.queryOne(
            "SELECT train_images, train_observations, test_precision, test_recall " +
                "FROM model_classes WHERE fw_taxon_id = ?",
            listOf(fwTaxonId),
        ) { r -> listOf(r.getInt(0), r.getInt(1), r.getDouble(2), r.getDouble(3)) }

        return SpeciesDetail(
            taxon = taxon,
            commonNames = commonNames,
            synonyms = synonyms,
            habitats = habitats,
            traits = traits,
            diagnosticFeatures = features,
            similarSpecies = similar,
            safetyWarnings = warnings,
            regions = regionRows,
            sources = sources,
            trainImages = (support?.get(0) as? Int) ?: 0,
            trainObservations = (support?.get(1) as? Int) ?: 0,
            testPrecision = support?.get(2) as? Double,
            testRecall = support?.get(3) as? Double,
            language = language,
        )
    }

    /**
     * Search by common name, scientific name **or synonym**.
     *
     * Synonyms matter more than they look: an angler who learned "Stizostedion
     * lucioperca" decades ago should still find the zander.
     */
    fun search(queryText: String, limit: Int = 50): List<Taxon> {
        val q = queryText.trim()
        if (q.length < 2) return emptyList()
        val like = "$q%"
        val contains = "%$q%"
        return driver.query(
            """
            SELECT DISTINCT t.fw_taxon_id, t.scientific_name, t.rank, t.genus,
                   t.family, t.taxon_order, t.class,
                   (SELECT name FROM common_names c
                     WHERE c.fw_taxon_id = t.fw_taxon_id AND c.lang = ?
                     ORDER BY c.preferred DESC LIMIT 1) AS display,
                   CASE
                     WHEN t.scientific_name LIKE ? COLLATE NOCASE THEN 0
                     WHEN EXISTS (SELECT 1 FROM common_names c2
                                  WHERE c2.fw_taxon_id = t.fw_taxon_id
                                    AND c2.name LIKE ? COLLATE NOCASE) THEN 1
                     ELSE 2
                   END AS rankOrder
            FROM taxa t
            WHERE t.scientific_name LIKE ? COLLATE NOCASE
               OR EXISTS (SELECT 1 FROM common_names c3
                          WHERE c3.fw_taxon_id = t.fw_taxon_id
                            AND c3.name LIKE ? COLLATE NOCASE)
               OR EXISTS (SELECT 1 FROM taxon_synonyms s
                          WHERE s.fw_taxon_id = t.fw_taxon_id
                            AND s.name LIKE ? COLLATE NOCASE)
            ORDER BY rankOrder, t.scientific_name
            LIMIT ?
            """,
            listOf(language, like, like, contains, contains, contains, limit),
        ) { mapTaxon(it) }
    }

    fun allSources(): List<SourceRef> = driver.query(
        "SELECT source_id, title, url, license, license_url, citation, retrieved_on " +
            "FROM sources ORDER BY title",
    ) { mapSource(it) }

    private fun sourcesById(ids: List<String>): List<SourceRef> {
        val placeholders = ids.joinToString(",") { "?" }
        return driver.query(
            "SELECT source_id, title, url, license, license_url, citation, " +
                "retrieved_on FROM sources WHERE source_id IN ($placeholders) " +
                "ORDER BY title",
            ids,
        ) { mapSource(it) }
    }

    fun packMetadata(): Map<String, String> = driver.query(
        "SELECT key, value FROM pack_metadata",
    ) { (it.getString(0) ?: "") to (it.getString(1) ?: "") }.toMap()

    fun classCount(): Int =
        driver.queryOne("SELECT count(*) FROM model_classes") { it.getInt(0) ?: 0 } ?: 0

    /**
     * `(count, min(class_index), max(class_index))` in one query, so the
     * caller can confirm class indices are dense from 0 - not merely that
     * there are the right *number* of rows.
     *
     * That distinction matters: a `model_classes` table with indices
     * `0, 1, 2, ..., 1976, 5000` has exactly as many rows as a dense one, so
     * a count-only check (`repository.classCount() == spec.numClasses`,
     * `IdentificationEngine.open`'s check before this existed) would accept
     * it - and then every class beyond the gap resolves to the wrong taxon,
     * silently, which is exactly the failure `docs/SECURITY.md` §2.2 already
     * documented as required to prevent, without the engine actually doing
     * so at load time.
     */
    fun classIndexDensity(): Triple<Int, Int?, Int?> =
        driver.queryOne(
            "SELECT count(*), min(class_index), max(class_index) FROM model_classes"
        ) { r -> Triple(r.getInt(0) ?: 0, r.getInt(1), r.getInt(2)) }
            ?: Triple(0, null, null)

    /** Class index -> genus key, for the ranker's coarse fallback. */
    fun genusIndexByClass(): Pair<IntArray, Array<String>> {
        val rows = driver.query(
            "SELECT m.class_index, coalesce(t.genus, t.scientific_name) " +
                "FROM model_classes m JOIN taxa t ON t.fw_taxon_id = m.fw_taxon_id " +
                "ORDER BY m.class_index",
        ) { (it.getInt(0) ?: 0) to (it.getString(1) ?: "?") }
        val genera = LinkedHashMap<String, Int>()
        val arr = IntArray(rows.size)
        for ((idx, genus) in rows) {
            if (idx in arr.indices) arr[idx] = genera.getOrPut(genus) { genera.size }
        }
        return arr to genera.keys.toTypedArray()
    }

    private fun mapTaxon(r: SqliteDriver.Row) = Taxon(
        id = r.getLong(0) ?: 0L,
        scientificName = r.getString(1) ?: "",
        rank = parseRank(r.getString(2)),
        genus = r.getString(3),
        family = r.getString(4),
        order = r.getString(5),
        klass = r.getString(6),
        commonName = r.getString(7),
    )

    private fun mapSource(r: SqliteDriver.Row) = SourceRef(
        sourceId = r.getString(0) ?: "",
        title = r.getString(1) ?: "",
        url = r.getString(2),
        license = r.getString(3) ?: "",
        licenseUrl = r.getString(4),
        citation = r.getString(5) ?: "",
        retrievedOn = r.getString(6) ?: "",
    )

    private fun parseRank(s: String?): Rank = when (s?.lowercase()) {
        "genus" -> Rank.GENUS
        "family" -> Rank.FAMILY
        "order" -> Rank.ORDER
        "class" -> Rank.CLASS
        else -> Rank.SPECIES
    }

    override fun close() = driver.close()
}
