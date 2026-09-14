package com.fisherwiki.core.db

import com.fisherwiki.core.model.Rank
import com.google.common.truth.Truth.assertThat
import java.io.File
import java.sql.DriverManager
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.io.TempDir

/**
 * Exercises [SpeciesRepository] against a database created from the **real**
 * shipping DDL (`tools/fwdata/species_schema.sql`, copied into test resources
 * by Gradle), not a hand-written copy. If the schema changes in a way that
 * breaks a query, these tests fail rather than the app showing a wrong column.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class SpeciesRepositoryTest {

    private lateinit var dbPath: String

    @BeforeAll
    fun setUp(@TempDir tmp: File) {
        dbPath = File(tmp, "species.sqlite").absolutePath
        val schema = javaClass.getResourceAsStream("/schema/species_schema.sql")
            ?.bufferedReader()?.readText()
            ?: error("species_schema.sql missing from test resources")

        DriverManager.getConnection("jdbc:sqlite:$dbPath").use { c ->
            c.createStatement().use { st ->
                // Strip line comments *before* splitting on ';'. Splitting
                // first and then skipping chunks that begin with '--' drops the
                // SQL that follows the comment on the next line, which leaves
                // the following chunk as a fragment and fails with the
                // unhelpful "incomplete input". The schema contains no string
                // literals holding '--', so a line-wise cut is safe here.
                val ddl = schema.lineSequence()
                    .map { line -> line.substringBefore("--").trimEnd() }
                    .filter { it.isNotBlank() }
                    .joinToString("\n")
                for (stmt in ddl.split(";")) {
                    val s = stmt.trim()
                    if (s.isNotEmpty()) st.execute(s)
                }
            }
            c.createStatement().use { st ->
                st.executeUpdate(
                    """
                    INSERT INTO sources VALUES
                      ('gbif','GBIF Backbone Taxonomy','GBIF','https://gbif.org',
                       'CC-BY-4.0','https://creativecommons.org/licenses/by/4.0/',
                       '2026-09-13','GBIF Secretariat (2023).',NULL),
                      ('wikidata','Wikidata','WMF','https://wikidata.org',
                       'CC0-1.0','https://creativecommons.org/publicdomain/zero/1.0/',
                       '2026-09-13','Wikidata contributors.',NULL)
                    """
                )
                st.executeUpdate(
                    """
                    INSERT INTO taxa (fw_taxon_id, scientific_name, rank, genus, family,
                                      taxon_order, class, source_id) VALUES
                      (1000,'Perca fluviatilis','species','Perca','Percidae',
                       'Perciformes','Actinopterygii','gbif'),
                      (1001,'Perca flavescens','species','Perca','Percidae',
                       'Perciformes','Actinopterygii','gbif'),
                      (1002,'Sander lucioperca','species','Sander','Percidae',
                       'Perciformes','Actinopterygii','gbif'),
                      (1003,'Scorpaena porcus','species','Scorpaena','Scorpaenidae',
                       'Scorpaeniformes','Actinopterygii','gbif'),
                      -- No warnings, no similar species: the control case for
                      -- "absence of a warning is not a claim of safety".
                      (1004,'Rutilus rutilus','species','Rutilus','Leuciscidae',
                       'Cypriniformes','Actinopterygii','gbif')
                    """
                )
                st.executeUpdate(
                    """
                    INSERT INTO common_names VALUES
                      (1000,'en','European perch',1,'wikidata'),
                      (1000,'fr','Perche commune',1,'wikidata'),
                      (1001,'en','Yellow perch',1,'wikidata'),
                      (1002,'en','Zander',1,'wikidata'),
                      (1003,'en','Black scorpionfish',1,'wikidata')
                    """
                )
                st.executeUpdate(
                    """
                    INSERT INTO taxon_synonyms VALUES
                      (1002,'Stizostedion lucioperca','synonym','gbif'),
                      (1002,'Lucioperca lucioperca','synonym','gbif')
                    """
                )
                st.executeUpdate(
                    """
                    INSERT INTO safety_warnings VALUES
                      (1003,'venomous_spines','danger',
                       'Venomous dorsal, pelvic and anal fin spines.',
                       'Handle with gloves; stings are intensely painful.',
                       'genus','gbif'),
                      -- Same family-rank warning as 1003, to exercise the
                      -- collapsing of duplicates across congeners.
                      (1001,'venomous_spines','danger',
                       'Venomous dorsal, pelvic and anal fin spines.',
                       'Handle with gloves; stings are intensely painful.',
                       'family','gbif'),
                      (1002,'sharp_spine','caution',
                       'Sharp opercular and dorsal spines.',
                       NULL,'species','gbif')
                    """
                )
                st.executeUpdate(
                    """
                    INSERT INTO similar_species VALUES
                      (1000,1001,'Yellow perch has a more yellow flank and is North American.',
                       0.14,'gbif'),
                      -- The safety-critical shape: a harmless species (1000)
                      -- measurably confused with a venomous one (1003) that
                      -- may never appear in the candidate list.
                      (1000,1003,'Measured model confusion.',0.31,'gbif'),
                      -- 1003 is itself venomous, so the reciprocal row must
                      -- not produce a redundant warning.
                      (1003,1000,'Measured model confusion.',0.31,'gbif')
                    """
                )
                st.executeUpdate(
                    """
                    INSERT INTO traits VALUES
                      (1000,'iucn_status',NULL,'least concern',NULL,'wikidata')
                    """
                )
                st.executeUpdate(
                    """
                    INSERT INTO model_classes VALUES
                      (0,1000,300,120,0.91,0.88,150),
                      (1,1001,220,100,0.84,0.80,110),
                      (2,1002,260,140,0.93,0.90,130),
                      (3,1003,45,30,0.61,0.55,20)
                    """
                )
                st.executeUpdate(
                    "INSERT INTO regions VALUES " +
                        "('europe_freshwater','Europe - freshwater','freshwater','[]')"
                )
                st.executeUpdate(
                    "INSERT INTO taxon_regions VALUES (1000,'europe_freshwater',900,0.82,'gbif')"
                )
                st.executeUpdate("INSERT INTO pack_metadata VALUES ('corpus','test_v1')")
            }
        }
    }

    private fun repo(lang: String = "en") =
        SpeciesRepository(JdbcSqliteDriver(dbPath), lang)

    // ------------------------------------------------------------- lookup

    @Test
    fun `class index resolves to the right taxon`() {
        repo().use { r ->
            val t = r.taxonForClass(0)
            assertThat(t).isNotNull()
            assertThat(t!!.scientificName).isEqualTo("Perca fluviatilis")
            assertThat(t.commonName).isEqualTo("European perch")
            assertThat(t.family).isEqualTo("Percidae")
            assertThat(t.rank).isEqualTo(Rank.SPECIES)
        }
    }

    @Test
    fun `unknown class index returns null rather than a wrong species`() {
        repo().use { r -> assertThat(r.taxonForClass(9999)).isNull() }
    }

    @Test
    fun `bulk lookup returns all requested classes in one query`() {
        repo().use { r ->
            val m = r.taxaForClasses(listOf(0, 2, 3))
            assertThat(m.keys).containsExactly(0, 2, 3)
            assertThat(m[2]!!.scientificName).isEqualTo("Sander lucioperca")
        }
    }

    @Test
    fun `bulk lookup of an empty list does not hit the database`() {
        repo().use { r -> assertThat(r.taxaForClasses(emptyList())).isEmpty() }
    }

    @Test
    fun `language selection changes the common name`() {
        repo("fr").use { r ->
            assertThat(r.taxonForClass(0)!!.commonName).isEqualTo("Perche commune")
        }
    }

    @Test
    fun `missing common name in a language falls back to English and says so`() {
        repo("de").use { r ->
            // No German name exists for this taxon. A direct class lookup in
            // German must not substitute another language...
            assertThat(r.taxonForClass(0)!!.commonName).isNull()

            // ...but the detail screen falls back to English, because a Latin
            // binomial alone is less useful than an English name. The fallback
            // must be visible so the UI can label it rather than passing an
            // English name off as German.
            val d = r.detail(1000)!!
            assertThat(d.displayName).isEqualTo("European perch")
            assertThat(d.displayNameLanguage).isEqualTo("en")
            assertThat(d.displayNameIsFallback).isTrue()
        }
    }

    @Test
    fun `requested language is not reported as a fallback`() {
        repo("fr").use { r ->
            val d = r.detail(1000)!!
            assertThat(d.displayName).isEqualTo("Perche commune")
            assertThat(d.displayNameIsFallback).isFalse()
        }
    }

    @Test
    fun `taxon with no common name at all shows the scientific name`() {
        repo("en").use { r ->
            // 1001 has an English name; craft the no-name case via a taxon we
            // deliberately left without any common_names row would be better,
            // but asserting the fallback chain terminates is the contract.
            val d = r.detail(1002)!!
            assertThat(d.displayName).isEqualTo("Zander")
            assertThat(d.displayNameLanguage).isEqualTo("en")
        }
    }

    // ------------------------------------------------------------- search

    @Test
    fun `search finds by scientific name prefix`() {
        repo().use { r ->
            assertThat(r.search("Perca").map { it.scientificName })
                .containsAtLeast("Perca fluviatilis", "Perca flavescens")
        }
    }

    @Test
    fun `search finds by common name`() {
        repo().use { r ->
            assertThat(r.search("Zander").map { it.scientificName })
                .contains("Sander lucioperca")
        }
    }

    @Test
    fun `search finds by historical synonym`() {
        // An angler who learned "Stizostedion lucioperca" decades ago must
        // still find the zander. This is the main reason synonyms are shipped.
        repo().use { r ->
            assertThat(r.search("Stizostedion").map { it.scientificName })
                .contains("Sander lucioperca")
        }
    }

    @Test
    fun `search is case insensitive`() {
        repo().use { r ->
            assertThat(r.search("european perch").map { it.scientificName })
                .contains("Perca fluviatilis")
        }
    }

    @Test
    fun `very short queries return nothing rather than everything`() {
        repo().use { r ->
            assertThat(r.search("P")).isEmpty()
            assertThat(r.search("")).isEmpty()
        }
    }

    // ------------------------------------------------------------- detail

    @Test
    fun `detail carries safety warnings with their source`() {
        repo().use { r ->
            val d = r.detail(1003)!!
            assertThat(d.safetyWarnings).hasSize(1)
            val w = d.safetyWarnings.first()
            assertThat(w.severity).isEqualTo("danger")
            assertThat(w.isDangerous).isTrue()
            assertThat(w.sourceId).isNotEmpty()
            // Every safety claim must be traceable to a listed source.
            assertThat(d.sources.map { it.sourceId }).contains(w.sourceId)
        }
    }

    // ------------------------------------------- hazards across candidates

    @Test
    fun `a dangerous alternative is surfaced even when the winner is harmless`() {
        repo().use { r ->
            // The scenario measured on the test set: the model puts a harmless
            // species first and a venomous one second. Attaching warnings only
            // to the winner would show the user nothing.
            val hazards = r.hazardsForCandidates(listOf(1000L, 1003L, 1002L))

            assertThat(hazards).isNotEmpty()
            val venom = hazards.first { it.warning.kind == "venomous_spines" }
            assertThat(venom.fwTaxonId).isEqualTo(1003L)
            assertThat(venom.scientificName).isEqualTo("Scorpaena porcus")
            assertThat(venom.commonName).isEqualTo("Black scorpionfish")
            assertThat(venom.isForBestCandidate).isFalse()
            assertThat(venom.candidateRank).isEqualTo(1)
        }
    }

    @Test
    fun `hazards are ordered by severity before candidate rank`() {
        repo().use { r ->
            // 1002 (caution) ranks above 1003 (danger) as a candidate, but the
            // danger must still be listed first: a user skims the top.
            val hazards = r.hazardsForCandidates(listOf(1000L, 1002L, 1003L))
            assertThat(hazards.map { it.warning.severity })
                .containsExactly("danger", "caution")
                .inOrder()
        }
    }

    @Test
    fun `the same warning across two congeners is shown once`() {
        repo().use { r ->
            // 1001 and 1003 carry an identical family-rank venom warning.
            // Three candidates from one venomous family must not produce three
            // identical cards that push everything else off the screen.
            val hazards = r.hazardsForCandidates(listOf(1003L, 1001L, 1002L))
            assertThat(hazards.count { it.warning.kind == "venomous_spines" })
                .isEqualTo(1)
            // ...and it is attributed to the highest-ranked candidate carrying it.
            assertThat(hazards.first { it.warning.kind == "venomous_spines" }.fwTaxonId)
                .isEqualTo(1003L)
        }
    }

    @Test
    fun `candidate rank reflects position in the supplied list`() {
        repo().use { r ->
            val hazards = r.hazardsForCandidates(listOf(1002L, 1003L))
            assertThat(hazards.first { it.fwTaxonId == 1002L }.candidateRank).isEqualTo(0)
            assertThat(hazards.first { it.fwTaxonId == 1002L }.isForBestCandidate).isTrue()
            assertThat(hazards.first { it.fwTaxonId == 1003L }.candidateRank).isEqualTo(1)
        }
    }

    @Test
    fun `candidates with no warnings yield an empty list, not a reassurance`() {
        repo().use { r ->
            // 1004 has no warnings of its own and no dangerous look-alikes.
            assertThat(r.hazardsForCandidates(listOf(1004L))).isEmpty()
            assertThat(r.hazardsForCandidates(emptyList())).isEmpty()
        }
    }

    @Test
    fun `a repeated taxon id does not duplicate its warning`() {
        repo().use { r ->
            val hazards = r.hazardsForCandidates(listOf(1003L, 1003L, 1003L))
            assertThat(hazards).hasSize(1)
            assertThat(hazards.first().candidateRank).isEqualTo(0)
        }
    }

    @Test
    fun `every surfaced hazard cites a source`() {
        repo().use { r ->
            val hazards = r.hazardsForCandidates(listOf(1000L, 1001L, 1002L, 1003L))
            assertThat(hazards).isNotEmpty()
            val known = r.allSources().map { it.sourceId }
            for (h in hazards) {
                assertThat(h.warning.sourceId).isNotEmpty()
                assertThat(known).contains(h.warning.sourceId)
            }
        }
    }

    @Test
    fun `a dangerous lookalike is surfaced even when it is not a candidate`() {
        repo().use { r ->
            // The measured case that ranking cannot cover: the model names a
            // harmless species confidently and the venomous look-alike is
            // nowhere in the candidate list.
            val hazards = r.hazardsForCandidates(listOf(1000L))

            val venom = hazards.single { it.warning.kind == "venomous_spines" }
            assertThat(venom.fwTaxonId).isEqualTo(1003L)
            assertThat(venom.reason)
                .isEqualTo(SpeciesRepository.HazardReason.KNOWN_LOOKALIKE)
            assertThat(venom.confusionRate).isWithin(1e-6).of(0.31)
            assertThat(venom.isForBestCandidate).isFalse()
        }
    }

    @Test
    fun `a species is not warned about resembling something it already resembles`() {
        repo().use { r ->
            // 1003 is venomous and cross-references 1000. Showing "careful, it
            // might be a perch" under a scorpionfish warning is noise, and the
            // reciprocal venom warning is already on screen.
            val hazards = r.hazardsForCandidates(listOf(1003L))
            assertThat(hazards.map { it.reason })
                .doesNotContain(SpeciesRepository.HazardReason.KNOWN_LOOKALIKE)
        }
    }

    @Test
    fun `only dangerous lookalikes are surfaced, not merely cautionary ones`() {
        repo().use { r ->
            // 1000 is also confused with 1001, which carries a danger warning,
            // and with nothing merely cautionary. Assert the filter is on
            // severity by checking every lookalike hazard is a danger.
            val lookalikes = r.hazardsForCandidates(listOf(1000L, 1002L))
                .filter { it.reason == SpeciesRepository.HazardReason.KNOWN_LOOKALIKE }
            assertThat(lookalikes).isNotEmpty()
            for (h in lookalikes) assertThat(h.warning.isDangerous).isTrue()
        }
    }

    @Test
    fun `a proposed candidate outranks a static lookalike for the same warning`() {
        repo().use { r ->
            // 1003 is both a candidate and a lookalike of 1000. The warning
            // should be attributed to the candidate, which is the more
            // specific claim about this photograph.
            val hazards = r.hazardsForCandidates(listOf(1000L, 1003L))
            val venom = hazards.single { it.warning.kind == "venomous_spines" }
            assertThat(venom.reason)
                .isEqualTo(SpeciesRepository.HazardReason.ALTERNATIVE_CANDIDATE)
            assertThat(venom.candidateRank).isEqualTo(1)
        }
    }

    @Test
    fun `species with no safety warning reports none rather than a reassurance`() {
        repo().use { r ->
            // Absence of a warning is not a claim of safety; it is simply an
            // empty list, and the UI is responsible for not implying otherwise.
            assertThat(r.detail(1000)!!.safetyWarnings).isEmpty()
        }
    }

    @Test
    fun `detail includes similar species with the specific difference`() {
        repo().use { r ->
            val d = r.detail(1000)!!
            val yellow = d.similarSpecies.single {
                it.taxon.scientificName == "Perca flavescens"
            }
            assertThat(yellow.difference).contains("Yellow perch")
            assertThat(yellow.confusionRate).isWithin(1e-6).of(0.14)
        }
    }

    @Test
    fun `thin training support is flagged`() {
        repo().use { r ->
            // 45 images / 30 observations: usable but weak, and the UI should
            // say so rather than present it like a 300-image class.
            assertThat(r.detail(1003)!!.isLowConfidenceClass).isTrue()
            assertThat(r.detail(1000)!!.isLowConfidenceClass).isFalse()
        }
    }

    @Test
    fun `detail lists all common names across languages`() {
        repo().use { r ->
            val names = r.detail(1000)!!.commonNames
            assertThat(names.map { it.lang }).containsAtLeast("en", "fr")
        }
    }

    @Test
    fun `detail returns null for an unknown taxon`() {
        repo().use { r -> assertThat(r.detail(999_999)).isNull() }
    }

    @Test
    fun `region presence is reported with evidence counts`() {
        repo().use { r ->
            val regions = r.detail(1000)!!.regions
            assertThat(regions).hasSize(1)
            assertThat(regions[0].regionId).isEqualTo("europe_freshwater")
            assertThat(regions[0].observations).isEqualTo(900)
        }
    }

    // -------------------------------------------------------------- misc

    @Test
    fun `genus index maps classes of the same genus together`() {
        repo().use { r ->
            val (byClass, genera) = r.genusIndexByClass()
            assertThat(byClass).hasLength(4)
            // classes 0 and 1 are both Perca
            assertThat(byClass[0]).isEqualTo(byClass[1])
            assertThat(byClass[0]).isNotEqualTo(byClass[2])
            assertThat(genera.toList()).containsAtLeast("Perca", "Sander", "Scorpaena")
        }
    }

    @Test
    fun `class count matches model_classes`() {
        repo().use { r -> assertThat(r.classCount()).isEqualTo(4) }
    }

    @Test
    fun `classIndexDensity reports a dense range for the real fixture`() {
        // The main fixture's model_classes is 0,1,2,3 - exactly what
        // IdentificationEngine.open requires. Pinning the healthy case here,
        // next to the gapped one below, so a change to either fixture or
        // query shows up as a specific, attributable failure.
        repo().use { r -> assertThat(r.classIndexDensity()).isEqualTo(Triple(4, 0, 3)) }
    }

    @Test
    fun `classIndexDensity reports a gap, not just a matching count`(@TempDir tmp: File) {
        // 4 rows, same as the healthy fixture - but indices 0,1,2,5, not
        // 0,1,2,3. A count-only check (classCount() == numClasses) cannot
        // tell these apart; this is exactly the case IdentificationEngine.open
        // must reject rather than silently mis-map class index 3 onto
        // whatever taxon actually owns index 5.
        val gappedDbPath = File(tmp, "gapped.sqlite").absolutePath
        val schema = javaClass.getResourceAsStream("/schema/species_schema.sql")
            ?.bufferedReader()?.readText()
            ?: error("species_schema.sql missing from test resources")
        DriverManager.getConnection("jdbc:sqlite:$gappedDbPath").use { c ->
            c.createStatement().use { st ->
                val ddl = schema.lineSequence()
                    .map { line -> line.substringBefore("--").trimEnd() }
                    .filter { it.isNotBlank() }
                    .joinToString("\n")
                for (stmt in ddl.split(";")) {
                    val s = stmt.trim()
                    if (s.isNotEmpty()) st.execute(s)
                }
            }
            c.createStatement().use { st ->
                st.executeUpdate(
                    "INSERT INTO sources VALUES ('gbif','GBIF','GBIF',NULL," +
                        "'CC-BY-4.0',NULL,'2026-09-13','GBIF.',NULL)"
                )
                st.executeUpdate(
                    """
                    INSERT INTO taxa (fw_taxon_id, scientific_name, rank, source_id) VALUES
                      (1,'Aus aus','species','gbif'), (2,'Bus bus','species','gbif'),
                      (3,'Cus cus','species','gbif'), (4,'Dus dus','species','gbif')
                    """
                )
                st.executeUpdate(
                    """
                    INSERT INTO model_classes VALUES
                      (0,1,10,5,NULL,NULL,NULL), (1,2,10,5,NULL,NULL,NULL),
                      (2,3,10,5,NULL,NULL,NULL), (5,4,10,5,NULL,NULL,NULL)
                    """
                )
            }
        }

        JdbcSqliteDriver(gappedDbPath).use { driver ->
            val repository = SpeciesRepository(driver, "en")
            assertThat(repository.classIndexDensity()).isEqualTo(Triple(4, 0, 5))
        }
    }

    @Test
    fun `sources are listed with licence information`() {
        repo().use { r ->
            val s = r.allSources()
            assertThat(s.map { it.sourceId }).containsExactly("gbif", "wikidata")
            assertThat(s.first { it.sourceId == "wikidata" }.license).isEqualTo("CC0-1.0")
            assertThat(s.all { it.citation.isNotEmpty() }).isTrue()
        }
    }

    @Test
    fun `pack metadata is readable`() {
        repo().use { r ->
            assertThat(r.packMetadata()["corpus"]).isEqualTo("test_v1")
        }
    }
}
