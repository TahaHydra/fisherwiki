-- FisherWiki pack species database schema.
--
-- Single source of truth: tools/fwdata/speciesdb.py loads this file, and
-- Gradle copies it into :core's test resources so the Kotlin tests build
-- their fixtures from the same DDL that ships. Duplicating the schema in
-- two languages would let them drift, and a drifted schema means the app
-- reads a column that means something else.
--
-- Every fact table has a NOT NULL source_id: no biological claim exists in
-- this database without a citable source.

PRAGMA journal_mode = DELETE;   -- a shipped DB should be a single file
PRAGMA foreign_keys = ON;

CREATE TABLE sources (
    source_id     TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    publisher     TEXT,
    url           TEXT,
    license       TEXT NOT NULL,
    license_url   TEXT,
    retrieved_on  TEXT NOT NULL,
    citation      TEXT NOT NULL,
    notes         TEXT
);

CREATE TABLE taxa (
    fw_taxon_id      INTEGER PRIMARY KEY,
    scientific_name  TEXT NOT NULL,
    rank             TEXT NOT NULL,
    genus            TEXT,
    family           TEXT,
    taxon_order      TEXT,
    class            TEXT,
    authorship       TEXT,
    gbif_taxon_id    INTEGER,
    inat_taxon_id    INTEGER,
    worms_aphia_id   INTEGER,
    wikidata_qid     TEXT,
    source_id        TEXT NOT NULL REFERENCES sources(source_id)
);

CREATE TABLE taxon_synonyms (
    fw_taxon_id  INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,        -- 'synonym' | 'misspelling' | 'former'
    source_id    TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (fw_taxon_id, name)
);

CREATE TABLE common_names (
    fw_taxon_id  INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    lang         TEXT NOT NULL,        -- BCP-47: 'en', 'fr', 'en-GB'
    name         TEXT NOT NULL,
    preferred    INTEGER NOT NULL DEFAULT 0,
    source_id    TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (fw_taxon_id, lang, name)
);

CREATE TABLE regions (
    region_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    water       TEXT NOT NULL,
    boxes_json  TEXT NOT NULL
);

CREATE TABLE taxon_regions (
    fw_taxon_id   INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    region_id     TEXT NOT NULL REFERENCES regions(region_id),
    observations  INTEGER NOT NULL,
    share         REAL NOT NULL,        -- fraction of this taxon's records here
    source_id     TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (fw_taxon_id, region_id)
);

CREATE TABLE habitats (
    fw_taxon_id  INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    water        TEXT NOT NULL,         -- freshwater | marine | brackish
    detail       TEXT,
    source_id    TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (fw_taxon_id, water)
);

CREATE TABLE traits (
    fw_taxon_id  INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    key          TEXT NOT NULL,         -- 'max_length_cm', 'typical_length_cm', ...
    value_num    REAL,
    value_text   TEXT,
    unit         TEXT,
    source_id    TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (fw_taxon_id, key, source_id)
);

CREATE TABLE diagnostic_features (
    fw_taxon_id  INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    ordinal      INTEGER NOT NULL,
    feature      TEXT NOT NULL,
    body_part    TEXT,
    source_id    TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (fw_taxon_id, ordinal)
);

CREATE TABLE similar_species (
    fw_taxon_id        INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    other_fw_taxon_id  INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    difference         TEXT NOT NULL,
    confusion_rate     REAL,            -- measured on our held-out test set
    source_id          TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (fw_taxon_id, other_fw_taxon_id)
);

CREATE TABLE safety_warnings (
    fw_taxon_id  INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    kind         TEXT NOT NULL,         -- venomous_spines | toxic_flesh | bite | ...
    severity     TEXT NOT NULL,         -- info | caution | danger
    summary      TEXT NOT NULL,
    detail       TEXT,
    applies_to   TEXT,                  -- 'species' | 'genus' | 'family'
    source_id    TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (fw_taxon_id, kind)
);

CREATE TABLE model_classes (
    class_index  INTEGER PRIMARY KEY,
    fw_taxon_id  INTEGER NOT NULL REFERENCES taxa(fw_taxon_id),
    train_images INTEGER NOT NULL,
    train_observations INTEGER NOT NULL,
    -- Measured on the held-out test set; shown in the UI as a per-species
    -- reliability hint so a weak class is not presented like a strong one.
    test_precision REAL,
    test_recall    REAL,
    test_support   INTEGER
);

CREATE TABLE model_metadata (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE TABLE pack_metadata (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE INDEX idx_common_names_name   ON common_names(name COLLATE NOCASE);
CREATE INDEX idx_common_names_taxon  ON common_names(fw_taxon_id);
CREATE INDEX idx_taxa_sci            ON taxa(scientific_name COLLATE NOCASE);
CREATE INDEX idx_synonyms_name       ON taxon_synonyms(name COLLATE NOCASE);
CREATE INDEX idx_model_classes_taxon ON model_classes(fw_taxon_id);
CREATE INDEX idx_taxon_regions_reg   ON taxon_regions(region_id);
