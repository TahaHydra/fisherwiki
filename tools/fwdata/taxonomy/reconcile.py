"""Build the canonical fish taxonomy by reconciling iNaturalist with GBIF.

Strategy, and why it is this way round
-------------------------------------
The **taxon set** is driven by iNaturalist, not GBIF, for two reasons:

1. Our images come from iNaturalist and are labelled with iNaturalist taxon
   ids, so the set of things we can actually learn is exactly iNaturalist's.
2. GBIF's backbone does not link fish orders to class ``Actinopterygii``
   (``Perciformes``'s parent is phylum Chordata and ``classKey`` is NULL for
   its descendants - verified against the live API). Selecting fish from GBIF
   by class is therefore impossible without a hand-maintained order list.

GBIF is then joined **by canonical name** to contribute what iNaturalist's
export lacks: an explicit synonym -> accepted mapping, authorship, and a second
opinion on accepted status. Disagreements are recorded, not silently resolved.

Outputs
-------
``taxa``            one row per accepted fish taxon with the full rank ladder
``synonyms``        alternative names that should resolve to an accepted taxon
``reconcile_report`` counts and disagreements, written to the work dir
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from ..config import PATHS
from ..sources import gbif
from ..sources import inaturalist as inat
from .names import canonical_form
from .registry import TaxonRegistry

#: Ranks we keep. Everything below subspecies is folded into its parent.
KEEP_RANKS = ("species", "subspecies", "genus", "family", "order", "class")


@dataclass
class ReconcileStats:
    inat_fish_taxa: int = 0
    inat_species: int = 0
    gbif_matched: int = 0
    gbif_synonyms: int = 0
    status_disagreements: int = 0
    newly_minted: int = 0
    registry_size: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _register_udfs(con: duckdb.DuckDBPyConnection) -> None:
    """Define the canonicaliser as SQL macros rather than a Python UDF.

    The first implementation registered :func:`canonical_form` as a scalar UDF.
    That is correct but unusably slow: it crosses the Python boundary once per
    row over 5.5M GBIF rows, and the job had not finished after 10 minutes.

    These macros reproduce the same result for the inputs that actually occur
    in these two sources, both of which supply authorship-free names already
    (GBIF ``canonical_name``, iNaturalist ``name``). The remaining work is:
    strip parenthesised subgenus, collapse whitespace, then capitalise only the
    first letter - which is exactly right for ``Genus epithet epithet``.

    ``tests/test_taxonomy_sql.py`` asserts that these macros agree with the
    Python implementation on a sample of real names drawn from both sources, so
    the two cannot silently drift apart and mint mismatched IDs.
    """
    con.execute(
        """
        CREATE OR REPLACE MACRO nz(s) AS
            trim(regexp_replace(
                 regexp_replace(coalesce(s, ''), '\\([^)]*\\)', ' ', 'g'),
                 '[[:space:]]+', ' ', 'g'));

        CREATE OR REPLACE MACRO canon(s) AS
            CASE WHEN nz(s) = '' THEN ''
                 ELSE upper(substr(nz(s), 1, 1)) || lower(substr(nz(s), 2))
            END;
        """
    )


def build_inat_fish_table(con: duckdb.DuckDBPyConnection) -> None:
    """Materialise every iNaturalist taxon under a fish root, with its ladder.

    ``ancestry`` is a ``/``-joined path of ancestor taxon ids, so membership is
    a string-prefix test and the rank ladder is a join against the same table.
    """
    roots = list(inat.FISH_ROOT_TAXA)
    pred = " OR ".join(
        f"(t.ancestry = '{r}' OR t.ancestry LIKE '%/{r}' OR t.ancestry LIKE '%/{r}/%')"
        for r in roots
    )
    con.execute(f"CREATE OR REPLACE TEMP VIEW inat_all AS SELECT * FROM {inat.read_csv_sql('taxa')}")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE inat_fish AS
        SELECT
            t.taxon_id,
            t.name,
            t.rank,
            t.rank_level,
            t.active = 'true' AS active,
            t.ancestry,
            -- ancestor ids as a list, for ladder lookups
            string_split(t.ancestry, '/') AS ancestor_ids
        FROM inat_all t
        WHERE ({pred})
          AND t.rank IN ({', '.join(f"'{r}'" for r in KEEP_RANKS)})
        """
    )
    # Rank ladder: join each taxon to its ancestors and pivot the useful ranks.
    con.execute(
        """
        CREATE OR REPLACE TABLE inat_ladder AS
        WITH exploded AS (
            SELECT f.taxon_id, TRY_CAST(a AS BIGINT) AS anc_id
            FROM inat_fish f, UNNEST(f.ancestor_ids) AS u(a)
            WHERE TRY_CAST(a AS BIGINT) IS NOT NULL
        ),
        named AS (
            SELECT e.taxon_id, p.rank AS anc_rank, p.name AS anc_name
            FROM exploded e
            JOIN inat_all p ON p.taxon_id = e.anc_id
            WHERE p.rank IN ('class','order','family','genus')
        )
        SELECT
            taxon_id,
            MAX(CASE WHEN anc_rank='class'  THEN anc_name END) AS class_name,
            MAX(CASE WHEN anc_rank='order'  THEN anc_name END) AS order_name,
            MAX(CASE WHEN anc_rank='family' THEN anc_name END) AS family_name,
            MAX(CASE WHEN anc_rank='genus'  THEN anc_name END) AS genus_name
        FROM named GROUP BY taxon_id
        """
    )


def build_gbif_fish_table(con: duckdb.DuckDBPyConnection) -> None:
    """Pull GBIF rows whose canonical name matches an iNaturalist fish name.

    Joining by name (rather than by class) is forced by GBIF's broken fish
    class linkage; it also means we pick up synonyms whose own ladder is
    incomplete.
    """
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW gbif_all AS SELECT * FROM {gbif.read_backbone_sql()}"
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gbif_fish AS
        WITH wanted AS (
            SELECT DISTINCT canon(name) AS c FROM inat_fish WHERE canon(name) <> ''
        ),
        -- accepted usages whose name we already know about
        direct AS (
            SELECT g.* FROM gbif_all g
            JOIN wanted w ON canon(g.canonical_name) = w.c
            WHERE g.canonical_name IS NOT NULL
        ),
        -- plus every synonym that points at one of those accepted usages
        syn AS (
            SELECT g.* FROM gbif_all g
            WHERE g.is_synonym = 't'
              AND g.parent_or_accepted_key IN (
                  SELECT taxon_key FROM direct WHERE is_synonym = 'f'
              )
        )
        SELECT * FROM direct
        UNION ALL
        SELECT * FROM syn
        """
    )


def reconcile(
    registry_path: Path | None = None,
    *,
    out_dir: Path | None = None,
    save: bool = True,
    log=lambda _m: None,
) -> ReconcileStats:
    """Run the full reconciliation and update the canonical taxon registry."""
    out_dir = Path(out_dir or PATHS.work)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    _register_udfs(con)

    log("scanning iNaturalist taxa for fish clades")
    build_inat_fish_table(con)
    log("joining GBIF backbone by canonical name")
    build_gbif_fish_table(con)

    stats = ReconcileStats()
    stats.inat_fish_taxa = con.execute("SELECT count(*) FROM inat_fish").fetchone()[0]
    stats.inat_species = con.execute(
        "SELECT count(*) FROM inat_fish WHERE rank='species' AND active"
    ).fetchone()[0]

    # --- accepted taxa table ------------------------------------------------
    con.execute(
        """
        CREATE OR REPLACE TABLE taxa_joined AS
        SELECT
            f.taxon_id           AS inat_taxon_id,
            f.name               AS inat_name,
            canon(f.name)        AS canonical_name,
            f.rank,
            f.active,
            l.class_name, l.order_name, l.family_name, l.genus_name,
            g.taxon_key          AS gbif_taxon_key,
            g.status             AS gbif_status,
            g.is_synonym         AS gbif_is_synonym,
            g.parent_or_accepted_key AS gbif_accepted_key,
            g.authorship,
            g.year
        FROM inat_fish f
        LEFT JOIN inat_ladder l USING (taxon_id)
        LEFT JOIN (
            -- Exactly one GBIF row per canonical name, or this join fans out
            -- and silently duplicates every image of the affected taxon.
            --
            -- Cross-kingdom homonyms are the cause and they are not rare:
            -- `Argentina` is a herring-smelt genus *and* a plant genus,
            -- `Rondeletia` likewise, `Monotaxis` matched three GBIF usages.
            -- Restricting to kingdom Animalia (kingdom_key = 1) removes the
            -- botanical twins; the row_number() is a deterministic tie-break
            -- for anything still ambiguous within the animal kingdom.
            SELECT * FROM (
                SELECT g.*,
                       row_number() OVER (
                           PARTITION BY canon(g.canonical_name)
                           ORDER BY g.taxon_key
                       ) AS rn
                FROM gbif_fish g
                WHERE g.is_synonym = 'f'
                  AND g.status = 'ACCEPTED'
                  AND g.kingdom_key = 1
            ) WHERE rn = 1
        ) g ON canon(g.canonical_name) = canon(f.name)
        """
    )
    # Fail loudly rather than let a fanout through to the corpus.
    fanout = con.execute(
        "SELECT count(*) FROM (SELECT inat_taxon_id FROM taxa_joined "
        "GROUP BY inat_taxon_id HAVING count(*) > 1)"
    ).fetchone()[0]
    if fanout:
        raise RuntimeError(
            f"taxa_joined has {fanout} iNat taxa with multiple rows; the GBIF "
            "join is fanning out and would duplicate training images"
        )
    stats.gbif_matched = con.execute(
        "SELECT count(*) FROM taxa_joined WHERE gbif_taxon_key IS NOT NULL"
    ).fetchone()[0]

    # --- synonyms -----------------------------------------------------------
    con.execute(
        """
        CREATE OR REPLACE TABLE synonyms AS
        SELECT DISTINCT
            canon(acc.canonical_name) AS accepted_canonical,
            canon(s.canonical_name)   AS synonym_canonical,
            s.scientific_name         AS synonym_scientific_name,
            s.status                  AS synonym_status,
            s.taxon_key               AS gbif_synonym_key,
            'gbif'                    AS source
        FROM gbif_fish s
        JOIN gbif_fish acc ON acc.taxon_key = s.parent_or_accepted_key
        WHERE s.is_synonym = 't'
          AND s.canonical_name IS NOT NULL
          AND canon(s.canonical_name) <> canon(acc.canonical_name)
        """
    )
    stats.gbif_synonyms = con.execute("SELECT count(*) FROM synonyms").fetchone()[0]

    # iNaturalist's inactive taxa look like a second synonym source, but the
    # open-data export carries no inactive -> active pointer. Recovering one by
    # name similarity would invent relationships that the data does not assert,
    # which is precisely the failure mode this module exists to prevent. GBIF
    # supplies real, curated synonymy; we use only that.

    # --- disagreements ------------------------------------------------------
    dis = con.execute(
        """
        SELECT count(*) FROM taxa_joined t
        JOIN gbif_fish g ON canon(g.canonical_name) = t.canonical_name
        WHERE t.active AND g.is_synonym = 't'
        """
    ).fetchone()[0]
    stats.status_disagreements = int(dis)

    # --- mint canonical ids -------------------------------------------------
    log("minting canonical taxon ids")
    reg = TaxonRegistry(registry_path)
    before = len(reg)
    rows = con.execute(
        """
        SELECT canonical_name, rank, inat_taxon_id, gbif_taxon_key
        FROM taxa_joined
        WHERE active AND canonical_name <> ''
        ORDER BY canonical_name
        """
    ).fetchall()
    for canon_name, rank, inat_id, gbif_key in rows:
        rec = reg.ensure(canon_name, rank)
        if rec.inat_taxon_id is None and inat_id is not None:
            rec.inat_taxon_id = int(inat_id)
        if rec.gbif_taxon_id is None and gbif_key is not None:
            rec.gbif_taxon_id = int(gbif_key)
    stats.newly_minted = len(reg) - before
    stats.registry_size = len(reg)
    if save:
        reg.save()

    # --- persist the reconciled tables --------------------------------------
    db_path = out_dir / "taxonomy.duckdb"
    if save:
        if db_path.exists():
            db_path.unlink()
        con.execute(f"ATTACH '{db_path.as_posix()}' AS out")
        for t in ("inat_fish", "inat_ladder", "taxa_joined", "synonyms"):
            con.execute(f"CREATE TABLE out.{t} AS SELECT * FROM {t}")
        con.execute("DETACH out")
        (out_dir / "reconcile_report.json").write_text(
            json.dumps(stats.as_dict(), indent=2), encoding="utf-8"
        )

    con.close()
    return stats
