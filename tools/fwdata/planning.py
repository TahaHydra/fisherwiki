"""Decide pack composition from measured data coverage, not from intuition.

The brief proposes a regional split (Europe Freshwater, North America Atlantic,
...) but asks that the real division be determined from available training data,
taxonomic ambiguity, geographic range, model size and expected accuracy. This
module produces the evidence for that decision and a concrete proposal.

Key ideas
---------
*Species tiers.* A species is only a **species-level class** if it has enough
images *and* enough independent observations. Photo count alone is misleading:
80 photos from 3 observations teach the model three individual fish, not a
species. Taxa that fail the bar are folded up to genus, and taxa whose genus
also fails are folded to family, so the model says "some kind of *Sebastes*"
instead of confidently naming the wrong rockfish.

*Region membership is measured.* A species belongs to a region when a
meaningful share of its observations fall inside that region's boxes, not
because a human wrote it on a list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from .config import PATHS
from .licenses import Policy
from .regions import REGIONS

#: Minimum evidence for a species-level class. Both must hold.
MIN_IMAGES_SPECIES = 40
MIN_OBSERVATIONS_SPECIES = 25

#: Weaker bar for genus-level classes, which are a fallback rather than a claim.
MIN_IMAGES_GENUS = 60
MIN_OBSERVATIONS_GENUS = 30

#: A species counts as present in a region at this share of its observations.
REGION_SHARE = 0.05
REGION_MIN_OBSERVATIONS = 15


@dataclass
class Tiering:
    species_level: int = 0
    genus_level: int = 0
    family_level: int = 0
    dropped: int = 0
    images_species: int = 0
    images_genus: int = 0
    images_family: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _candidates_rel(con: duckdb.DuckDBPyConnection, parquet: Path, policy: Policy) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW cand AS
        SELECT * FROM read_parquet('{parquet.as_posix()}')
        WHERE license IN {policy.sql_in_list()}
        """
    )


def species_stats(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE sp AS
        SELECT canonical_name, rank, family_name, genus_name, order_name, class_name,
               any_value(inat_taxon_id)      AS inat_taxon_id,
               count(*)                      AS images,
               count(DISTINCT group_key)     AS observations,
               count(DISTINCT observer_key)  AS observers
        FROM cand
        GROUP BY canonical_name, rank, family_name, genus_name, order_name, class_name
        """
    )


def assign_tiers(con: duckdb.DuckDBPyConnection) -> Tiering:
    """Label every taxon species / genus / family level, or dropped."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE tiered AS
        WITH genus_roll AS (
            SELECT genus_name,
                   sum(images)                AS g_images,
                   sum(observations)          AS g_observations
            FROM sp WHERE genus_name IS NOT NULL GROUP BY genus_name
        ),
        family_roll AS (
            SELECT family_name,
                   sum(images)                AS f_images,
                   sum(observations)          AS f_observations
            FROM sp WHERE family_name IS NOT NULL GROUP BY family_name
        )
        SELECT sp.*, gr.g_images, gr.g_observations, fr.f_images, fr.f_observations,
            CASE
              WHEN sp.rank = 'species'
                   AND sp.images       >= {MIN_IMAGES_SPECIES}
                   AND sp.observations >= {MIN_OBSERVATIONS_SPECIES}
                THEN 'species'
              WHEN sp.genus_name IS NOT NULL
                   AND gr.g_images       >= {MIN_IMAGES_GENUS}
                   AND gr.g_observations >= {MIN_OBSERVATIONS_GENUS}
                THEN 'genus'
              WHEN sp.family_name IS NOT NULL
                   AND fr.f_images       >= {MIN_IMAGES_GENUS}
                   AND fr.f_observations >= {MIN_OBSERVATIONS_GENUS}
                THEN 'family'
              ELSE 'dropped'
            END AS tier,
            CASE
              WHEN sp.rank = 'species'
                   AND sp.images       >= {MIN_IMAGES_SPECIES}
                   AND sp.observations >= {MIN_OBSERVATIONS_SPECIES}
                THEN sp.canonical_name
              WHEN sp.genus_name IS NOT NULL
                   AND gr.g_images       >= {MIN_IMAGES_GENUS}
                   AND gr.g_observations >= {MIN_OBSERVATIONS_GENUS}
                THEN sp.genus_name
              WHEN sp.family_name IS NOT NULL
                   AND fr.f_images       >= {MIN_IMAGES_GENUS}
                   AND fr.f_observations >= {MIN_OBSERVATIONS_GENUS}
                THEN sp.family_name
              ELSE NULL
            END AS class_label
        FROM sp
        LEFT JOIN genus_roll  gr ON gr.genus_name  = sp.genus_name
        LEFT JOIN family_roll fr ON fr.family_name = sp.family_name
        """
    )
    t = Tiering()
    for tier, n, imgs in con.execute(
        "SELECT tier, count(*), sum(images) FROM tiered GROUP BY tier"
    ).fetchall():
        if tier == "species":
            t.species_level, t.images_species = n, int(imgs or 0)
        elif tier == "genus":
            t.genus_level, t.images_genus = n, int(imgs or 0)
        elif tier == "family":
            t.family_level, t.images_family = n, int(imgs or 0)
        else:
            t.dropped = n
    return t


def region_membership(con: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    """Measure which taxa have real observation support in each region."""
    out: dict[str, dict] = {}
    for rid, reg in REGIONS.items():
        pred = reg.sql_predicate()
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE reg_counts AS
            SELECT canonical_name,
                   count(DISTINCT group_key) FILTER (WHERE {pred}) AS in_region,
                   count(DISTINCT group_key)                       AS total
            FROM cand
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            GROUP BY canonical_name
            """
        )
        row = con.execute(
            f"""
            SELECT
              count(*) FILTER (WHERE in_region >= {REGION_MIN_OBSERVATIONS}
                               AND in_region >= {REGION_SHARE} * total)  AS taxa,
              sum(in_region)                                             AS observations
            FROM reg_counts
            """
        ).fetchone()
        classes = con.execute(
            f"""
            SELECT count(*) FROM reg_counts r
            JOIN tiered t ON t.canonical_name = r.canonical_name
            WHERE t.tier = 'species'
              AND r.in_region >= {REGION_MIN_OBSERVATIONS}
              AND r.in_region >= {REGION_SHARE} * r.total
            """
        ).fetchone()[0]
        out[rid] = {
            "name": reg.name,
            "water": reg.water,
            "taxa_with_support": int(row[0] or 0),
            "species_level_classes": int(classes or 0),
            "observations_in_region": int(row[1] or 0),
            "rationale": reg.rationale,
        }
    return out


def plan_packs(
    *,
    policy: Policy,
    parquet: Path | None = None,
    log=print,
) -> dict:
    """Produce the evidence table and a pack proposal."""
    p = Path(parquet or (PATHS.work / "candidates_inaturalist.parquet"))
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `tools/dataset.py extract` first")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    _candidates_rel(con, p, policy)
    species_stats(con)
    tier = assign_tiers(con)

    total_images = con.execute("SELECT count(*) FROM cand").fetchone()[0]
    log(f"licensed candidate photos under {policy.name}: {total_images:,}")
    log("")
    log("class tiering (evidence bars: species >= "
        f"{MIN_IMAGES_SPECIES} imgs & {MIN_OBSERVATIONS_SPECIES} obs)")
    log(f"  species-level classes : {tier.species_level:6,}  "
        f"({tier.images_species:,} images)")
    log(f"  genus-level fallback  : {tier.genus_level:6,}  "
        f"({tier.images_genus:,} images)")
    log(f"  family-level fallback : {tier.family_level:6,}  "
        f"({tier.images_family:,} images)")
    log(f"  dropped (too sparse)  : {tier.dropped:6,}")
    log("")

    regions = region_membership(con)
    log(f"{'region':30} {'species classes':>16} {'taxa':>8} {'obs in region':>14}")
    for rid, r in sorted(
        regions.items(), key=lambda kv: -kv[1]["species_level_classes"]
    ):
        log(f"{rid:30} {r['species_level_classes']:16,} "
            f"{r['taxa_with_support']:8,} {r['observations_in_region']:14,}")

    top_families = con.execute(
        """
        SELECT family_name, count(*) AS taxa, sum(images) AS images
        FROM sp WHERE family_name IS NOT NULL
        GROUP BY family_name ORDER BY images DESC LIMIT 25
        """
    ).fetchall()

    report = {
        "policy": policy.name,
        "total_licensed_images": int(total_images),
        "tiering": tier.as_dict(),
        "thresholds": {
            "min_images_species": MIN_IMAGES_SPECIES,
            "min_observations_species": MIN_OBSERVATIONS_SPECIES,
            "min_images_genus": MIN_IMAGES_GENUS,
            "min_observations_genus": MIN_OBSERVATIONS_GENUS,
            "region_share": REGION_SHARE,
            "region_min_observations": REGION_MIN_OBSERVATIONS,
        },
        "regions": regions,
        "top_families": [
            {"family": f, "taxa": int(t), "images": int(i)} for f, t, i in top_families
        ],
    }
    con.close()
    return report
