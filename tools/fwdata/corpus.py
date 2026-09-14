"""Candidate extraction: turn the bulk dumps into a licensed, fish-only table.

This is the step that decides *what we could train on*, before anything is
downloaded. It runs entirely over the local bulk exports - no API calls - and
its output is a Parquet table of candidate photographs, one row per photo, each
already carrying its licence, its taxon, its observation group and its
coordinates.

Scale notes
-----------
``observations.csv.gz`` is 13.1 GB and ``photos.csv.gz`` is 20.1 GB compressed.
Neither is ever fully materialised: DuckDB streams both, and the fish filter is
applied to observations first so that the photo scan only has to probe a hash
table of a few million observation UUIDs. Memory is capped and spill goes to
the data volume, not to ``C:``.

Licence handling
----------------
The ``license`` column on ``photos`` is the **photographer's** licence and is
the only one used. It is normalised through :func:`fwdata.licenses.normalize`
in SQL via a mapping table rather than a Python UDF, so the 400M-row scan stays
inside DuckDB. Rows whose licence does not normalise into the closed vocabulary
land as ``UNKNOWN`` and are excluded from every policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from .config import PATHS
from .licenses import License, Policy, normalize
from .sources import inaturalist as inat

#: Quality grades iNaturalist assigns. ``research`` means the community has
#: converged on the identification; ``needs_id`` has not. We keep only research
#: grade for training labels - a wrong label is worse than a missing image.
RESEARCH_GRADE = "research"


@dataclass
class ExtractStats:
    fish_observations: int = 0
    research_grade: int = 0
    candidate_photos: int = 0
    licensed_photos: int = 0
    distinct_taxa: int = 0
    distinct_species: int = 0
    by_license: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "fish_observations": self.fish_observations,
            "research_grade": self.research_grade,
            "candidate_photos": self.candidate_photos,
            "licensed_photos": self.licensed_photos,
            "distinct_taxa": self.distinct_taxa,
            "distinct_species": self.distinct_species,
            "by_license": self.by_license,
            "seconds": round(self.seconds, 1),
        }


def connect(memory_limit_gb: int = 20, threads: int = 8) -> duckdb.DuckDBPyConnection:
    """DuckDB connection configured for large out-of-core joins."""
    con = duckdb.connect()
    tmp = PATHS.work / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    return con


def create_license_map(con: duckdb.DuckDBPyConnection) -> None:
    """Materialise raw-licence -> normalised-licence as a joinable table.

    Built by running the Python normaliser over the *distinct* licence strings
    only (there are a couple of dozen), which keeps the closed vocabulary and
    the fail-closed semantics of :mod:`fwdata.licenses` without paying a Python
    call per photo row.
    """
    distinct = con.execute(
        f"SELECT DISTINCT license FROM {inat.read_csv_sql('photos')}"
    ).fetchall()
    rows = []
    for (raw,) in distinct:
        lic = normalize(raw)
        rows.append((raw, lic.value, lic.url, int(lic.allows_commercial)))
    con.execute(
        "CREATE OR REPLACE TABLE license_map ("
        "  license_raw VARCHAR, license VARCHAR, license_url VARCHAR, commercial BOOLEAN)"
    )
    con.executemany("INSERT INTO license_map VALUES (?,?,?,?)", rows)


def extract_candidates(
    *,
    out_path: Path | None = None,
    quality_grade: str = RESEARCH_GRADE,
    require_coordinates: bool = False,
    memory_limit_gb: int = 20,
    threads: int = 8,
    log=print,
) -> ExtractStats:
    """Build the candidate photograph table from the iNaturalist bulk exports.

    Writes Parquet to ``<work>/candidates_inaturalist.parquet`` and returns
    summary statistics. Idempotent: re-running overwrites the output.
    """
    t0 = time.time()
    out_path = Path(out_path or (PATHS.work / "candidates_inaturalist.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = connect(memory_limit_gb, threads)
    stats = ExtractStats()

    log("[1/6] loading reconciled fish taxonomy")
    tax_db = PATHS.work / "taxonomy.duckdb"
    if not tax_db.exists():
        raise FileNotFoundError(
            f"{tax_db} missing - run `python tools/taxonomy.py reconcile` first"
        )
    con.execute(f"ATTACH '{tax_db.as_posix()}' AS tax (READ_ONLY)")
    con.execute(
        """
        CREATE OR REPLACE TABLE fish_taxa AS
        SELECT t.inat_taxon_id AS taxon_id,
               t.canonical_name,
               t.rank,
               t.class_name, t.order_name, t.family_name, t.genus_name
        FROM tax.taxa_joined t
        WHERE t.active AND t.inat_taxon_id IS NOT NULL
        """
    )
    n_taxa = con.execute("SELECT count(*) FROM fish_taxa").fetchone()[0]
    log(f"      {n_taxa:,} active fish taxa")

    log("[2/6] scanning observations.csv.gz (13.1 GB) for fish records")
    coord_filter = (
        "AND o.latitude IS NOT NULL AND o.longitude IS NOT NULL"
        if require_coordinates
        else ""
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE fish_obs AS
        SELECT o.observation_uuid, o.observer_id, o.latitude, o.longitude,
               o.positional_accuracy, o.taxon_id, o.quality_grade, o.observed_on,
               o.anomaly_score
        FROM {inat.read_csv_sql('observations')} o
        SEMI JOIN fish_taxa f ON f.taxon_id = o.taxon_id
        WHERE o.quality_grade = '{quality_grade}' {coord_filter}
        """
    )
    stats.research_grade = con.execute("SELECT count(*) FROM fish_obs").fetchone()[0]
    stats.fish_observations = stats.research_grade
    log(f"      {stats.research_grade:,} research-grade fish observations "
        f"({time.time()-t0:.0f}s)")

    log("[3/6] building licence map from distinct photo licence strings")
    create_license_map(con)
    log("      " + ", ".join(
        f"{r[0]!r}->{r[1]}" for r in
        con.execute("SELECT license_raw, license FROM license_map").fetchall()
    ))

    log("[4/6] scanning photos.csv.gz (20.1 GB), joining to fish observations")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE fish_photos AS
        SELECT p.photo_uuid, p.photo_id, p.observation_uuid, p.observer_id,
               p.extension, p.license AS license_raw, p.width, p.height, p.position
        FROM {inat.read_csv_sql('photos')} p
        SEMI JOIN fish_obs o ON o.observation_uuid = p.observation_uuid
        """
    )
    stats.candidate_photos = con.execute("SELECT count(*) FROM fish_photos").fetchone()[0]
    log(f"      {stats.candidate_photos:,} candidate photos ({time.time()-t0:.0f}s)")

    log("[5/6] assembling candidate table with provenance columns")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE candidates AS
        SELECT
            'inaturalist'                               AS source_dataset,
            CAST(p.photo_id AS VARCHAR)                 AS source_record_id,
            'https://inaturalist-open-data.s3.amazonaws.com/photos/'
                || p.photo_id || '/{inat.DEFAULT_SIZE}.'
                || lower(coalesce(p.extension, 'jpg'))  AS image_url,
            'https://www.inaturalist.org/observations/' || o.observation_uuid
                                                        AS source_url,
            CAST(o.taxon_id AS VARCHAR)                 AS source_taxon_id,
            f.canonical_name                            AS original_scientific_name,
            f.canonical_name                            AS accepted_scientific_name,
            lm.license                                  AS license,
            p.license_raw                               AS license_raw,
            lm.license_url                              AS license_url,
            coalesce(nullif(ob.name, ''), ob.login)     AS creator,
            coalesce(nullif(ob.name, ''), ob.login)     AS copyright_holder,
            o.observation_uuid                          AS group_key,
            CAST(p.observer_id AS VARCHAR)              AS observer_key,
            o.latitude, o.longitude, o.positional_accuracy,
            o.observed_on, o.quality_grade, o.anomaly_score,
            p.position                                  AS position_in_observation,
            p.width                                     AS declared_width,
            p.height                                    AS declared_height,
            lower(coalesce(p.extension, 'jpg'))         AS ext,
            f.canonical_name, f.rank,
            f.class_name, f.order_name, f.family_name, f.genus_name,
            o.taxon_id                                  AS inat_taxon_id
        FROM fish_photos p
        JOIN fish_obs   o ON o.observation_uuid = p.observation_uuid
        JOIN fish_taxa  f ON f.taxon_id = o.taxon_id
        JOIN license_map lm ON lm.license_raw IS NOT DISTINCT FROM p.license_raw
        LEFT JOIN {inat.read_csv_sql('observers')} ob
               ON ob.observer_id = p.observer_id
        -- iNaturalist attaches some photos to more than one observation, so
        -- photo_id is not unique in photos.csv. Keep one row per photograph:
        -- duplicates would break the candidate primary key and, worse, let the
        -- same image reach two splits through two different group keys.
        QUALIFY row_number() OVER (
            PARTITION BY p.photo_id
            ORDER BY o.observation_uuid, p.position
        ) = 1
        """
    )
    stats.licensed_photos = con.execute(
        f"SELECT count(*) FROM candidates WHERE license <> '{License.ARR.value}'"
        f" AND license <> '{License.UNKNOWN.value}'"
    ).fetchone()[0]
    stats.by_license = {
        lic: int(n)
        for lic, n in con.execute(
            "SELECT license, count(*) FROM candidates GROUP BY license ORDER BY 2 DESC"
        ).fetchall()
    }
    stats.distinct_taxa = con.execute(
        "SELECT count(DISTINCT inat_taxon_id) FROM candidates"
    ).fetchone()[0]
    stats.distinct_species = con.execute(
        "SELECT count(DISTINCT inat_taxon_id) FROM candidates WHERE rank='species'"
    ).fetchone()[0]

    log(f"[6/6] writing {out_path.name}")
    con.execute(
        f"COPY candidates TO '{out_path.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION zstd)"
    )
    stats.seconds = time.time() - t0

    report = PATHS.work / "extract_report.json"
    report.write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    con.close()
    return stats


def species_coverage(
    parquet: Path | None = None, policy: Policy | None = None, min_images: int = 1
) -> list[tuple]:
    """Images-per-species under a licence policy: the key planning table."""
    p = Path(parquet or (PATHS.work / "candidates_inaturalist.parquet"))
    con = duckdb.connect()
    where = f"WHERE license IN {policy.sql_in_list()}" if policy else ""
    return con.execute(
        f"""
        SELECT canonical_name, inat_taxon_id, family_name,
               count(*) AS images,
               count(DISTINCT group_key)   AS observations,
               count(DISTINCT observer_key) AS observers
        FROM read_parquet('{p.as_posix()}')
        {where}
        GROUP BY 1,2,3
        HAVING count(*) >= {int(min_images)}
        ORDER BY images DESC
        """
    ).fetchall()
