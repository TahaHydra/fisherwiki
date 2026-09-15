#!/usr/bin/env python
"""What is actually still available to download, per licence policy and cap.

    python tools/yield_census.py

Reads the iNaturalist candidate census that ``dataset.py extract`` already
produced (4.2M photos, 15,466 species) and reports how many *new* images each
policy and per-species cap would add on top of what is already in the CAS.

Nothing here touches the network: the census is a local Parquet file. The point
is to decide where the next hours of bandwidth go before spending any of them -
the previous acquisition run downloaded 483,271 images and exhausted 96% of the
commercially redistributable pool without that being visible anywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fwdata.config import PATHS  # noqa: E402
from fwdata.licenses import POLICIES  # noqa: E402

CENSUS = "candidates_inaturalist.parquet"


def main(argv=None) -> int:
    import duckdb

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", default=None)
    ap.add_argument("--caps", default="300,1000,100000")
    ap.add_argument("--min-observations", type=int, default=3)
    args = ap.parse_args(argv)

    census = Path(args.census or (PATHS.work / CENSUS))
    if not census.exists():
        raise SystemExit(f"{census} missing - run `tools/dataset.py extract` first")

    con = duckdb.connect(str(PATHS.provenance_db), read_only=True)
    con.execute("PRAGMA threads=8")
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW census AS "
        f"SELECT * FROM read_parquet('{census.as_posix()}')")
    # One row per photo, matching the fetch path's dedupe exactly, so these
    # numbers are what a fetch would really select rather than an upper bound.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pool AS
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY source_record_id
                      ORDER BY group_key, position_in_observation) AS dup
            FROM census)
        WHERE dup = 1""")

    have = con.execute(
        "SELECT count(DISTINCT source_record_id) FROM provenance").fetchone()[0]
    total = con.execute("SELECT count(*) FROM pool").fetchone()[0]
    print(f"census        : {total:,} deduplicated photos, "
          f"{con.execute('SELECT count(DISTINCT canonical_name) FROM pool').fetchone()[0]:,} species")
    print(f"already stored: {have:,} photos\n")

    print(f"{'policy':>14} {'cap':>8} {'eligible':>11} {'species':>9} "
          f"{'new':>11} {'new species':>12} {'est. GB @118KB':>15}")
    print("-" * 85)
    for name in ("production", "production_sa", "research_nc"):
        policy = POLICIES[name]
        lic = policy.sql_in_list()
        for cap in [int(c) for c in args.caps.split(",")]:
            row = con.execute(f"""
                WITH sel AS (
                    SELECT *, row_number() OVER (PARTITION BY canonical_name
                              ORDER BY position_in_observation, hash(group_key),
                                       hash(observer_key), hash(source_record_id)) rn,
                           count(DISTINCT group_key) OVER (PARTITION BY canonical_name) obs
                    FROM pool WHERE license IN {lic})
                SELECT count(*), count(DISTINCT canonical_name),
                       count(*) FILTER (WHERE NOT EXISTS (
                           SELECT 1 FROM provenance p
                           WHERE p.source_record_id = sel.source_record_id))
                FROM sel WHERE rn <= {cap} AND obs >= {args.min_observations}
            """).fetchone()
            new_species = con.execute(f"""
                WITH sel AS (
                    SELECT *, count(DISTINCT group_key) OVER (PARTITION BY canonical_name) obs
                    FROM pool WHERE license IN {lic})
                SELECT count(DISTINCT canonical_name) FROM sel
                WHERE obs >= {args.min_observations}
                  AND canonical_name NOT IN (
                      SELECT DISTINCT accepted_scientific_name FROM provenance
                      WHERE accepted_scientific_name IS NOT NULL)
            """).fetchone()[0]
            print(f"{name:>14} {cap:>8,} {row[0]:>11,} {row[1]:>9,} "
                  f"{row[2]:>11,} {new_species:>12,} {row[2] * 118 / 1e6:>15.1f}")

    print("\nlicence mix of the whole census:")
    for lic, n in con.execute(
            "SELECT license, count(*) FROM pool GROUP BY 1 ORDER BY 2 DESC").fetchall():
        print(f"  {str(lic):22} {n:>10,}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
