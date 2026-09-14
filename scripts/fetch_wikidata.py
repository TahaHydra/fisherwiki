"""Fetch Wikidata common names / IUCN status / cross-ids for our class list.

    python scripts/fetch_wikidata.py [--min-images 40] [--min-observations 25]

Writes ``<work>/wikidata_taxa.json``. Safe to re-run; it overwrites.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import duckdb  # noqa: E402

from fwdata.config import PATHS  # noqa: E402
from fwdata.licenses import get_policy  # noqa: E402
from fwdata.sources import wikidata as wd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="production")
    ap.add_argument("--min-images", type=int, default=40)
    ap.add_argument("--min-observations", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=150)
    args = ap.parse_args()

    policy = get_policy(args.policy)
    parquet = PATHS.work / "candidates_inaturalist.parquet"
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    names = [
        r[0]
        for r in con.execute(
            f"""
            SELECT canonical_name
            FROM read_parquet('{parquet.as_posix()}')
            WHERE license IN {policy.sql_in_list()} AND rank = 'species'
            GROUP BY canonical_name
            HAVING count(*) >= {args.min_images}
               AND count(DISTINCT group_key) >= {args.min_observations}
            ORDER BY canonical_name
            """
        ).fetchall()
    ]
    con.close()

    def log(m: str) -> None:
        print(m, flush=True)

    log(f"{len(names):,} species in the class list")
    result = wd.fetch(names, batch_size=args.batch_size, log=log)

    out = PATHS.work / "wikidata_taxa.json"
    out.write_text(
        json.dumps({k: v.as_dict() for k, v in result.items()}, indent=1,
                   ensure_ascii=False),
        encoding="utf-8",
    )

    with_names = sum(1 for t in result.values() if t.common_names)
    with_en = sum(
        1 for t in result.values()
        if any(l.split("-")[0] == "en" for l, _ in t.common_names)
    )
    with_fr = sum(
        1 for t in result.values()
        if any(l.split("-")[0] == "fr" for l, _ in t.common_names)
    )
    with_iucn = sum(1 for t in result.values() if t.iucn_status)
    langs: dict[str, int] = {}
    for t in result.values():
        for lang, _ in t.common_names:
            base = lang.split("-")[0]
            langs[base] = langs.get(base, 0) + 1

    log("")
    log(f"matched on Wikidata : {len(result):,} / {len(names):,} "
        f"({100*len(result)/max(1,len(names)):.1f}%)")
    log(f"with any common name: {with_names:,}")
    log(f"  English           : {with_en:,}")
    log(f"  French            : {with_fr:,}")
    log(f"with IUCN status    : {with_iucn:,}")
    log(f"languages covered   : {len(langs)}")
    log("top languages       : " + ", ".join(
        f"{k}={v}" for k, v in sorted(langs.items(), key=lambda kv: -kv[1])[:12]
    ))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
