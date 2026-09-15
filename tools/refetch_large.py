#!/usr/bin/env python
"""Re-fetch the corpus at iNaturalist's 1024 px size instead of 500 px.

    python tools/refetch_large.py --plan            # counts and bytes, no network
    python tools/refetch_large.py                   # download, resumable

Why
---
Every one of the 483,271 images in the CAS was fetched from
``.../photos/<id>/medium.jpg``, which iNaturalist caps at **500 px** on the long
edge. Measured against the detector's own boxes, that leaves a median fish crop
of **334 px** and puts **73.3%** of crops below the 448 px training resolution -
so most training samples are upscaled before the model ever sees them, and no
amount of extra images fixes that.

``large`` is the same photograph, same photographer, same licence, same open
S3 bucket, at 1024 px. Roughly 4x the bytes: ~230 GB for the full corpus.

How this stays safe
-------------------
The re-fetched photo is a *different file*, so it gets a different sha256 and a
new provenance row. It keeps the same ``source_record_id`` (the iNaturalist
photo id) and the same ``group_key``, which is what matters: V2 splits are
assigned per observation group, so a large version joins the group its medium
version is already in and inherits the same split. Nothing about split
immutability is disturbed and no image can change sides.

It is filed under ``source_dataset='inaturalist-large'`` so that candidate ids
do not collide with the existing rows, both resolutions are distinguishable in
provenance, and preparation can prefer the larger one (see
``prepare_shards_v2.py --prefer-source``).

Resumability is inherited from the normal fetch path: images already in the CAS
are skipped, permanent failures are recorded and never retried, and the run can
be interrupted and restarted with the same command.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from fwdata.config import PATHS  # noqa: E402
from fwdata.licenses import get_policy  # noqa: E402

SOURCE = "inaturalist-large"
CENSUS = "candidates_inaturalist.parquet"
DERIVED = "candidates_inaturalist_large.parquet"

#: iNaturalist serves a fixed set of sizes under the photo id. ``original`` is
#: also available but is unbounded - single photos run past 10 MB, which is a
#: terabyte across this corpus for detail no 448 px crop will ever use.
SIZES = ("medium", "large", "original")


def build_census(size: str, *, only_stored: bool, log=print) -> Path:
    """Derive the re-fetch census from the one we already have.

    Rewriting the URL beats re-running ``dataset.py extract``: that is an 8
    minute scan of a 20 GB photos dump to arrive at the same rows with one
    token changed.
    """
    import duckdb

    src = PATHS.work / CENSUS
    if not src.exists():
        raise SystemExit(f"{src} missing - run `tools/dataset.py extract` first")
    out = PATHS.work / DERIVED

    con = duckdb.connect(str(PATHS.provenance_db), read_only=True)
    con.execute("PRAGMA threads=8")
    # Only photos we actually hold, by default: re-fetching at 1024 px is about
    # improving the corpus we have, not about growing it. Growing it is a
    # licence-policy decision, which is a different argument.
    keep = ("WHERE source_record_id IN (SELECT DISTINCT source_record_id "
            "FROM provenance WHERE source_dataset = 'inaturalist')"
            if only_stored else "")
    con.execute(f"""
        COPY (
            SELECT * REPLACE (
                regexp_replace(image_url, '/(medium|small|square|thumb)\\.',
                               '/{size}.') AS image_url,
                '{SOURCE}' AS source_dataset,
                NULL AS declared_width,
                NULL AS declared_height
            )
            FROM read_parquet('{src.as_posix()}')
            {keep}
        ) TO '{out.as_posix()}' (FORMAT PARQUET)""")
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    changed = con.execute(
        f"SELECT count(*) FROM read_parquet('{out.as_posix()}') "
        f"WHERE image_url LIKE '%/{size}.%'").fetchone()[0]
    con.close()
    log(f"wrote {out} - {n:,} rows, {changed:,} pointing at /{size}.")
    if changed < n:
        log(f"  {n - changed:,} rows kept their original URL (not a sized "
            f"iNaturalist path); they will be fetched unchanged")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="large", choices=SIZES)
    ap.add_argument("--policy", default="production")
    ap.add_argument("--per-species-cap", type=int, default=100000)
    ap.add_argument("--min-observations", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all-candidates", action="store_true",
                    help="re-fetch every admissible census photo, not only the "
                         "ones already in the CAS")
    ap.add_argument("--plan", action="store_true",
                    help="report what would be fetched and stop")
    args = ap.parse_args(argv)

    from fwdata.fetch_images import fetch, select_candidates
    from fwml.progress import Progress

    policy = get_policy(args.policy)
    census = build_census(args.size, only_stored=not args.all_candidates)
    rows = select_candidates(
        parquet=census, policy=policy,
        per_species_cap=args.per_species_cap,
        min_observations=args.min_observations, limit=args.limit)

    # 118 KB is the measured mean at 500 px; 1024 px is close to four times the
    # pixels, and JPEG cost scales with pixels rather than with the long edge.
    est_gb = len(rows) * 118 * 4 / 1e6
    print(f"\n{len(rows):,} photographs to re-fetch at {args.size}")
    print(f"estimated {est_gb:.0f} GB  (~{est_gb * 1000 / 118 / 4:.0f} files)")
    free_gb = __import__("shutil").disk_usage(str(PATHS.root)).free / 1e9
    print(f"{PATHS.root.drive} has {free_gb:.0f} GB free; "
          f"{'OK' if free_gb - est_gb > 50 else 'NOT ENOUGH with a 50 GB reserve'}")
    if args.plan:
        print("\n--plan: nothing downloaded")
        return 0
    if free_gb - est_gb < 50:
        raise SystemExit("refusing to start: would leave under 50 GB free")

    with Progress(f"refetch_{args.size}", total=len(rows), resumable=True,
                  extra={"size": args.size, "policy": policy.name}) as prog:
        t0 = time.time()
        stats = fetch(rows, policy=policy, workers=args.workers, log=prog.log)
        prog.processed = stats.downloaded + stats.already_present
        prog.failures = stats.http_failures + stats.decode_failures
        prog.note = (f"{stats.downloaded:,} downloaded, "
                     f"{stats.already_present:,} already present, "
                     f"{stats.bytes_fetched / 1e9:.1f} GB in "
                     f"{time.time() - t0:.0f}s")
        prog.log(json.dumps(stats.as_dict(), indent=2))
    (PATHS.work / f"refetch_{args.size}_report.json").write_text(
        json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    print("\nnext: `python tools/dataset.py v2-assign --batch refetch_large` so the "
          "new files join the observation groups they belong to, then prepare "
          "with --prefer-source inaturalist-large")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
