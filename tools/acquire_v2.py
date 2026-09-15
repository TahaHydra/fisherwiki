#!/usr/bin/env python
"""Acquire V2 fish training data, then stop before detection/cropping/training.

Default full run::

    python -u tools/acquire_v2.py --all

Safe to Ctrl+C and re-run. Discovery checkpoints are persisted per source,
candidate insertion is idempotent, image downloads use resumable ``.part``
files, and V2 split assignment only inserts previously unseen groups. This
command NEVER launches fish detection, crop preparation, shard generation or
training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

import duckdb  # noqa: E402

from fwdata.acquisition import (  # noqa: E402
    DiskReserveReached,
    ScanState,
    fetch_registered_source,
    pending_count,
    prioritized_species,
    register_candidates,
)
from fwdata.config import PATHS, free_space_gb  # noqa: E402
from fwdata.fetch_images import _row_to_provenance  # noqa: E402
from fwdata.provenance import ProvenanceDB  # noqa: E402
from fwdata.sources import commons_media, fathomnet_media, gbif_media  # noqa: E402
from fwdata.splits_v2 import V2SplitStore  # noqa: E402
from fwml.progress import Progress  # noqa: E402

SOURCE_ORDER = ("inaturalist", "gbif-media", "wikimedia-commons", "fathomnet")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def register_inaturalist(
    db: ProvenanceDB,
    *,
    cap: int,
    min_observations: int = 3,
    chunk: int = 50_000,
) -> int:
    """Register iNaturalist candidates without applying a licence allow-list."""
    parquet = PATHS.work / "candidates_inaturalist.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} missing; run `python tools/dataset.py extract` first"
        )
    # Same diversity ordering as the original fetcher, but deliberately no
    # `WHERE license IN (...)`: acquisition records everything and release policy
    # is a later, separate decision.
    sql = f"""
    WITH deduped AS (
      SELECT * FROM read_parquet('{parquet.as_posix()}')
      QUALIFY row_number() OVER (
        PARTITION BY source_record_id ORDER BY group_key, position_in_observation
      ) = 1
    ), pool AS (
      SELECT *,
        row_number() OVER (
          PARTITION BY canonical_name
          ORDER BY position_in_observation ASC, hash(group_key),
                   hash(observer_key), hash(source_record_id)
        ) rn,
        count(DISTINCT group_key) OVER (PARTITION BY canonical_name) sp_obs
      FROM deduped
    )
    SELECT source_dataset, source_record_id, image_url, source_url, source_taxon_id,
           original_scientific_name, accepted_scientific_name, license, license_raw,
           license_url, creator, copyright_holder, group_key, observer_key,
           latitude, longitude, positional_accuracy, observed_on, quality_grade,
           position_in_observation, declared_width, declared_height, ext,
           canonical_name, anomaly_score
    FROM pool
    WHERE rn <= {int(cap)} AND sp_obs >= {int(min_observations)}
    ORDER BY canonical_name, rn
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    cur = con.execute(sql)
    total = 0
    try:
        while True:
            rows = cur.fetchmany(chunk)
            if not rows:
                break
            provs = [_row_to_provenance(r) for r in rows]
            total += register_candidates(db, provs)
            db.flush()
            log(f"  iNaturalist candidates registered: {total:,}")
    finally:
        con.close()
    return total


def discover_source(
    db: ProvenanceDB,
    source: str,
    *,
    gbif_cap: int,
    commons_cap: int,
    fathomnet_cap: int,
    taxa_limit: int | None,
) -> int:
    if source == "gbif-media":
        taxa = prioritized_species(db, id_attribute="gbif_taxon_id")
        cap = gbif_cap
        fn = gbif_media.discover_taxon
    elif source == "wikimedia-commons":
        taxa = prioritized_species(db)
        cap = commons_cap
        fn = commons_media.discover_taxon
    elif source == "fathomnet":
        taxa = prioritized_species(db)
        cap = fathomnet_cap
        fn = fathomnet_media.discover_taxon
    else:
        return 0
    if taxa_limit:
        taxa = taxa[: int(taxa_limit)]
    state = ScanState(source)
    added = 0
    with Progress(f"discover-{source}", total=len(taxa), resumable=True) as p:
        for i, taxon in enumerate(taxa, 1):
            try:
                added += fn(db, taxon, cap=cap, state=state, log=log)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # A provider/taxon failure is not allowed to kill a multi-hour
                # census. It is NOT checkpointed by the adapter, so next run
                # retries it.
                p.log(f"{taxon.canonical_name}: {type(exc).__name__}: {exc}")
                p.advance(1, failures=1)
                continue
            p.advance(1)
            if i % 250 == 0:
                db.flush()
                p.checkpointed(f"{i}/{len(taxa)} taxa; +{added} candidates")
        db.flush()
        p.finish(note=f"registered {added:,} new candidates")
    return added


def _reopen_old_policy_refusals(db: ProvenanceDB, selected: tuple[str, ...]) -> int:
    """Undo only failures created by the old acquisition-time licence gate.

    Old `fetch_images.py` stored a policy rejection as a *permanent network
    failure*. Broad acquisition deliberately no longer has that gate, so leaving
    those rows would make hundreds of thousands of newly selected candidates
    look permanently dead. HTTP/decode failures remain untouched.
    """
    placeholders = ",".join("?" * len(selected))
    params = [*selected]
    before = db.con.execute(
        f"SELECT count(*) FROM fetch_failures f JOIN candidates c USING(candidate_id) "
        f"WHERE c.source_dataset IN ({placeholders}) AND f.reason='permanent' "
        "AND lower(f.detail) LIKE 'licence refused:%'",
        params,
    ).fetchone()[0]
    if before:
        db.con.execute(
            f"DELETE FROM fetch_failures WHERE candidate_id IN ("
            f" SELECT c.candidate_id FROM candidates c WHERE c.source_dataset IN ({placeholders})"
            ") AND reason='permanent' AND lower(detail) LIKE 'licence refused:%'",
            params,
        )
        log(f"reopened {before:,} candidates previously blocked only by licence policy")
    return int(before)


def fetch_source(db: ProvenanceDB, source: str, args) -> dict:
    pending = pending_count(db, source)
    if pending == 0:
        return {"source": source, "selected": 0, "downloaded": 0, "already_done": 0}
    with Progress(f"acquire-{source}", total=pending, resumable=True) as p:
        stats = fetch_registered_source(
            db,
            source,
            workers=args.workers,
            batch_size=args.batch_size,
            reserve_gb=args.reserve_gb,
            progress=p,
            log=log,
        )
        p.finish(note=json.dumps(stats.as_dict(), sort_keys=True))
    return stats.as_dict()


def assign_splits(batch: str) -> dict:
    with V2SplitStore() as store:
        stats = store.assign(batch=batch, near_duplicates=True, coverage_topup=True, log=log)
        checks = store.verify()
        fp = store.fingerprint()
    if any(checks.values()):
        raise RuntimeError(f"V2 split verification failed: {checks}")
    return {"assign": stats.as_dict(), "verify": checks, "fingerprint": fp}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run all configured sources")
    ap.add_argument("--sources", default=None,
                    help="comma-separated subset: inaturalist,gbif-media,wikimedia-commons,fathomnet")
    ap.add_argument("--inat-cap", type=int, default=1000,
                    help="max iNaturalist photos/species (raise and re-run later safely)")
    ap.add_argument("--gbif-cap", type=int, default=300)
    ap.add_argument("--commons-cap", type=int, default=40)
    ap.add_argument("--fathomnet-cap", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--reserve-gb", type=float, default=50.0)
    ap.add_argument("--taxa-limit", type=int, default=None,
                    help="debug/short run only: source discovery taxa limit")
    ap.add_argument("--discover-only", action="store_true")
    ap.add_argument("--no-split", action="store_true")
    args = ap.parse_args(argv)

    selected = SOURCE_ORDER if args.all or not args.sources else tuple(
        x.strip() for x in args.sources.split(",") if x.strip()
    )
    bad = set(selected) - set(SOURCE_ORDER)
    if bad:
        raise SystemExit(f"unknown source(s): {sorted(bad)}")

    PATHS.ensure()
    if free_space_gb(PATHS.root) < args.reserve_gb:
        raise SystemExit(
            f"refusing acquisition: only {free_space_gb(PATHS.root):.1f} GB free; "
            f"reserve is {args.reserve_gb:.1f} GB"
        )

    report = {
        "started_at": time.time(),
        "sources": list(selected),
        "registered": {},
        "fetch": {},
        "split": None,
    }
    try:
        with ProvenanceDB() as db:
            report["reopened_policy_refusals"] = _reopen_old_policy_refusals(db, selected)
            if "inaturalist" in selected:
                report["registered"]["inaturalist"] = register_inaturalist(
                    db, cap=args.inat_cap
                )
            for source in selected:
                if source == "inaturalist":
                    continue
                report["registered"][source] = discover_source(
                    db, source,
                    gbif_cap=args.gbif_cap,
                    commons_cap=args.commons_cap,
                    fathomnet_cap=args.fathomnet_cap,
                    taxa_limit=args.taxa_limit,
                )

            if not args.discover_only:
                for source in selected:
                    report["fetch"][source] = fetch_source(db, source, args)
    except DiskReserveReached as exc:
        log(f"STOPPED CLEANLY: {exc}")
        report["stopped"] = str(exc)
        (PATHS.work / "acquisition_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return 3

    if not args.discover_only and not args.no_split:
        batch = "acquire_" + time.strftime("%Y%m%d_%H%M%S")
        report["split"] = assign_splits(batch)

    report["finished_at"] = time.time()
    report["free_gb"] = free_space_gb(PATHS.root)
    out = PATHS.work / "acquisition_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\nwrote {out}")
    log("STOP BOUNDARY reached: no detection, crop, shard or training stage was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
