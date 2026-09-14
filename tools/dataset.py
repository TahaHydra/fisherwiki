#!/usr/bin/env python
"""FisherWiki dataset command line.

    python tools/dataset.py discover
    python tools/dataset.py download --source inaturalist
    python tools/dataset.py download --source gbif
    python tools/dataset.py extract
    python tools/dataset.py coverage --policy production --min-images 40
    python tools/dataset.py plan --policy production
    python tools/dataset.py fetch --policy production --per-species-cap 300
    python tools/dataset.py verify
    python tools/dataset.py dedupe
    python tools/dataset.py attributions --corpus europe_freshwater_v1

Every subcommand is safe to interrupt and re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fwdata import net  # noqa: E402
from fwdata.config import PATHS, free_space_gb  # noqa: E402
from fwdata.licenses import POLICIES, get_policy  # noqa: E402
from fwdata.sources import gbif  # noqa: E402
from fwdata.sources import inaturalist as inat  # noqa: E402


def _log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------
def cmd_discover(args: argparse.Namespace) -> int:
    _log(f"data root : {PATHS.root}")
    _log(f"free space: {free_space_gb(PATHS.root):.1f} GB")
    _log("")
    _log("iNaturalist Open Data  (s3://inaturalist-open-data, licence per photo)")
    total = 0
    for f in inat.discover():
        have = f.local.exists()
        size = f.size or 0
        total += size
        mark = "present" if have else "missing"
        _log(f"  {f.name:32} {size/1e9:7.2f} GB  {mark}")
    _log(f"  {'total':32} {total/1e9:7.2f} GB")
    _log("")
    _log(f"GBIF Backbone Taxonomy  (CC BY 4.0, version {gbif.BACKBONE_VERSION})")
    bf = gbif.backbone_file()
    sz = net.content_length(bf.url) or 0
    _log(f"  {bf.local.name:32} {sz/1e9:7.2f} GB  "
         f"{'present' if bf.local.exists() else 'missing'}")
    _log("")
    _log("Licence policies:")
    for name, pol in POLICIES.items():
        flag = "commercial-safe" if pol.commercial_safe else "RESEARCH ONLY"
        _log(f"  {name:15} [{flag}] {sorted(x.value for x in pol.allowed)}")
    return 0


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def cmd_download(args: argparse.Namespace) -> int:
    PATHS.ensure()
    if args.source == "inaturalist":
        keys = args.files or ["taxa", "observers", "observations", "photos"]
        files = {f.key: f for f in inat.discover()}
        total = sum(files[k].size or 0 for k in keys)
        _log(f"downloading {keys} ({total/1e9:.2f} GB) with {args.workers} workers")
        prog = net.Progress(total_bytes=total, total_items=len(keys))
        digests = inat.download_bulk(keys, workers=args.workers, progress=prog)
        for k, d in digests.items():
            _log(f"  {files[k].name}: sha256={d}")
    elif args.source == "gbif":
        prog = net.Progress(total_items=1)
        d = gbif.download_backbone(progress=prog, workers=args.workers)
        _log(f"  {gbif.backbone_file().local.name}: sha256={d}")
    else:
        _log(f"unknown source {args.source!r}")
        return 2
    _log("done: " + prog.line())
    return 0


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def cmd_extract(args: argparse.Namespace) -> int:
    from fwdata.corpus import extract_candidates

    stats = extract_candidates(
        threads=args.threads,
        memory_limit_gb=args.memory_gb,
        require_coordinates=args.require_coordinates,
        log=_log,
    )
    _log(json.dumps(stats.as_dict(), indent=2))
    return 0


# ---------------------------------------------------------------------------
# coverage / plan
# ---------------------------------------------------------------------------
def cmd_coverage(args: argparse.Namespace) -> int:
    from fwdata.corpus import species_coverage

    pol = get_policy(args.policy)
    rows = species_coverage(policy=pol, min_images=args.min_images)
    _log(f"{len(rows):,} taxa with >= {args.min_images} images under {pol.name}")
    _log("")
    _log(f"{'species':38} {'family':22} {'imgs':>7} {'obs':>7} {'phot':>6}")
    for name, _tid, fam, imgs, obs, phot in rows[: args.top]:
        _log(f"{(name or '')[:37]:38} {(fam or '')[:21]:22} {imgs:7,} {obs:7,} {phot:6,}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                [
                    {
                        "species": r[0], "inat_taxon_id": r[1], "family": r[2],
                        "images": r[3], "observations": r[4], "observers": r[5],
                    }
                    for r in rows
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        _log(f"\nwrote {args.out}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from fwdata.planning import plan_packs

    pol = get_policy(args.policy)
    report = plan_packs(policy=pol, log=_log)
    out = Path(args.out or (PATHS.work / "pack_plan.json"))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"\nwrote {out}")
    return 0



# ---------------------------------------------------------------------------
# build-corpus
# ---------------------------------------------------------------------------
def cmd_build_corpus(args: argparse.Namespace) -> int:
    from fwdata.splits import CorpusBuilder, SplitStrategy

    builder = CorpusBuilder()
    try:
        stats = builder.build(
            args.corpus,
            strategy=args.strategy,
            min_images_per_class=args.min_images,
            min_observations_per_class=args.min_observations,
            geo_holdout_cells=args.geo_holdout_cells,
            log=_log,
        )
        if stats.leakage_groups or stats.leakage_hashes:
            _log("")
            _log("REFUSING to export: the split leaks between train and test.")
            return 1
        out = builder.export_manifest(args.corpus)
        _log("")
        _log(f"wrote {out}")
        _log(f"wrote {out.parent / 'labels.json'}")
    finally:
        builder.close()
    return 0


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
def cmd_fetch(args: argparse.Namespace) -> int:
    from fwdata.fetch_images import main as fetch_main

    argv = [
        "--policy", args.policy,
        "--per-species-cap", str(args.per_species_cap),
        "--min-observations", str(args.min_observations),
        "--workers", str(args.workers),
    ]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.where:
        argv += ["--where", args.where]
    return fetch_main(argv)


# ---------------------------------------------------------------------------
# verify / dedupe / attributions
# ---------------------------------------------------------------------------
def cmd_verify(args: argparse.Namespace) -> int:
    from fwdata.provenance import ProvenanceDB
    from fwdata.verify import verify_store

    with ProvenanceDB(read_only=True) as db:
        report = verify_store(db, sample=args.sample, log=_log)
    _log(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


def cmd_dedupe(args: argparse.Namespace) -> int:
    from fwdata.dedupe import find_duplicates

    report = find_duplicates(threshold=args.threshold, log=_log)
    _log(json.dumps(report, indent=2))
    return 0


def cmd_attributions(args: argparse.Namespace) -> int:
    from fwdata.provenance import ProvenanceDB, export_attributions

    out = Path(args.out or (PATHS.artifacts / "attributions"))
    with ProvenanceDB(read_only=True) as db:
        summary = export_attributions(db, out, corpus=args.corpus)
    _log(json.dumps(summary, indent=2))
    _log(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# V2 splits
# ---------------------------------------------------------------------------
def cmd_v2_assign(args: argparse.Namespace) -> int:
    from fwdata.splits_v2 import V2SplitStore

    with V2SplitStore() as store:
        stats = store.assign(
            batch=args.batch,
            near_duplicates=not args.no_near_duplicates,
            coverage_topup=not args.no_coverage_topup,
            limit=args.limit,
            log=_log,
        )
        checks = store.verify()
        _log("")
        _log(f"  verification: {checks}")
        _log(f"  fingerprint : {store.fingerprint()[:16]}")
    if any(checks.values()):
        _log("")
        _log("REFUSING: the split state failed its structural checks.")
        return 1
    out = PATHS.work / f"v2_assign_{args.batch}.json"
    out.write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    _log(f"wrote {out}")
    return 0


def cmd_v2_verify(args: argparse.Namespace) -> int:
    from fwdata.splits_v2 import V2SplitStore

    with V2SplitStore(read_only=True) as store:
        checks = store.verify()
        _log(json.dumps(checks, indent=2))
        _log(f"fingerprint: {store.fingerprint()}")
    return 1 if any(checks.values()) else 0


def cmd_v2_export(args: argparse.Namespace) -> int:
    from fwdata.splits_v2 import V2SplitStore

    with V2SplitStore() as store:
        checks = store.verify()
        if any(checks.values()):
            _log(f"REFUSING to export: {checks}")
            return 1
        result = store.export_manifests(
            args.corpus,
            min_images_per_class=args.min_images,
            min_observations_per_class=args.min_observations,
            log=_log,
        )
    _log(json.dumps(result, indent=2))
    return 0


def cmd_v2_release_eval(args: argparse.Namespace) -> int:
    """The only door to final_test, and it keeps a record of everyone through it."""
    from fwdata.splits_v2 import V2SplitStore, release_eval_authorised

    with V2SplitStore() as store:
        try:
            path = store.open_final_test(
                args.corpus,
                reason=args.reason or "",
                authorised=release_eval_authorised(args.i_am_releasing),
            )
        except (PermissionError, ValueError, FileNotFoundError) as exc:
            _log(str(exc))
            return 1
        _log(f"final_test unsealed: {path}")
        _log("")
        _log("accesses recorded for this dataset version:")
        for when, reason, commit, rows in store.access_log():
            _log(f"  {when}  {commit[:12]}  {rows:>7,} rows  {reason}")
    return 0


def cmd_v2_status(args: argparse.Namespace) -> int:
    from fwdata.splits_v2 import DATASET_VERSION, V2SplitStore

    with V2SplitStore(read_only=True) as store:
        rows = store.con.execute(
            "SELECT split, count(*) FROM split_groups WHERE dataset_version = ? "
            "GROUP BY split ORDER BY split", [DATASET_VERSION],
        ).fetchall()
        images = store.con.execute(
            "SELECT g.split, count(*) FROM split_group_members m "
            "JOIN split_groups g USING (dataset_version, group_id) "
            "WHERE m.dataset_version = ? GROUP BY g.split ORDER BY g.split",
            [DATASET_VERSION],
        ).fetchall()
        rules = store.con.execute(
            "SELECT rule, count(*) FROM split_groups WHERE dataset_version = ? "
            "GROUP BY rule ORDER BY 2 DESC", [DATASET_VERSION],
        ).fetchall()
        quarantined = store.con.execute(
            "SELECT count(*) FROM split_quarantine WHERE dataset_version = ?",
            [DATASET_VERSION],
        ).fetchone()[0]
        _log(f"dataset_version {DATASET_VERSION}")
        frozen = store.stored_fractions()
        _log(f"  fractions      : "
             f"{frozen if frozen else '(not frozen yet - set on first assign)'}")
        _log(f"  groups by split: {dict(rows)}")
        _log(f"  images by split: {dict(images)}")
        _log(f"  assignment rule: {dict(rules)}")
        _log(f"  quarantined    : {quarantined:,}")
        _log(f"  fingerprint    : {store.fingerprint()}")
        log = store.access_log()
        _log(f"  final_test reads: {len(log)}")
        for when, reason, commit, n in log:
            _log(f"    {when}  {commit[:12]}  {n:,} rows  {reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("discover", help="show sources, sizes and policies")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("download", help="fetch official bulk exports")
    p.add_argument("--source", required=True, choices=["inaturalist", "gbif"])
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=12)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("extract", help="build the candidate photo table")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory-gb", type=int, default=20)
    p.add_argument("--require-coordinates", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("coverage", help="images per species under a policy")
    p.add_argument("--policy", default="production")
    p.add_argument("--min-images", type=int, default=1)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("plan", help="propose regional pack composition")
    p.add_argument("--policy", default="production")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("build-corpus", help="split into train/val/test without leakage")
    p.add_argument("--corpus", required=True)
    p.add_argument("--strategy", default="observation",
                   choices=["observation", "observer", "observer_then_observation"])
    p.add_argument("--min-images", type=int, default=40)
    p.add_argument("--min-observations", type=int, default=25)
    p.add_argument("--geo-holdout-cells", type=int, default=0)
    p.set_defaults(func=cmd_build_corpus)

    p = sub.add_parser("fetch", help="download selected images into the CAS")
    p.add_argument("--policy", default="production")
    p.add_argument("--per-species-cap", type=int, default=300)
    p.add_argument("--min-observations", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--where", default="")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("verify", help="check CAS integrity against provenance")
    p.add_argument("--sample", type=int, default=2000)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("dedupe", help="find exact and perceptual duplicates")
    p.add_argument("--threshold", type=int, default=6)
    p.set_defaults(func=cmd_dedupe)

    p = sub.add_parser("attributions", help="export ATTRIBUTIONS.csv")
    p.add_argument("--corpus", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_attributions)

    # --- V2 splits ---------------------------------------------------------
    p = sub.add_parser(
        "v2-assign",
        help="assign V2 splits to every unassigned image (safe to re-run)",
    )
    p.add_argument("--batch", required=True,
                   help="name of the acquisition batch, recorded on each new row")
    p.add_argument("--no-near-duplicates", action="store_true",
                   help="skip perceptual near-duplicate grouping (faster, weaker)")
    p.add_argument("--no-coverage-topup", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_v2_assign)

    p = sub.add_parser("v2-verify", help="structural checks on V2 split state")
    p.set_defaults(func=cmd_v2_verify)

    p = sub.add_parser("v2-status", help="V2 split state and final_test access log")
    p.set_defaults(func=cmd_v2_status)

    p = sub.add_parser(
        "v2-export",
        help="write manifest.parquet (train/validation/dev_test) + sealed final_test",
    )
    p.add_argument("--corpus", default="global_v2")
    p.add_argument("--min-images", type=int, default=40,
                   help="minimum *train* images per class (not total, unlike V1: "
                        "40 here admits 1,654 classes, 25 admits 1,963 - see "
                        "docs/DATA_PROVENANCE.md §10)")
    p.add_argument("--min-observations", type=int, default=25)
    p.set_defaults(func=cmd_v2_export)

    p = sub.add_parser(
        "v2-release-eval",
        help="unseal final_test for a release evaluation (recorded, not free)",
    )
    p.add_argument("--corpus", default="global_v2")
    p.add_argument("--reason", required=True)
    p.add_argument("--i-am-releasing", action="store_true",
                   help="required; or set FISHERWIKI_RELEASE_EVAL=1")
    p.set_defaults(func=cmd_v2_release_eval)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
