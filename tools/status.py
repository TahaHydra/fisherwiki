#!/usr/bin/env python
"""Where is everything, without asking the process that is doing it.

    python tools/status.py
    python tools/status.py --watch

Reads the durable state every stage leaves behind - status files under
``work/status``, shard sidecars, checkpoint directories, the provenance
database - and prints one screen showing what is running, how far it got, and
whether stopping now would lose anything.

This exists because the last long run's only progress record was the scrollback
of a background shell owned by an assistant session. When the session ended the
run was invisible: still going, no way to see how far, and no way to know
whether killing it was safe. Nothing in this project should ever be in that
position again.

Every read here is cheap and non-locking. It is safe to run while a job is
going, and safe to run when nothing is.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from fwdata.config import PATHS  # noqa: E402
from fwml.progress import all_status, log_dir, status_dir  # noqa: E402

#: Anything whose command line mentions one of these is one of ours.
OUR_SCRIPTS = ("prepare_shards_v2", "detect_fish", "v2_pipeline", "fetch_images",
               "dataset.py", "train_v2", "pilot_backbone", "gbif_census",
               "refetch_large")

SHARD_ROOTS = (Path(r"E:\FisherWiki\shards\v2"), Path(r"D:\fisherwiki-data\v2\shards"))
CHECKPOINT_ROOTS = (Path(r"E:\FisherWiki\runs"), PATHS.root / "runs")
SPLITS = ("train", "validation", "dev_test", "final_test")


def human(n: float, unit: str = "") -> str:
    for suffix, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{suffix}{unit}"
    return f"{n:.0f}{unit}"


def duration(seconds: float | None) -> str:
    if not seconds or seconds < 0:
        return "-"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def running_jobs() -> list[tuple[int, str]]:
    """FisherWiki processes actually alive right now."""
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if proc.info["pid"] == os.getpid():
            continue
        hit = next((s for s in OUR_SCRIPTS if s in cmd), None)
        if hit and "python" in (proc.info.get("name") or "").lower():
            out.append((proc.info["pid"], hit))
    return out


def disks() -> list[tuple[str, float, float]]:
    out = []
    for drive in ("C:", "D:", "E:"):
        try:
            usage = shutil.disk_usage(drive + "\\")
        except OSError:
            continue
        out.append((drive, usage.free / 1e9, usage.total / 1e9))
    return out


def shard_state(root: Path) -> dict | None:
    """Finalised shards and samples, straight off the disk."""
    if not root.exists():
        return None
    from fwml.shards import read_fingerprint, scan_finalised

    per_split, samples, shards, tmp = {}, 0, 0, 0
    for split in SPLITS:
        state = scan_finalised(root / split)
        if not state["shards"] and not state["stale_tmp"]:
            continue
        per_split[split] = (len(state["shards"]), len(state["keys"]))
        shards += len(state["shards"])
        samples += len(state["keys"])
        tmp += len(state["stale_tmp"])
    if not per_split:
        return None
    return {"root": root, "splits": per_split, "shards": shards,
            "samples": samples, "tmp": tmp,
            "fingerprint": read_fingerprint(root)}


def corpus_state() -> dict | None:
    """Counts from the provenance store, read-only and lock-free."""
    try:
        import duckdb
    except ImportError:
        return None
    if not PATHS.provenance_db.exists():
        return None
    try:
        con = duckdb.connect(str(PATHS.provenance_db), read_only=True)
    except Exception:
        # A writer holds the file. Not an error worth shouting about - the
        # running-jobs section above already says who.
        return {"locked": True}
    try:
        def one(sql, default=0):
            try:
                return con.execute(sql).fetchone()[0]
            except Exception:
                return default
        return {
            "images": one("SELECT count(DISTINCT sha256) FROM provenance"),
            "detected": one("SELECT count(DISTINCT sha256) FROM detections"),
            "assigned": one("SELECT count(*) FROM split_group_members "
                            "WHERE dataset_version='v2'"),
            "species": one("SELECT count(DISTINCT species_taxon_id) FROM provenance "
                           "WHERE species_taxon_id IS NOT NULL"),
            "pending_detection": one(
                "SELECT count(*) FROM (SELECT DISTINCT p.sha256 FROM provenance p "
                "WHERE p.cas_path IS NOT NULL AND NOT EXISTS "
                "(SELECT 1 FROM detections d WHERE d.sha256=p.sha256))"),
            "quarantined": one("SELECT count(*) FROM split_quarantine "
                               "WHERE dataset_version='v2'"),
        }
    finally:
        con.close()


def checkpoints() -> list[dict]:
    out = []
    for root in CHECKPOINT_ROOTS:
        if not root.exists():
            continue
        for run in sorted(root.iterdir()):
            if not run.is_dir():
                continue
            found = {n: (run / n) for n in ("latest.pt", "latest.prev.pt", "best.pt")
                     if (run / n).exists()}
            if not found:
                continue
            newest = max(p.stat().st_mtime for p in found.values())
            entry = {"run": run.name, "root": str(root), "have": sorted(found),
                     "age_s": time.time() - newest}
            meta = run / "state.json"
            if meta.exists():
                try:
                    entry["state"] = json.loads(meta.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            out.append(entry)
    return out


def render() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"FisherWiki status   {now}")
    print("=" * 78)

    jobs = running_jobs()
    print("\nRUNNING")
    if jobs:
        for pid, what in jobs:
            print(f"  pid {pid:<8} {what}")
    else:
        print("  nothing - no preparation, detection, fetch or training process")

    print("\nSTAGES (from work/status)")
    entries = all_status()
    if not entries:
        print("  no stage has reported yet")
    for s in sorted(entries, key=lambda e: e.get("updated_at") or "", reverse=True):
        done = s.get("already_done", 0) + s.get("processed", 0)
        total = s.get("total")
        pct = f"{done / total * 100:5.1f}%" if total else "    -"
        stale = ""
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(s["updated_at"])).total_seconds()
            if s.get("state") == "running" and age > 120:
                stale = f"  (no update for {duration(age)} - probably dead)"
        except (KeyError, ValueError):
            pass
        print(f"  {s.get('stage', '?'):<12} {s.get('state', '?'):<12} {pct} "
              f"{done:>9,}/{total or 0:<9,} {s.get('rate_per_s', 0):>7.1f}/s  "
              f"ETA {duration(s.get('eta_s')):<8} "
              f"fail {s.get('failures', 0):<6} "
              f"{'resumable' if s.get('resumable') else 'NOT RESUMABLE'}{stale}")
        if s.get("last_checkpoint"):
            print(f"               last durable point: {s['last_checkpoint']}")
        if s.get("note"):
            print(f"               {s['note']}")

    print("\nCORPUS")
    corpus = corpus_state()
    if corpus is None:
        print("  no provenance database")
    elif corpus.get("locked"):
        print("  database is held by a running job (see above)")
    else:
        print(f"  images in CAS      {corpus['images']:>10,}")
        print(f"  species            {corpus['species']:>10,}")
        print(f"  detected           {corpus['detected']:>10,}   "
              f"pending {corpus['pending_detection']:,}")
        print(f"  assigned to splits {corpus['assigned']:>10,}   "
              f"quarantined {corpus['quarantined']:,}")

    print("\nPREPARED SHARDS")
    any_shards = False
    for root in SHARD_ROOTS:
        state = shard_state(root)
        if not state:
            continue
        any_shards = True
        fp = state["fingerprint"]
        recipe = (f"long_edge={fp['long_edge']} q={fp['jpeg_quality']} "
                  f"draft={fp['decode_draft']} v{fp['prep_impl_version']}"
                  if fp else "NO FINGERPRINT - cannot be resumed into")
        print(f"  {str(root)}")
        print(f"    {state['shards']} shards, {state['samples']:,} samples"
              + (f", {state['tmp']} interrupted .tmp" if state["tmp"] else ""))
        print(f"    recipe: {recipe}")
        for split, (n_shards, n_samples) in state["splits"].items():
            print(f"      {split:<12} {n_shards:>4} shards  {n_samples:>9,} samples")
    if not any_shards:
        print("  none prepared yet")

    print("\nCHECKPOINTS")
    cps = checkpoints()
    if not cps:
        print("  no training run has checkpointed yet")
    for c in cps:
        print(f"  {c['run']:<24} {', '.join(c['have'])}  "
              f"(written {duration(c['age_s'])} ago)")
        st = c.get("state") or {}
        if st:
            print(f"      epoch {st.get('epoch')}  step {st.get('global_step')}  "
                  f"best {st.get('best_metric')}")

    print("\nDISK")
    for drive, free, total in disks():
        reserve = 50 if drive == "D:" else 30
        flag = "  LOW" if free < reserve else ""
        print(f"  {drive} {free:6.1f} GB free of {total:6.1f} GB "
              f"(reserve {reserve} GB){flag}")

    print(f"\nlogs: {log_dir()}    status: {status_dir()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true",
                    help="redraw every --interval seconds until interrupted")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--json", action="store_true",
                    help="machine-readable dump instead of the screen")
    args = ap.parse_args(argv)

    if args.json:
        print(json.dumps({
            "running": running_jobs(), "stages": all_status(),
            "corpus": corpus_state(),
            "shards": [s for s in (shard_state(r) for r in SHARD_ROOTS) if s],
            "checkpoints": checkpoints(),
            "disks": [{"drive": d, "free_gb": round(f, 1), "total_gb": round(t, 1)}
                      for d, f, t in disks()],
        }, indent=2, default=str))
        return 0

    while True:
        render()
        if not args.watch:
            return 0
        time.sleep(args.interval)
        print("\n" * 2)


if __name__ == "__main__":
    raise SystemExit(main())
