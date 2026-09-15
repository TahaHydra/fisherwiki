#!/usr/bin/env python
"""Cold-cache microbenchmark for shard preparation.

    python tools/bench_prepare.py --images 300 --workers 0,2,4,6
    python tools/bench_prepare.py --images 300 --workers 6 --out-drive D

Every configuration reads a **disjoint** slice of the CAS, in ``cas_path``
order. That is the whole point: re-reading the same 300 files four times
measures the page cache, and this project has already once approved a
multi-hour run off a warm 2k-image number that was wrong by 10x.

What it reports, and why each number is here
--------------------------------------------
``img/s``            end to end, including the tar write.
``read``             seconds inside ``Path.read_bytes`` - the D: spindle.
``decode``           JPEG decode, after ``draft`` has had its say.
``crop+resize``      box rescale, crop, LANCZOS.
``encode``           JPEG encode of the derivative.
``shard``            main-thread tar write plus index bookkeeping.
``src MB/s``         bytes read from the CAS per wall-clock second.
``out MB/s``         derivative bytes written per wall-clock second.

Worker stage times are summed **across processes**, so with 6 workers they
add up to roughly six times the wall clock. That is deliberate: the ratios
between them say where the CPU budget goes, and comparing their total against
the wall clock says whether the pool is saturated or waiting for the disk.

Writes go to a scratch directory that is deleted afterwards. Nothing here
touches the real shard set or the provenance database.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from fwdata.config import PATHS  # noqa: E402
from fwdata.splits_v2 import DATASET_VERSION  # noqa: E402

SPLITS = ("train", "validation", "dev_test", "final_test")


def slices(n_configs: int, per_config: int, offset: int) -> list[tuple[int, int]]:
    return [(offset + i * per_config, per_config) for i in range(n_configs)]


def fetch_rows(total: int, offset: int) -> list[tuple]:
    """A contiguous run of the corpus in CAS order, skipping ``offset`` rows."""
    import duckdb

    con = duckdb.connect(str(PATHS.provenance_db), read_only=True)
    try:
        return con.execute(
            f"""
            SELECT m.sha256, g.split, g.taxon_id, g.group_id,
                   any_value(p.cas_path) AS cas_path,
                   any_value(p.accepted_scientific_name) AS name,
                   any_value(d.x0), any_value(d.y0), any_value(d.x1),
                   any_value(d.y1)
            FROM split_group_members m
            JOIN split_groups g USING (dataset_version, group_id)
            JOIN provenance p ON p.sha256 = m.sha256
            LEFT JOIN detections d ON d.sha256 = m.sha256 AND d.is_primary
            WHERE m.dataset_version = ?
              AND g.split IN ({",".join("?" * len(SPLITS))})
            GROUP BY m.sha256, g.split, g.taxon_id, g.group_id
            ORDER BY any_value(p.cas_path)
            LIMIT ? OFFSET ?
            """,
            [DATASET_VERSION, *SPLITS, total, offset],
        ).fetchall()
    finally:
        con.close()


def run_config(rows, *, workers: int, chunk: int, out_root: Path, timeout: float,
               decode_draft: bool, jpeg_optimize: bool, shard_size: int,
               long_edge: int, quality: int, label: str) -> dict:
    import multiprocessing as mp

    import psutil

    import prepare_shards_v2 as prep
    from fwml.shards import Sample, ShardWriter

    cfg = {"cas_root": PATHS.cas, "long_edge": long_edge, "jpeg_quality": quality,
           "jpeg_optimize": jpeg_optimize, "decode_draft": decode_draft,
           "crop_pad": 0.25, "min_crop_px": 224, "whole_frame_fraction": 0.15}
    jobs = [(r[0], r[1], int(r[2]), r[3], r[4], r[6], r[7], r[8], r[9])
            for r in rows]

    out_dir = out_root / label
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    proc = psutil.Process()
    src_bytes = sum((PATHS.cas / r[4]).stat().st_size for r in rows if r[4])

    stage = [0.0, 0.0, 0.0, 0.0]
    shard_s = 0.0
    out_bytes = n = fails = 0
    writers: dict[str, ShardWriter] = {}
    pool = None
    cpu0 = psutil.cpu_times()
    proc.cpu_percent(None)
    t0 = time.time()
    try:
        if workers > 0:
            pool = mp.Pool(workers, initializer=prep._init_worker, initargs=(cfg,))
            results = pool.imap(prep.prepare_one, jobs, chunksize=chunk)
        else:
            prep._init_worker(cfg)
            results = map(prep.prepare_one, jobs)

        import hashlib
        for sha, split, taxon, group_id, _variant, data, t in results:
            for k in range(4):
                stage[k] += t[k]
            if split is None:
                fails += 1
                continue
            ts = time.time()
            w = writers.get(split)
            if w is None:
                w = ShardWriter(out_dir / split, prefix=split,
                                shard_size=shard_size)
                writers[split] = w
            w.write(Sample(key=sha, data=data, class_id=0, taxon_id=int(taxon),
                           sha256=hashlib.sha256(data).hexdigest(), split=split,
                           group_id=group_id))
            shard_s += time.time() - ts
            out_bytes += len(data)
            n += 1
            if time.time() - t0 > timeout:
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        ts = time.time()
        for w in writers.values():
            w.close()
        shard_s += time.time() - ts

    wall = time.time() - t0
    cpu1 = psutil.cpu_times()
    busy = ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system))
    rss = proc.memory_info().rss / 1e6
    shutil.rmtree(out_dir, ignore_errors=True)

    done_frac = n / max(len(jobs), 1)
    return {
        "label": label, "workers": workers, "images": n,
        "seconds": round(wall, 1), "img_s": round(n / max(wall, 1e-9), 1),
        "read_s": round(stage[0], 1), "decode_s": round(stage[1], 1),
        "resize_s": round(stage[2], 1), "encode_s": round(stage[3], 1),
        "shard_s": round(shard_s, 2),
        "src_mb_s": round(src_bytes * done_frac / 1e6 / max(wall, 1e-9), 1),
        "out_mb_s": round(out_bytes / 1e6 / max(wall, 1e-9), 1),
        "mean_out_kb": round(out_bytes / max(n, 1) / 1024, 1),
        "cpu_cores_busy": round(busy / max(wall, 1e-9), 1),
        "rss_mb": round(rss), "failed": fails,
    }


HEAD = (f"{'configuration':>22} {'w':>3} {'img':>5} {'sec':>6} {'img/s':>7} "
        f"{'srcMB/s':>8} {'outMB/s':>8} {'read':>7} {'decode':>7} {'resize':>7} "
        f"{'encode':>7} {'shard':>6} {'cores':>6} {'RSS':>6}")


def show(r: dict) -> None:
    print(f"{r['label']:>22} {r['workers']:>3} {r['images']:>5} "
          f"{r['seconds']:>6} {r['img_s']:>7} {r['src_mb_s']:>8} "
          f"{r['out_mb_s']:>8} {r['read_s']:>7} {r['decode_s']:>7} "
          f"{r['resize_s']:>7} {r['encode_s']:>7} {r['shard_s']:>6} "
          f"{r['cpu_cores_busy']:>6} {r['rss_mb']:>6}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=300,
                    help="images per configuration - each gets its own slice")
    ap.add_argument("--workers", default="0,2,4,6")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=35.0,
                    help="hard cap per configuration")
    ap.add_argument("--offset", type=int, default=0,
                    help="start of the first slice; move it to stay off pages "
                         "an earlier benchmark warmed")
    ap.add_argument("--out-drive", default="E",
                    help="where derivatives are written during the benchmark")
    ap.add_argument("--long-edge", type=int, default=512)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--skip-old", action="store_true")
    args = ap.parse_args(argv)

    worker_list = [int(x) for x in args.workers.split(",") if x.strip() != ""]
    configs: list[tuple[str, dict]] = []
    if not args.skip_old:
        # The v1 recipe: serial, full-resolution decode, optimized Huffman
        # tables. Same images, same output size - only the implementation
        # differs, which is what makes the comparison mean anything.
        configs.append(("old serial", dict(workers=0, decode_draft=False,
                                           jpeg_optimize=True)))
    for w in worker_list:
        configs.append((f"new w={w}", dict(workers=w, decode_draft=True,
                                           jpeg_optimize=False)))

    out_root = Path(f"{args.out_drive}:/fw-benchmark/prepare")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"{len(configs)} configurations x {args.images} images, disjoint CAS "
          f"slices from offset {args.offset}, {args.timeout:.0f}s cap each")
    print(f"reading D:{PATHS.cas}, writing {out_root}\n")
    print(HEAD)
    print("-" * len(HEAD))

    # One query, sliced in Python. The ORDER BY runs over the whole corpus, so
    # issuing it once per configuration would put a minute of DuckDB inside a
    # three-minute budget.
    t = time.time()
    all_rows = fetch_rows(args.images * len(configs), args.offset)
    print(f"  ({len(all_rows):,} rows from duckdb in {time.time() - t:.1f}s)\n")

    results = []
    for i, (label, kw) in enumerate(configs):
        rows = all_rows[i * args.images:(i + 1) * args.images]
        if not rows:
            print(f"  {label}: no rows at that offset")
            continue
        r = run_config(rows, chunk=args.chunk, out_root=out_root,
                       timeout=args.timeout, shard_size=args.shard_size,
                       long_edge=args.long_edge, quality=args.quality,
                       label=label, **kw)
        show(r)
        results.append(r)

    shutil.rmtree(out_root, ignore_errors=True)
    if not results:
        return 1
    old = next((r for r in results if r["label"].startswith("old")), None)
    best = max(results, key=lambda r: r["img_s"])
    print()
    if old:
        print(f"old preparation : {old['img_s']} img/s")
    print(f"new preparation : {best['img_s']} img/s "
          f"({best['workers']} workers)")
    if old and old["img_s"] > 0:
        print(f"speedup         : {best['img_s'] / old['img_s']:.1f}x")
    print(f"mean derivative : {best['mean_out_kb']} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
