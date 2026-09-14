#!/usr/bin/env python
"""Microbenchmark the detection pipeline on real pending CAS images.

    python tools/bench_detect.py --images 512 --workers 0,4,8
    python tools/bench_detect.py --images 1000 --workers 8 --stages

Writes nothing to the provenance database: it reads the pending list, runs the
real decode/letterbox/inference path, and discards the results. Safe to run
while nothing else holds the DB lock, and safe to interrupt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from fwdata.config import PATHS  # noqa: E402

DETECTOR = Path(r"D:\fisherwiki-data\models\fishial\detector\model.pt")


def pending_paths(n: int) -> list[tuple[str, str]]:
    import duckdb

    con = duckdb.connect(str(PATHS.provenance_db), read_only=True)
    try:
        has = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name='detections'"
        ).fetchone()[0]
        where = ("AND NOT EXISTS (SELECT 1 FROM detections d "
                 "WHERE d.sha256 = p.sha256)") if has else ""
        return con.execute(
            f"""
            SELECT p.sha256, any_value(p.cas_path) FROM provenance p
            WHERE p.sha256 IS NOT NULL AND p.cas_path IS NOT NULL {where}
            GROUP BY p.sha256 ORDER BY any_value(p.cas_path) LIMIT {int(n)}
            """
        ).fetchall()
    finally:
        con.close()


def run_once(rows, workers: int, batch: int, timeout: float) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from ultralytics.utils import nms

    import detect_fish as det

    net = YOLO(str(DETECTOR)).model
    if torch.cuda.is_available():
        net = net.to("cuda")
    net = net.eval().fuse()

    loader = DataLoader(
        det._DetectSet(rows, PATHS.cas), batch_size=batch, shuffle=False,
        num_workers=workers, collate_fn=det._collate_detect,
        prefetch_factor=(4 if workers else None),
    )

    n = 0
    t_wait = t_gpu = 0.0
    start = time.time()
    last = start
    for loaded in loader:
        t_wait += time.time() - last
        if loaded is not None:
            keep, stack = loaded
            g = time.time()
            if torch.cuda.is_available():
                stack = stack.cuda(non_blocking=True)
            stack = stack.float().div_(255)
            with torch.no_grad():
                raw = net(stack)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            dets = nms.non_max_suppression(raw, 0.25, 0.45, nc=1, max_det=10)
            [d.cpu().tolist() for d in dets]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gpu += time.time() - g
            n += len(keep)
        last = time.time()
        if last - start > timeout:
            break

    wall = time.time() - start
    return {"workers": workers, "images": n, "seconds": round(wall, 1),
            "img_s": round(n / max(wall, 1e-9), 1),
            "loader_wait_s": round(t_wait, 1), "gpu_s": round(t_gpu, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=512)
    ap.add_argument("--workers", default="0,4,8")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    rows = pending_paths(args.images)
    print(f"{len(rows)} pending images, batch {args.batch}, "
          f"{args.timeout:.0f}s cap per configuration\n", flush=True)
    print(f"{'workers':>8} {'images':>8} {'sec':>7} {'img/s':>8} "
          f"{'loader_wait':>12} {'gpu':>8}", flush=True)
    print("-" * 56, flush=True)

    best = None
    for w in [int(x) for x in args.workers.split(",") if x.strip() != ""]:
        r = run_once(rows, w, args.batch, args.timeout)
        print(f"{r['workers']:>8} {r['images']:>8} {r['seconds']:>7} "
              f"{r['img_s']:>8} {r['loader_wait_s']:>12} {r['gpu_s']:>8}",
              flush=True)
        if best is None or r["img_s"] > best["img_s"]:
            best = r
    print(f"\nfastest: {best['workers']} workers at {best['img_s']} img/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
