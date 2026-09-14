#!/usr/bin/env python
"""Decide the V2 backbone on FisherWiki validation numbers, not ImageNet ones.

    python ml/pilot_backbone.py --shards D:/fisherwiki-data/v2/shards --hours 2

Trains each candidate for an equal budget on identical shards and reports
validation top-1/top-5. Equal *time*, not equal epochs: the question is which
model is better for a fixed amount of this machine, and the candidates differ by
14% in throughput, so equal epochs would quietly hand one of them more compute.

The candidates, and why these two:

* ``efficientnet_v2_s`` - the better deployment model. 85 MB fp32 ONNX against
  114, slightly faster on ONNX CPU, better ImageNet accuracy per parameter.
* ``convnext_tiny`` - the better training model *here*. Measured 86.9 img/s
  against 76.5 at 384, 4.9 GB against 7.8, and it uses LayerNorm so it needs no
  SyncBatchNorm under DDP, which at the small per-GPU batches this resolution
  forces is worth another 10-20% on 4-8 GPUs.

ImageNet says EfficientNetV2-S should win on accuracy. ImageNet is not
fine-grained fish photography taken by anglers, which is the entire reason this
script exists rather than a citation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from train_v2 import V2Config, train  # noqa: E402

CANDIDATES = ("efficientnet_v2_s", "convnext_tiny")

#: Per-candidate batch size at 320px, from the measured VRAM ceilings.
#: ConvNeXt fits more; giving both the same batch would waste its advantage.
BATCH = {"efficientnet_v2_s": 32, "convnext_tiny": 40}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True,
                    help="root holding train/ and validation/ subdirectories")
    ap.add_argument("--out", default=r"D:\fisherwiki-data\v2\pilots")
    ap.add_argument("--hours", type=float, default=2.0,
                    help="budget PER candidate")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=320)
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    args = ap.parse_args(argv)

    root = Path(args.shards)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []

    for backbone in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        print(f"\n{'=' * 70}\npilot: {backbone}  ({args.hours}h budget)\n{'=' * 70}",
              flush=True)
        cfg = V2Config(
            shards=str(root / "train"),
            val_shards=str(root / "validation"),
            out=str(out_root / backbone),
            backbone=backbone,
            batch_size=BATCH.get(backbone, 32),
            epochs=args.epochs,
            workers=args.workers,
            lr=1e-3,
            # Fixed resolution for the comparison: progressive resizing would
            # give each candidate a different schedule for the same wall clock
            # and make the numbers incomparable.
            resolution_schedule=[(0.0, args.resolution)],
            checkpoint_minutes=15.0,
        )
        t0 = time.time()
        train(cfg, hours=args.hours, resume=True, log=print)
        elapsed = time.time() - t0

        from fwml.checkpoint import CheckpointManager, TrainerState

        ck = CheckpointManager(Path(cfg.out)).load()
        state = TrainerState.from_dict(ck["state"]) if ck else TrainerState()
        best_val = [h for h in state.history if "val_top1" in h]
        results.append({
            "backbone": backbone,
            "hours": round(elapsed / 3600, 2),
            "epochs_completed": state.epoch,
            "global_step": state.global_step,
            "samples_seen": state.samples_total,
            "best_val_top1": state.best_metric if state.best_metric > float("-inf") else None,
            "best_epoch": state.best_epoch,
            "history": best_val[-6:],
        })
        print(json.dumps(results[-1], indent=2), flush=True)

    report = out_root / "pilot_report.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\nPILOT RESULT\n{'=' * 70}")
    for r in sorted(results, key=lambda r: -(r["best_val_top1"] or -1)):
        print(f"  {r['backbone']:22} val top1 {r['best_val_top1']}  "
              f"({r['epochs_completed']} epochs, {r['samples_seen']:,} samples, "
              f"{r['hours']}h)")
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
