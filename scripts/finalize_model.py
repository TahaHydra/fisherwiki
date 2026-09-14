"""Take a finished training run through to a distributable pack.

    python scripts/finalize_model.py --run <run_dir> --pack-id global_v1

Runs, in order, stopping at the first failure:

1. **Calibrate on val.** Fits the temperature and picks the unknown threshold
   from the measured coverage/accuracy curve.
2. **Report test, once.** The test split is read here and nowhere else; nothing
   downstream tunes against it.
3. **Open-set evaluation.** Rejection rates against held-out species, non-fish
   images, and synthetic noise.
4. **Export fp16 ONNX**, verify it matches PyTorch, and measure the accuracy of
   the exported graph on real images.
5. **Build the pack**, with the fitted calibration embedded.
6. **Verify the pack** by loading it the way the device does.

The ordering is the point. Calibration must precede the pack build, because a
pack shipped with default thresholds would present an uncalibrated model's
confidence as if it meant something. `build_pack.py` warns when it finds no
`calibration.json`; this script makes sure it never has to.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from fwdata.config import PATHS  # noqa: E402

PY_TRAIN = REPO / ".venv-train" / "Scripts" / "python.exe"
PY_DATA = REPO / ".venv" / "Scripts" / "python.exe"


def run(label: str, argv: list[str], *, allow_fail: bool = False) -> bool:
    print("=" * 72, flush=True)
    print(f"  {label}", flush=True)
    print("=" * 72, flush=True)
    t0 = time.time()
    proc = subprocess.run([str(a) for a in argv], cwd=REPO)
    ok = proc.returncode == 0
    print(f"\n  -> {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s)\n", flush=True)
    if not ok and not allow_fail:
        raise SystemExit(f"step failed: {label}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pack-id", default="global_v1")
    ap.add_argument("--display-name", default="Global Angler")
    ap.add_argument("--pack-version", type=int, default=1)
    ap.add_argument("--quantization", default="fp16",
                    choices=["fp32", "fp16", "int8"],
                    help="fp32 ships unquantized; int8 uses the static "
                         "estimator (see docs/MODEL.md for why: dynamic INT8 "
                         "measured ~25x slower, and static's own accuracy "
                         "cost is the reason fp16 is the default)")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="cap evaluation images (0 = all)")
    ap.add_argument("--skip-openset", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    if not (run_dir / "best.pt").exists():
        raise SystemExit(f"{run_dir / 'best.pt'} not found")

    limit = ["--limit", str(args.eval_limit)] if args.eval_limit else []

    # 1. calibrate on val
    run("1/6  fit calibration on the validation split",
        [PY_TRAIN, "ml/evaluate.py", "--run", run_dir,
         "--split", "val", "--fit-calibration", *limit])

    calib = run_dir / "calibration.json"
    if not calib.exists():
        raise SystemExit("calibration.json was not written; refusing to continue")
    cal = json.loads(calib.read_text(encoding="utf-8"))
    print(f"fitted temperature {cal['temperature']:.4f}, "
          f"unknown threshold {cal['unknown_threshold']}", flush=True)

    # 2. test, exactly once
    run("2/6  report the held-out test split (read once, never tuned against)",
        [PY_TRAIN, "ml/evaluate.py", "--run", run_dir, "--split", "test", *limit])

    # 3. open set
    if not args.skip_openset:
        run("3/6  open-set rejection",
            [PY_TRAIN, "ml/evaluate_openset.py", "--run", run_dir],
            allow_fail=True)
    else:
        print("3/6  open-set evaluation skipped\n", flush=True)

    # 4. export
    #
    # This script's own --quantization vocabulary (fp32/fp16/int8) is the
    # user-facing one build_pack.py also uses, to pick which exported .onnx
    # file to embed; ml/export.py's --quantize vocabulary is one level more
    # specific (none/fp16/int8_dynamic/int8_static), because int8 needs to
    # say *which* calibration method produced it. Passing args.quantization
    # straight through used to just be wrong for "fp32" and "int8" -
    # export.py would reject "fp32" outright (not one of its choices) and
    # silently do the wrong quantisation method for "int8". Mapped explicitly
    # here instead of papering over the mismatch by widening export.py's own
    # vocabulary, which callers other than this script also depend on.
    export_quantize = {
        "fp32": "none",
        "fp16": "fp16",
        "int8": "int8_static",
    }[args.quantization]
    run(f"4/6  export {args.quantization} ONNX and measure the exported graph",
        [PY_TRAIN, "ml/export.py", "--run", run_dir,
         "--quantize", export_quantize, "--check-accuracy", "2000"])

    # 5. pack
    run("5/6  build the pack",
        [PY_DATA, "tools/build_pack.py", "--run", run_dir,
         "--pack-id", args.pack_id, "--display-name", args.display_name,
         "--pack-version", str(args.pack_version),
         "--quantization", args.quantization])

    pack = PATHS.artifacts / "packs" / f"{args.pack_id}-v{args.pack_version}.fwpack"
    if not pack.exists():
        raise SystemExit(f"expected pack at {pack}")

    # 6. verify it the way the device does
    # PY_TRAIN, not PY_DATA: verify_pack runs real ONNX inference, and
    # onnxruntime lives in the training environment.
    run("6/6  verify the pack loads and infers",
        [PY_TRAIN, "scripts/verify_pack.py", "--pack", pack])

    size_mb = pack.stat().st_size / 1e6
    print("=" * 72, flush=True)
    print(f"  pack ready: {pack}  ({size_mb:.1f} MB)", flush=True)
    print("=" * 72, flush=True)
    print("\nreports written to the run directory:", flush=True)
    for name in ("eval_val.json", "eval_test.json", "eval_openset.json",
                 "calibration.json", "export/export_report.json"):
        p = run_dir / name
        if p.exists():
            print(f"  {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
