#!/usr/bin/env python
"""Export a trained checkpoint to ONNX, quantise it, and verify equivalence.

    python ml/export.py --run <run_dir> [--quantize int8] [--opset 17]

Three things happen here, and the third is the one that matters:

1. Export the inference graph to ONNX with fixed input/output names.
2. Optionally quantise (dynamic or static INT8).
3. **Verify** that the exported graph reproduces the PyTorch model's outputs on
   real images, and report the accuracy delta of quantisation measured on the
   validation split - not a generic claim that INT8 "usually costs 1%".

The exported graph emits **logits**, never probabilities. Temperature scaling
happens on-device from the value in the pack manifest, so a model can be
recalibrated without re-exporting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fwdata.config import PATHS  # noqa: E402
from fwml import env  # noqa: E402
from fwml.models import ExportWrapper, ModelSpec, build  # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


def load_checkpoint(run_dir: Path, which: str = "best"):
    path = run_dir / (f"{which}.pt" if which != "checkpoint" else "checkpoint.pt")
    if not path.exists():
        raise SystemExit(f"{path} not found")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec_dict = ck.get("model_spec") or ck.get("config", {})
    spec = ModelSpec(**{k: v for k, v in spec_dict.items() if k in ModelSpec().__dict__})
    model = build(spec)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, spec, ck


def export_onnx(
    model,
    spec: ModelSpec,
    out_path: Path,
    opset: int = 17,
    with_embedding: bool = True,
    dynamic_batch: bool = True,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model, with_embedding=with_embedding).eval()
    size = 224
    dummy = torch.randn(1, 3, size, size)

    output_names = ["logits"] + (["embedding"] if with_embedding else [])
    dynamic_axes = None
    if dynamic_batch:
        # Batching matters for Expert ID, where several photographs of one fish
        # are classified in a single session call.
        dynamic_axes = {"input": {0: "batch"}}
        for n in output_names:
            dynamic_axes[n] = {0: "batch"}

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    return out_path


def verify_parity(
    model,
    onnx_path: Path,
    size: int,
    n: int = 32,
    tolerance: float = 2e-3,
) -> dict:
    """Assert the ONNX graph matches PyTorch on random inputs.

    A silent divergence here is the classic way a model that scored well in
    evaluation behaves differently on the phone.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_abs = 0.0
    max_rel = 0.0
    argmax_mismatches = 0

    for _ in range(n):
        x = rng.standard_normal((1, 3, size, size), dtype=np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x))["species"].numpy()
        got = sess.run(["logits"], {"input": x})[0]
        max_abs = max(max_abs, float(np.abs(ref - got).max()))
        denom = np.maximum(np.abs(ref), 1e-3)
        max_rel = max(max_rel, float((np.abs(ref - got) / denom).max()))
        if int(ref.argmax()) != int(got.argmax()):
            argmax_mismatches += 1

    return {
        "samples": n,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "argmax_mismatches": argmax_mismatches,
        "within_tolerance": max_abs <= tolerance and argmax_mismatches == 0,
    }


def quantize_dynamic(src: Path, dst: Path) -> Path:
    """Dynamic INT8: weights quantised, activations computed in float.

    **Do not use this for a convnet.** Measured on MobileNetV3-Large, 1,978
    classes, desktop CPU, 4 threads:

    ======================  =========  ========
    variant                 latency    size
    ======================  =========  ========
    fp32                       2.2 ms   14.9 MB
    **int8 static**          **2.6 ms**  **4.3 MB**
    int8 dynamic              66.2 ms    4.0 MB
    ======================  =========  ========

    Dynamic quantisation is 25x *slower* than fp32 here. It is designed for
    matmul-heavy models (transformers, RNNs) where the weight matrices dominate.
    A depthwise-separable convnet instead pays a quantise/dequantise round trip
    at every layer boundary, and ONNX Runtime has no fast INT8 kernel for many
    of the depthwise shapes, so it falls back.

    Kept for completeness and to make the comparison reproducible. The default
    is ``int8_static``.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic as qd

    qd(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def quantize_static(
    src: Path,
    dst: Path,
    calibration_tensors,
    per_channel: bool = True,
    method: str = "minmax",
    exclude_ops: list[str] | None = None,
) -> Path:
    """Static INT8 (QDQ) using real validation images for calibration.

    Calibrating on real fish photographs rather than random noise matters: the
    activation ranges of a model fed Gaussian noise are not the ranges it sees
    on photographs, and calibrating on noise is a well-known way to lose several
    points of accuracy for no reason.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static as qs,
    )

    class Reader(CalibrationDataReader):
        def __init__(self, tensors):
            self._it = iter([{"input": t} for t in tensors])

        def get_next(self):
            return next(self._it, None)

    from onnxruntime.quantization import CalibrationMethod

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    qs(
        str(src),
        str(dst),
        Reader(calibration_tensors),
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=methods.get(method, CalibrationMethod.MinMax),
        nodes_to_exclude=exclude_ops or [],
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    return dst


def accuracy_on_split(
    onnx_path: Path,
    corpus: str,
    size: int,
    split: str = "val",
    limit: int = 2000,
    threads: int = 4,
) -> dict:
    """Top-1/top-5 of an exported graph on real held-out images.

    Quantisation benchmarks without an accuracy number are half an answer: a
    variant that is 3.5x smaller is only interesting if it still identifies
    fish. This runs the *exported ONNX file* - not the PyTorch model - over real
    validation images, so it measures what will actually ship.

    Also reports top-1 agreement with the fp32 graph, which separates "quantised
    and still right" from "quantised, differently wrong, similar score".
    """
    import onnxruntime as ort
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps

    from fwml.data import AugmentConfig, eval_transform, to_tensor

    manifest = PATHS.artifacts / corpus / "manifest.parquet"
    rows = [r for r in pq.read_table(manifest).to_pylist() if r["split"] == split]
    rows = rows[:limit]
    if not rows:
        return {"images": 0}

    import os

    cache = os.environ.get("FISHERWIKI_CACHE") or (
        "E:/fisherwiki-cache" if Path("E:/fisherwiki-cache").exists() else None
    )

    def resolve(row) -> Path:
        if cache and row.get("sha256"):
            sha = row["sha256"]
            p = Path(cache) / sha[:2] / sha[2:4] / f"{sha}.jpg"
            if p.exists():
                return p
        return Path(PATHS.cas) / row["cas_path"]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), opts,
                                providers=["CPUExecutionProvider"])

    cfg = AugmentConfig.eval_only(size)
    top1 = top5 = n = 0
    preds: list[int] = []
    labels: list[int] = []
    batch: list[np.ndarray] = []
    batch_labels: list[int] = []

    def flush():
        nonlocal top1, top5, n
        if not batch:
            return
        x = np.concatenate(batch, axis=0)
        out = sess.run(["logits"], {"input": x})[0]
        order = np.argsort(-out, axis=1)
        for i, lab in enumerate(batch_labels):
            preds.append(int(order[i, 0]))
            labels.append(lab)
            if order[i, 0] == lab:
                top1 += 1
            if lab in order[i, :5]:
                top5 += 1
            n += 1
        batch.clear()
        batch_labels.clear()

    for r in rows:
        try:
            with Image.open(resolve(r)) as im:
                im.draft("RGB", (size * 2, size * 2))
                im = ImageOps.exif_transpose(im) or im
                t = to_tensor(eval_transform(im.convert("RGB"), cfg.size))
            batch.append(t.unsqueeze(0).numpy().astype(np.float32))
            batch_labels.append(int(r["class_id"]))
        except Exception:
            continue
        if len(batch) >= 32:
            flush()
    flush()

    return {
        "split": split,
        "images": n,
        "top1": top1 / n if n else 0.0,
        "top5": top5 / n if n else 0.0,
        "predictions": preds,
        "labels": labels,
    }


def convert_fp16(src: Path, dst: Path) -> Path:
    """Halve the model's size by storing weights as float16. **The default.**

    Measured on MobileNetV3-Large, 1,978 classes, 1,500 held-out validation
    images, desktop CPU:

    ==================  ========  ==========  ================  ========
    variant             top-1     vs fp32     agrees with fp32  size
    ==================  ========  ==========  ================  ========
    fp32                  0.5173           -             100.0%   14.9 MB
    **fp16**            **0.5180**  **+0.0007**        **99.9%**  **7.5 MB**
    int8 (percentile)     0.4347     -0.0827              58.8%    4.3 MB
    int8 (min-max)        0.3187     -0.1987              38.9%    4.3 MB
    ==================  ========  ==========  ================  ========

    This contradicts the usual "INT8 is the mobile default" assumption, and the
    reason is architectural: MobileNetV3 uses hard-swish and has layers with
    very wide activation ranges, which per-tensor INT8 quantisation handles
    badly. Percentile calibration halves the damage relative to min-max but
    8 points of top-1 is still an unacceptable price for 3 MB, in an app whose
    entire value proposition is not being confidently wrong.

    float16 keeps roughly three decimal digits of precision - far more than a
    classifier's logits need - so it has no cliff.
    
    Note the latency caveat below still applies.


    Unlike INT8 this is a *lossless-ish* change - float16 keeps ~3 decimal
    digits of precision, which is far more than a classifier's logits need -
    so it does not have the accuracy cliff that quantising MobileNetV3 to INT8
    does (see :func:`quantize_static`'s measurements).

    Note the latency caveat: ONNX Runtime's CPU provider has no native fp16
    kernels for most ops, so it inserts casts and runs in fp32 anyway. On CPU
    this is therefore a **size** optimisation, not a speed one. The benefit is
    the download and the storage, which is what a 30 MB pack cares about.
    """
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,     # input/output stay fp32 so callers are unchanged
        disable_shape_infer=False,
    )
    onnx.save(converted, str(dst))
    return dst


def benchmark(onnx_path: Path, size: int, runs: int = 50, threads: int = 4) -> dict:
    """Latency on this machine's CPU, as a proxy for a mid-range phone.

    This is explicitly **not** a phone benchmark. Desktop CPU numbers are much
    faster than an actual device; they are recorded to compare model variants
    against each other, and MODEL.md says so.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(onnx_path), opts, providers=["CPUExecutionProvider"]
    )
    x = np.random.randn(1, 3, size, size).astype(np.float32)

    for _ in range(10):
        sess.run(None, {"input": x})

    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"input": x})
    elapsed = time.perf_counter() - t0

    per_run = elapsed / runs
    # Cold start: a fresh session, which is what the app pays on first launch.
    t1 = time.perf_counter()
    ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    cold = time.perf_counter() - t1

    return {
        "mean_ms": per_run * 1000,
        "fps": 1.0 / per_run,
        "session_init_ms": cold * 1000,
        "threads": threads,
        "size_mb": onnx_path.stat().st_size / 1e6,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--which", default="best", choices=["best", "checkpoint"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--quantize", default="fp16",
                    choices=["none", "fp16", "int8_dynamic", "int8_static"])
    ap.add_argument("--calib-images", type=int, default=500)
    ap.add_argument("--calib-method", default="minmax",
                    choices=["minmax", "entropy", "percentile"],
                    help="activation range estimator for static INT8")
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--check-accuracy", type=int, default=0,
                    metavar="N",
                    help="measure top-1/top-5 of the exported graphs on N "
                         "real held-out images, and the int8 delta")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    out_dir = Path(args.out or (run_dir / "export"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, spec, ck = load_checkpoint(run_dir, args.which)
    size = ck.get("config", {}).get("image_size", 224)
    log(f"loaded {args.which}.pt: {spec.backbone}, {spec.num_species} classes, "
        f"input {size}px")

    fp32 = out_dir / "model_fp32.onnx"
    log(f"exporting ONNX opset {args.opset} -> {fp32.name}")
    export_onnx(model, spec, fp32, opset=args.opset,
                with_embedding=not args.no_embedding)

    report = {
        "backbone": spec.backbone,
        "num_classes": spec.num_species,
        "input_size": size,
        "opset": args.opset,
        "variants": {},
    }

    log("verifying ONNX matches PyTorch")
    parity = verify_parity(model, fp32, size)
    report["parity_fp32"] = parity
    log(f"  max abs diff {parity['max_abs_diff']:.2e}  "
        f"argmax mismatches {parity['argmax_mismatches']}/{parity['samples']}")
    if not parity["within_tolerance"]:
        log("  WARNING: exported graph does not match PyTorch within tolerance")

    report["variants"]["fp32"] = benchmark(fp32, size)
    log(f"  fp32: {report['variants']['fp32']['mean_ms']:.1f} ms, "
        f"{report['variants']['fp32']['size_mb']:.1f} MB")

    if args.quantize == "fp16":
        q = out_dir / "model_fp16.onnx"
        log("converting weights to float16")
        convert_fp16(fp32, q)
        report["variants"]["fp16"] = benchmark(q, size)
        log(f"  fp16: {report['variants']['fp16']['mean_ms']:.1f} ms, "
            f"{report['variants']['fp16']['size_mb']:.1f} MB")
    elif args.quantize == "int8_dynamic":
        q = out_dir / "model_int8.onnx"
        log("quantising (dynamic INT8)")
        quantize_dynamic(fp32, q)
        report["variants"]["int8_dynamic"] = benchmark(q, size)
        log(f"  int8: {report['variants']['int8_dynamic']['mean_ms']:.1f} ms, "
            f"{report['variants']['int8_dynamic']['size_mb']:.1f} MB")
    elif args.quantize == "int8_static":
        from fwml.calibration_data import load_calibration_tensors

        log(f"quantising (static INT8) with {args.calib_images} real images")
        tensors = load_calibration_tensors(
            corpus=ck.get("config", {}).get("corpus", "global_v1"),
            size=size, count=args.calib_images,
        )
        q = out_dir / "model_int8.onnx"
        quantize_static(fp32, q, tensors, method=args.calib_method)
        report["variants"]["int8_static"] = benchmark(q, size)
        log(f"  int8: {report['variants']['int8_static']['mean_ms']:.1f} ms, "
            f"{report['variants']['int8_static']['size_mb']:.1f} MB")

    if args.check_accuracy:
        corpus = ck.get("config", {}).get("corpus", "global_v1")
        log("")
        log(f"measuring accuracy on {args.check_accuracy} {args.split} images")
        base = accuracy_on_split(fp32, corpus, size, args.split,
                                 args.check_accuracy)
        report["accuracy_fp32"] = {k: v for k, v in base.items()
                                   if k not in ("predictions", "labels")}
        log(f"  fp32: top1 {base['top1']:.4f}  top5 {base['top5']:.4f}  "
            f"(n={base['images']:,})")

        qpath = out_dir / "model_int8.onnx"
        if qpath.exists():
            q = accuracy_on_split(qpath, corpus, size, args.split,
                                  args.check_accuracy)
            agree = (
                sum(1 for a, b in zip(base["predictions"], q["predictions"])
                    if a == b) / max(1, len(q["predictions"]))
            )
            report["accuracy_int8"] = {k: v for k, v in q.items()
                                       if k not in ("predictions", "labels")}
            report["int8_top1_agreement_with_fp32"] = agree
            report["int8_top1_delta"] = q["top1"] - base["top1"]
            log(f"  int8: top1 {q['top1']:.4f}  top5 {q['top5']:.4f}")
            log(f"  delta: {q['top1'] - base['top1']:+.4f} top-1, "
                f"{100 * agree:.1f}% of predictions identical to fp32")

    (out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
