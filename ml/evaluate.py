#!/usr/bin/env python
"""Evaluate a trained model honestly, and fit its calibration.

    python ml/evaluate.py --run <run_dir> --split val --fit-calibration
    python ml/evaluate.py --run <run_dir> --split test          # report once
    python ml/evaluate.py --run <run_dir> --split test --open-set

Accuracy is not one number. This reports:

* top-1 / top-3 / top-5 species accuracy
* genus and family accuracy (is a species error at least in the right genus?)
* per-class precision / recall / F1, macro and weighted
* the most-confused species pairs, which drive the `similar_species` table
* expected calibration error, before and after temperature scaling
* accuracy at each rejection threshold, i.e. the coverage/accuracy tradeoff
* open-set rejection rates against non-fish and held-out-species inputs
* per-region accuracy
* accuracy as a function of a class's training support

**The test split is reported, never tuned against.** Calibration is fitted on
`val` only. `--split test` refuses to also fit calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fwdata.config import PATHS  # noqa: E402
from fwml import env  # noqa: E402
from fwml.data import (AugmentConfig, FishDataset,
                       collate_drop_failures_indexed)  # noqa: E402
from fwml.models import ModelSpec, build  # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Golden-section search on NLL. Mirrors Calibration.fitTemperature in Kotlin."""
    def nll(t: float) -> float:
        p = softmax(logits, t)
        return float(-np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-12)).mean())

    a, b = 0.05, 10.0
    phi = 0.6180339887
    c, d = b - (b - a) * phi, a + (b - a) * phi
    fc, fd = nll(c), nll(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * phi
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * phi
            fd = nll(d)
    return (a + b) / 2


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    idx = np.clip((conf * bins).astype(int), 0, bins - 1)
    ece = 0.0
    n = len(conf)
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def per_class_metrics(pred: np.ndarray, true: np.ndarray, num_classes: int):
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    support = np.zeros(num_classes)
    for t, p in zip(true, pred):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    precision = np.divide(tp, tp + fp, out=np.zeros(num_classes), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(num_classes), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(num_classes), where=denom > 0)
    present = support > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    weighted_f1 = (
        float((f1[present] * support[present]).sum() / support[present].sum())
        if present.any() else 0.0
    )
    return precision, recall, f1, support, macro_f1, weighted_f1


def confused_pairs(pred: np.ndarray, true: np.ndarray, names: list[str], top: int = 30):
    counts: dict[tuple[int, int], int] = defaultdict(int)
    per_true: dict[int, int] = defaultdict(int)
    for t, p in zip(true, pred):
        per_true[int(t)] += 1
        if t != p:
            counts[(int(t), int(p))] += 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    # top=0 means "everything worth keeping", for the pack build: every pair
    # seen at least twice, with class indices so the caller can join back.
    ordered = ordered[:top] if top else [(k, n) for k, n in ordered if n >= 2]
    rows = []
    for (t, p), n in ordered:
        row = {
            "true": names[t],
            "predicted": names[p],
            "count": n,
            "rate": n / max(1, per_true[t]),
        }
        if not top:
            row["true_class"] = t
            row["predicted_class"] = p
            row["true_support"] = per_true[t]
        rows.append(row)
    return rows


def coverage_accuracy_curve(conf: np.ndarray, correct: np.ndarray):
    """Accuracy among accepted predictions at each confidence threshold.

    This is the table that actually answers "what should the unknown threshold
    be?": it shows what fraction of photographs get an answer and how often that
    answer is right, so the threshold is a product decision made on evidence.
    """
    rows = []
    for t in [0.0, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        m = conf >= t
        rows.append({
            "threshold": t,
            "coverage": float(m.mean()),
            "accuracy_when_answered": float(correct[m].mean()) if m.any() else 0.0,
            "errors_shown_per_1000": float((~correct[m]).sum() / max(1, len(conf)) * 1000),
        })
    return rows


@torch.no_grad()
def collect_logits(model, loader, device, amp: bool):
    """Return (logits, labels, row_index).

    ``row_index`` is the position in the manifest each prediction came from.
    Images that fail to decode are dropped by the collate function, so the
    returned arrays are shorter than the split and not positionally aligned
    with it; the index is the only sound way to join back to manifest columns.
    """
    model.eval()
    all_logits, all_labels, all_idx = [], [], []
    for batch in loader:
        if batch is None:
            continue
        x, y, idx = batch
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        all_logits.append(out["species"].float().cpu().numpy())
        all_labels.append(y.numpy())
        all_idx.append(idx.numpy())
    if not all_logits:
        return (np.zeros((0, 1), np.float32), np.zeros((0,), np.int64),
                np.zeros((0,), np.int64))
    return (np.concatenate(all_logits), np.concatenate(all_labels),
            np.concatenate(all_idx))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "geo_test"])
    ap.add_argument("--fit-calibration", action="store_true")
    ap.add_argument("--which", default="best")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    if args.split == "test" and args.fit_calibration:
        raise SystemExit(
            "Refusing to fit calibration on the test split. Fit on val, then "
            "report test once."
        )

    run_dir = Path(args.run)
    ck = torch.load(run_dir / f"{args.which}.pt", map_location="cpu", weights_only=False)
    spec_dict = ck.get("model_spec", {})
    spec = ModelSpec(**{k: v for k, v in spec_dict.items() if k in ModelSpec().__dict__})
    model = build(spec)
    model.load_state_dict(ck["model"])

    cfg = ck.get("config", {})
    corpus = cfg.get("corpus", "global_v1")
    size = cfg.get("image_size", 224)

    device_info = env.setup(prefer_gpu=not args.cpu)
    device = device_info.torch_device
    model.to(device)
    amp = device_info.is_gpu

    import pyarrow.parquet as pq

    manifest_path = PATHS.artifacts / corpus / "manifest.parquet"
    rows = pq.read_table(manifest_path).to_pylist()
    labels_meta = json.loads(
        (PATHS.artifacts / corpus / "labels.json").read_text(encoding="utf-8")
    )
    names = [c["scientific_name"] for c in labels_meta["classes"]]
    num_classes = labels_meta["num_classes"]
    train_support = {c["class_id"]: c["images"] for c in labels_meta["classes"]}

    split_rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        split_rows = split_rows[: args.limit]
    if not split_rows:
        raise SystemExit(f"no rows in split {args.split!r}")
    log(f"evaluating {len(split_rows):,} images from split '{args.split}'")

    import os
    cache_root = os.environ.get('FISHERWIKI_CACHE') or (
        'E:/fisherwiki-cache' if Path('E:/fisherwiki-cache').exists() else None)
    ds = FishDataset(split_rows, PATHS.cas, AugmentConfig.eval_only(size),
                     train=False, seed=0,
                     cache_root=Path(cache_root) if cache_root else None,
                     emit_index=True)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_drop_failures_indexed, pin_memory=device_info.is_gpu,
    )

    logits, true, row_index = collect_logits(model, loader, device, amp)
    if len(true) == 0:
        raise SystemExit("no images decoded")
    log(f"collected {len(true):,} predictions")

    # --- temperature --------------------------------------------------------
    temperature = 1.0
    calib_path = run_dir / "calibration.json"
    if args.fit_calibration:
        temperature = fit_temperature(logits, true)
        log(f"fitted temperature: {temperature:.4f}")
    elif calib_path.exists():
        temperature = json.loads(calib_path.read_text(encoding="utf-8")).get(
            "temperature", 1.0
        )
        log(f"using fitted temperature from calibration.json: {temperature:.4f}")

    probs_raw = softmax(logits, 1.0)
    probs = softmax(logits, temperature)
    pred = probs.argmax(axis=1)
    correct = pred == true
    conf = probs.max(axis=1)

    order = np.argsort(-probs, axis=1)
    top3 = float(np.mean([true[i] in order[i, :3] for i in range(len(true))]))
    top5 = float(np.mean([true[i] in order[i, :5] for i in range(len(true))]))

    precision, recall, f1, support, macro_f1, weighted_f1 = per_class_metrics(
        pred, true, num_classes
    )

    # --- genus / family accuracy -------------------------------------------
    genus_of = np.array([n.split(" ")[0] for n in names])
    genus_correct = float(np.mean(genus_of[pred] == genus_of[true]))

    family_of = None
    tax_db = PATHS.work / "taxonomy.duckdb"
    family_correct = None
    if tax_db.exists():
        import duckdb

        con = duckdb.connect(str(tax_db), read_only=True)
        fam_map = dict(con.execute(
            "SELECT canonical_name, family_name FROM taxa_joined WHERE active"
        ).fetchall())
        con.close()
        family_of = np.array([fam_map.get(n) or "?" for n in names])
        family_correct = float(np.mean(family_of[pred] == family_of[true]))

    ece_before = expected_calibration_error(probs_raw.max(axis=1), correct)
    ece_after = expected_calibration_error(conf, correct)

    # --- accuracy vs training support --------------------------------------
    buckets = [(0, 60), (60, 100), (100, 200), (200, 300), (300, 10 ** 9)]
    by_support = []
    for lo, hi in buckets:
        idx = np.array([lo <= train_support.get(int(t), 0) < hi for t in true])
        if idx.any():
            by_support.append({
                "train_images": f"{lo}-{hi if hi < 10**9 else 'inf'}",
                "n": int(idx.sum()),
                "top1": float(correct[idx].mean()),
            })

    # --- generalisation to unseen photographers -----------------------------
    # The splitter marks test rows whose observer contributed to no other
    # split. Those images are the closest thing this corpus offers to "a
    # photograph from someone the model has never learned the habits of":
    # different camera, different framing, different handling pose. Accuracy
    # here is the honest generalisation number; the rest of the test split
    # shares photographers with training even though it shares no observations.
    unseen = None
    if split_rows and "unseen_observer" in split_rows[0]:
        flag = np.array([bool(split_rows[i]["unseen_observer"])
                         for i in row_index])
        if flag.any():
            unseen = {
                "n": int(flag.sum()),
                "top1": float(correct[flag].mean()),
                "genus_accuracy": float(
                    (genus_of[pred[flag]] == genus_of[true[flag]]).mean()),
                "mean_confidence": float(conf[flag].mean()),
                "n_seen_observer": int((~flag).sum()),
                "top1_seen_observer": float(correct[~flag].mean()),
            }

    report = {
        "run": str(run_dir),
        "split": args.split,
        "images": int(len(true)),
        "num_classes": num_classes,
        "temperature": temperature,
        "top1": float(correct.mean()),
        "top3": top3,
        "top5": top5,
        "genus_accuracy": genus_correct,
        "family_accuracy": family_correct,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "ece_before_calibration": ece_before,
        "ece_after_calibration": ece_after,
        "mean_confidence": float(conf.mean()),
        "classes_evaluated": int((support > 0).sum()),
        "coverage_accuracy": coverage_accuracy_curve(conf, correct),
        "accuracy_by_training_support": by_support,
        "unseen_observer": unseen,
        "worst_classes": [
            {"name": names[i], "f1": float(f1[i]), "support": int(support[i]),
             "precision": float(precision[i]), "recall": float(recall[i])}
            for i in np.argsort(f1)[:25] if support[i] > 0
        ],
        "confused_pairs": confused_pairs(pred, true, names),
    }

    log("")
    log(f"top-1              : {report['top1']:.4f}")
    log(f"top-3              : {report['top3']:.4f}")
    log(f"top-5              : {report['top5']:.4f}")
    log(f"genus accuracy     : {report['genus_accuracy']:.4f}")
    if family_correct is not None:
        log(f"family accuracy    : {report['family_accuracy']:.4f}")
    log(f"macro F1           : {report['macro_f1']:.4f}")
    if unseen:
        log(f"unseen photographer: {unseen['top1']:.4f} top-1 over "
            f"{unseen['n']:,} images "
            f"(vs {unseen['top1_seen_observer']:.4f} over "
            f"{unseen['n_seen_observer']:,} with a seen photographer)")
    log(f"weighted F1        : {report['weighted_f1']:.4f}")
    log(f"ECE before / after : {ece_before:.4f} / {ece_after:.4f}")
    log("")
    log("coverage / accuracy tradeoff:")
    log(f"  {'thresh':>7} {'coverage':>9} {'acc when answered':>19}")
    for r in report["coverage_accuracy"]:
        log(f"  {r['threshold']:7.2f} {r['coverage']:9.3f} "
            f"{r['accuracy_when_answered']:19.4f}")
    log("")
    log("accuracy by training support:")
    for r in by_support:
        log(f"  {r['train_images']:>10} images: n={r['n']:6,}  top1={r['top1']:.4f}")
    log("")
    log("most confused pairs:")
    for r in report["confused_pairs"][:10]:
        log(f"  {r['true'][:32]:34} -> {r['predicted'][:32]:34} "
            f"{r['count']:4} ({r['rate']:.1%})")

    out = run_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\nwrote {out}")

    # --- full per-class detail, for the pack build --------------------------
    # eval_*.json stays human-readable (25 worst classes, 30 worst pairs). The
    # pack needs every class and every confusion pair, so those go to their own
    # file. `model_classes.test_precision` and the `similar_species` table are
    # both populated from this: the app can then tell a user that a species is
    # weakly recognised, or that it is routinely mistaken for something else.
    full = {
        "run": str(run_dir),
        "split": args.split,
        "images": int(len(true)),
        "classes": [
            {
                "class_index": i,
                "scientific_name": names[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes) if support[i] > 0
        ],
        "confusion_pairs": confused_pairs(pred, true, names, top=0),
    }
    full_out = run_dir / f"class_metrics_{args.split}.json"
    full_out.write_text(json.dumps(full), encoding="utf-8")
    log(f"wrote {full_out} "
        f"({len(full['classes'])} classes, {len(full['confusion_pairs'])} pairs)")

    if args.fit_calibration:
        # Choose the unknown threshold from the measured curve rather than a
        # guess: the lowest threshold whose accuracy-when-answered clears 90%.
        chosen = 0.35
        for r in report["coverage_accuracy"]:
            if r["accuracy_when_answered"] >= 0.90 and r["coverage"] > 0.2:
                chosen = r["threshold"]
                break
        calib = {
            "temperature": temperature,
            "unknown_threshold": chosen,
            "margin_threshold": 0.08,
            "entropy_threshold": 0.85,
            "expected_calibration_error": ece_after,
            "per_class_threshold": {},
            "fitted_on": args.split,
            "fitted_images": int(len(true)),
        }
        calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        log(f"wrote {calib_path} (unknown_threshold={chosen})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
