#!/usr/bin/env python
"""Open-set evaluation: does the model refuse things it should refuse?

    python ml/evaluate_openset.py --run <run_dir>

The closed-set metrics in `evaluate.py` answer "when the answer is in the class
list, how often is it right?". This answers the question that actually decides
whether the app is safe to use: **when the answer is not in the class list, does
it say so?**

Negative sets, hardest first
----------------------------
``held_out_fish``
    Real fish photographs of species that exist in iNaturalist but did **not**
    clear the evidence bar, so the model has never seen them. This is by far the
    most important case and the hardest: a *Sebastes* the model does not know
    looks exactly like the ones it does. It is also the realistic one - the
    model covers 1,978 species and the world has ~35,000.

``non_fish``
    Birds, mammals, insects, plants and other non-fish taxa from the same
    source, so lighting, framing and photographer behaviour match. The model
    must not confidently name a heron as a fish.

``synthetic``
    Noise, flat colours and gradients. Easy, and included only as a floor: a
    model that fails *this* is broken rather than merely overconfident.

What is reported
----------------
For each set, the rejection rate under the shipped calibration, plus the
rejection rate as a function of threshold so the tradeoff against closed-set
coverage is visible rather than asserted. A rejection rule that refuses
everything scores perfectly here and is useless, so the closed-set coverage is
reported alongside and the two must be read together.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fwdata.config import PATHS  # noqa: E402
from fwml import env  # noqa: E402
from fwml.data import AugmentConfig, FishDataset, collate_drop_failures  # noqa: E402
from fwml.models import ModelSpec, build  # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    n = p.shape[1]
    h = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    return h / np.log(n)


def rejected(
    probs: np.ndarray,
    unknown_threshold: float,
    margin_threshold: float,
    entropy_threshold: float,
) -> np.ndarray:
    """The same three-signal rule the device applies, vectorised."""
    order = np.sort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    ent = normalized_entropy(probs)
    return (
        (top1 < unknown_threshold)
        | (margin < margin_threshold)
        | (ent > entropy_threshold)
    )


def synthetic_negatives(n: int, size: int, seed: int = 0) -> np.ndarray:
    """Noise, flats and gradients, already normalised. The easy floor."""
    rng = np.random.default_rng(seed)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    out = np.zeros((n, 3, size, size), dtype=np.float32)
    for i in range(n):
        kind = i % 4
        if kind == 0:
            img = rng.random((3, size, size), dtype=np.float32)
        elif kind == 1:
            img = np.full((3, size, size), rng.random(), dtype=np.float32)
        elif kind == 2:
            g = np.linspace(0, 1, size, dtype=np.float32)
            img = np.stack([np.tile(g, (size, 1))] * 3)
        else:
            g = np.linspace(0, 1, size, dtype=np.float32)
            img = np.stack([np.tile(g.reshape(-1, 1), (1, size))] * 3)
            img = img * rng.random() + rng.random() * 0.2
        out[i] = (np.clip(img, 0, 1) - mean) / std
    return out


@torch.no_grad()
def logits_for_loader(model, loader, device, amp: bool) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in loader:
        if batch is None:
            continue
        x, _ = batch
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        chunks.append(out["species"].float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, 1), np.float32)


@torch.no_grad()
def logits_for_array(model, arr: np.ndarray, device, amp: bool, bs: int = 128):
    model.eval()
    chunks = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        chunks.append(out["species"].float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, 1), np.float32)


def load_negative_rows(kind: str, limit: int, size: int, corpus: str = "global_v1") -> list[dict]:
    """Rows for a negative set, drawn from the provenance store."""
    import duckdb

    con = duckdb.connect(str(PATHS.provenance_db), read_only=True)
    try:
        if kind == "held_out_fish":
            # Fish we stored but whose *species* did not make the class list.
            # These are genuine fish photographs of species the model has
            # never seen.
            #
            # The obvious-looking query checks candidate_id membership in
            # corpus_members instead of taxon_id, which is wrong: it asks
            # "was this exact photo excluded", not "was this species
            # excluded". Measured impact on this corpus - 1,250 of 4,414
            # candidate rows (28.3%), spanning 629 distinct species, belong
            # to species that *are* trained classes. 1,243 of those 1,250
            # (99.4%) are exact sha256 duplicates of a photo that won the
            # cross-candidate dedup tie-break in splits.py's `eligible` CTE
            # and so is genuinely in corpus_members under a different
            # candidate_id - re-uploads, cross-posts, the same observation
            # submitted twice. A model that has trained on the winning
            # duplicate should recognise the losing one confidently and
            # correctly, which is the opposite of what this set is meant to
            # measure: those rows drag the reported rejection rate toward
            # "the model correctly answers a species it knows", not away
            # from open-set failure.
            #
            # The species-level check below is correct regardless of which
            # candidate happened to win that tie-break, and is scoped to the
            # corpus actually being evaluated rather than any corpus that
            # happens to share a provenance store.
            rows = con.execute(
                """
                SELECT p.sha256, p.cas_path, p.accepted_scientific_name AS name
                FROM provenance p
                WHERE p.species_taxon_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM corpus_members m
                      WHERE m.corpus = ?
                        AND m.taxon_id = p.species_taxon_id
                  )
                ORDER BY hash(p.sha256)
                LIMIT ?
                """,
                [corpus, int(limit)],
            ).fetchall()
        elif kind == "non_fish":
            rows = con.execute(
                f"""
                SELECT p.sha256, p.cas_path, p.accepted_scientific_name AS name
                FROM provenance p
                WHERE p.source_dataset = 'inaturalist-nonfish'
                ORDER BY hash(p.sha256)
                LIMIT {int(limit)}
                """
            ).fetchall()
        else:
            raise SystemExit(f"unknown negative kind {kind!r}")
    finally:
        con.close()
    # class_id 0, not -1. The label is unused for a negative - we only read the
    # logits - but -1 is `collate_drop_failures`'s sentinel for "this image
    # failed to decode", so every negative was being silently dropped from the
    # batch and both negative sets evaluated as empty.
    return [
        {"sha256": s, "cas_path": c, "class_id": 0, "name": n} for s, c, n in rows
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="best")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    ck = torch.load(run_dir / f"{args.which}.pt", map_location="cpu", weights_only=False)
    spec = ModelSpec(
        **{k: v for k, v in ck.get("model_spec", {}).items()
           if k in ModelSpec().__dict__}
    )
    model = build(spec)
    model.load_state_dict(ck["model"])

    cfg = ck.get("config", {})
    corpus = cfg.get("corpus", "global_v1")
    size = cfg.get("image_size", 224)

    device_info = env.setup(prefer_gpu=not args.cpu)
    device = device_info.torch_device
    model.to(device)
    amp = device_info.is_gpu

    calib_path = run_dir / "calibration.json"
    if not calib_path.exists():
        raise SystemExit(
            "calibration.json missing - run "
            "`ml/evaluate.py --split val --fit-calibration` first. Open-set "
            "numbers under uncalibrated defaults would be meaningless."
        )
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    T = calib.get("temperature", 1.0)
    ut = calib.get("unknown_threshold", 0.35)
    mt = calib.get("margin_threshold", 0.08)
    et = calib.get("entropy_threshold", 0.85)
    log(f"calibration: T={T:.3f} unknown>={ut} margin>={mt} entropy<={et}")

    import os

    cache = os.environ.get("FISHERWIKI_CACHE") or (
        "E:/fisherwiki-cache" if Path("E:/fisherwiki-cache").exists() else None
    )
    eval_cfg = AugmentConfig.eval_only(size)
    report: dict = {"run": str(run_dir), "calibration": calib, "sets": {}}

    def evaluate_set(name: str, logits: np.ndarray, expect_reject: bool):
        if len(logits) == 0:
            log(f"{name}: no images available, skipped")
            return
        probs = softmax(logits, T)
        rej = rejected(probs, ut, mt, et)
        top1 = probs.max(axis=1)
        curve = []
        for t in [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            r = rejected(probs, t, mt, et)
            curve.append({"unknown_threshold": t, "rejection_rate": float(r.mean())})
        entry = {
            "images": int(len(logits)),
            "rejection_rate": float(rej.mean()),
            "mean_top1_confidence": float(top1.mean()),
            "confidently_accepted": int((~rej).sum()),
            "worst_case_confidence": float(top1.max()),
            "threshold_curve": curve,
        }
        report["sets"][name] = entry
        verdict = "rejected" if expect_reject else "accepted"
        log(f"{name:16} n={entry['images']:6,}  {verdict} "
            f"{100*entry['rejection_rate']:5.1f}%  "
            f"mean conf {entry['mean_top1_confidence']:.3f}  "
            f"max conf {entry['worst_case_confidence']:.3f}")

    # --- closed set, for the coverage side of the tradeoff ------------------
    import pyarrow.parquet as pq

    manifest = PATHS.artifacts / corpus / "manifest.parquet"
    rows = [r for r in pq.read_table(manifest).to_pylist() if r["split"] == "test"]
    rows = rows[: args.limit]
    ds = FishDataset(rows, PATHS.cas, eval_cfg, train=False, seed=0,
                     cache_root=Path(cache) if cache else None)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate_drop_failures)
    closed = logits_for_loader(model, loader, device, amp)
    labels = np.array([r["class_id"] for r in rows[: len(closed)]])
    probs = softmax(closed, T)
    rej = rejected(probs, ut, mt, et)
    correct = probs.argmax(axis=1) == labels
    report["closed_set"] = {
        "images": int(len(closed)),
        "coverage": float((~rej).mean()),
        "accuracy_when_answered": float(correct[~rej].mean()) if (~rej).any() else 0.0,
        "accuracy_overall": float(correct.mean()),
    }
    log("")
    log(f"closed set (test): {len(closed):,} images, "
        f"answers {100*report['closed_set']['coverage']:.1f}% of them, "
        f"correct {100*report['closed_set']['accuracy_when_answered']:.1f}% "
        f"when it answers")
    log("")

    # --- negatives ----------------------------------------------------------
    for kind in ("held_out_fish", "non_fish"):
        try:
            neg_rows = load_negative_rows(kind, args.limit, size, corpus=corpus)
        except SystemExit:
            raise
        except Exception as exc:
            log(f"{kind}: unavailable ({exc})")
            continue
        if not neg_rows:
            log(f"{kind:16} no images available - see docs/MODEL.md")
            continue
        nds = FishDataset(neg_rows, PATHS.cas, eval_cfg, train=False, seed=0,
                          cache_root=Path(cache) if cache else None)
        nloader = DataLoader(nds, batch_size=args.batch_size,
                             num_workers=args.workers,
                             collate_fn=collate_drop_failures)
        evaluate_set(kind, logits_for_loader(model, nloader, device, amp), True)

    synth = synthetic_negatives(min(1000, args.limit), size)
    evaluate_set("synthetic", logits_for_array(model, synth, device, amp), True)

    out = run_dir / "eval_openset.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
