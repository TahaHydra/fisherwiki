#!/usr/bin/env python
"""How much does the genus fallback rescue, and what should the threshold be?

    python ml/analyse_fallback.py --run <run_dir> --split val

The species threshold is usually presented as a straight precision/coverage
trade: raise it and you are right more often but answer less often. That framing
is wrong for this app, because a rejected species claim does not degrade to
*nothing* - the ranker aggregates the distribution by genus and offers
"some kind of Sebastes" instead.

So the real question is not "what fraction of photographs get a species answer",
it is **"what fraction get a useful answer, and how often is a shown answer
wrong"** - where a correct genus counts as useful and a wrong genus counts
against us.

This script measures that directly, so the threshold in the shipped calibration
is chosen from evidence rather than taste.
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

#: Mass that must accumulate on one genus before the app will name it.
#: Mirrors CandidateRanker.COARSE_THRESHOLD in :core.
COARSE_THRESHOLD = 0.55


def log(m: str = "") -> None:
    print(m, flush=True)


def softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = x / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--unseen-observer", action="store_true",
                    help="restrict to test rows whose photographer contributed "
                         "to no other split - the hardest, most honest case")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    spec = ModelSpec(**{k: v for k, v in ck.get("model_spec", {}).items()
                        if k in ModelSpec().__dict__})
    model = build(spec)
    model.load_state_dict(ck["model"])
    corpus = ck.get("config", {}).get("corpus", "global_v1")
    size = ck.get("config", {}).get("image_size", 224)

    info = env.setup()
    model.to(info.torch_device)

    labels_meta = json.loads(
        (PATHS.artifacts / corpus / "labels.json").read_text(encoding="utf-8")
    )
    names = [c["scientific_name"] for c in labels_meta["classes"]]
    genus_name = [n.split(" ")[0] for n in names]
    genus_ids = {g: i for i, g in enumerate(sorted(set(genus_name)))}
    genus_of_class = np.array([genus_ids[g] for g in genus_name])
    n_genera = len(genus_ids)

    import pyarrow.parquet as pq

    rows = [r for r in
            pq.read_table(PATHS.artifacts / corpus / "manifest.parquet").to_pylist()
            if r["split"] == args.split]
    if args.unseen_observer:
        rows = [r for r in rows if r.get("unseen_observer")]
        if not rows:
            raise SystemExit(
                f"no unseen-observer rows in split {args.split!r} "
                "(only the test split carries them)")
        log(f"restricted to {len(rows):,} images from photographers who "
            "contributed to no other split")
    if args.limit:
        rows = rows[: args.limit]

    import os
    cache = os.environ.get("FISHERWIKI_CACHE") or (
        "E:/fisherwiki-cache" if Path("E:/fisherwiki-cache").exists() else None)
    ds = FishDataset(rows, PATHS.cas, AugmentConfig.eval_only(size), train=False,
                     seed=0, cache_root=Path(cache) if cache else None)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate_drop_failures)

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            x = x.to(info.torch_device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=info.is_gpu):
                out = model(x)
            all_logits.append(out["species"].float().cpu().numpy())
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_labels)
    log(f"{len(y):,} images from the {args.split} split")

    cal = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    T = cal.get("temperature", 1.0)
    margin_t = cal.get("margin_threshold", 0.08)
    entropy_t = cal.get("entropy_threshold", 0.85)

    p = softmax(logits, T)
    order = np.argsort(-p, axis=1)
    top1 = p[np.arange(len(p)), order[:, 0]]
    margin = top1 - p[np.arange(len(p)), order[:, 1]]
    ent = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1) / np.log(p.shape[1])
    species_correct = order[:, 0] == y

    # Genus distribution: sum species mass within each genus.
    genus_mass = np.zeros((len(p), n_genera), dtype=np.float32)
    np.add.at(genus_mass.T, genus_of_class, p.T)
    genus_pred = genus_mass.argmax(axis=1)
    genus_conf = genus_mass.max(axis=1)
    genus_true = genus_of_class[y]
    genus_correct = genus_pred == genus_true

    log(f"species top-1 : {species_correct.mean():.4f}")
    log(f"genus top-1   : {genus_correct.mean():.4f}  "
        f"({n_genera} genera)")
    log("")
    log("What actually reaches the user at each species threshold.")
    log("A rejected species falls back to a genus claim when one genus holds")
    log(f">= {COARSE_THRESHOLD:.0%} of the mass; otherwise the app says 'uncertain'.")
    log("")
    log(f"  {'thresh':>6} | {'species':>7} {'sp.prec':>8} | {'genus':>6} "
        f"{'gen.prec':>9} | {'silent':>7} | {'useful':>7} {'wrong':>7}")
    log(f"  {'-'*6}-+-{'-'*7}-{'-'*8}-+-{'-'*6}-{'-'*9}-+-{'-'*7}-+-{'-'*7}-{'-'*7}")

    table = []
    for t in [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        accept = (top1 >= t) & (margin >= margin_t) & (ent <= entropy_t)
        # Of the rejected, which get a genus claim?
        fallback = (~accept) & (genus_conf >= COARSE_THRESHOLD)
        silent = (~accept) & (~fallback)

        n = len(y)
        sp_cov = accept.mean()
        sp_prec = species_correct[accept].mean() if accept.any() else 0.0
        gen_cov = fallback.mean()
        gen_prec = genus_correct[fallback].mean() if fallback.any() else 0.0

        # "useful" = a correct species claim or a correct genus claim.
        useful = (species_correct & accept).sum() + (genus_correct & fallback).sum()
        # "wrong" = any claim shown that was incorrect at the rank claimed.
        wrong = ((~species_correct) & accept).sum() + \
                ((~genus_correct) & fallback).sum()

        log(f"  {t:6.2f} | {sp_cov:7.1%} {sp_prec:8.1%} | {gen_cov:6.1%} "
            f"{gen_prec:9.1%} | {silent.mean():7.1%} | "
            f"{useful/n:7.1%} {wrong/n:7.1%}")
        table.append({
            "threshold": t,
            "species_coverage": float(sp_cov),
            "species_precision": float(sp_prec),
            "genus_fallback_coverage": float(gen_cov),
            "genus_fallback_precision": float(gen_prec),
            "silent": float(silent.mean()),
            "useful_answers": float(useful / n),
            "wrong_answers": float(wrong / n),
        })

    log("")
    log("Reading this table: 'useful' counts a correct species OR a correct")
    log("genus claim; 'wrong' counts any shown claim that was incorrect at the")
    log("rank it was made. 'silent' is where the app says it does not know.")

    suffix = "_unseen_observer" if args.unseen_observer else ""
    out = run_dir / f"fallback_analysis_{args.split}{suffix}.json"
    out.write_text(json.dumps({
        "split": args.split,
        "unseen_observer_only": bool(args.unseen_observer),
        "images": int(len(y)),
        "species_top1": float(species_correct.mean()),
        "genus_top1": float(genus_correct.mean()),
        "num_genera": n_genera,
        "coarse_threshold": COARSE_THRESHOLD,
        "table": table,
    }, indent=2), encoding="utf-8")
    log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
