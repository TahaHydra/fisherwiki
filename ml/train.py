#!/usr/bin/env python
"""Train a fish classifier from a built corpus.

    python ml/train.py --config ml/configs/global_v1.yaml
    python ml/train.py --config ml/configs/global_v1.yaml --resume
    python ml/train.py --config ml/configs/smoke.yaml       # 5-minute sanity run

Interrupting and re-running with the same output directory resumes from the
last completed epoch. The run directory is self-describing: it holds the exact
config, the code commit, the dataset manifest hash, per-epoch metrics and the
checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fwdata.config import PATHS  # noqa: E402
from fwml import env  # noqa: E402
from fwml.data import (  # noqa: E402
    AugmentConfig,
    FishDataset,
    collate_drop_failures,
    sqrt_balanced_sampler,
)
from fwml.models import ModelSpec, build  # noqa: E402
from fwml.train_loop import TrainConfig, train  # noqa: E402


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _default_cache() -> Path | None:
    """Use FISHERWIKI_CACHE if set, else a conventional location if present."""
    import os

    env = os.environ.get("FISHERWIKI_CACHE")
    if env:
        return Path(env)
    for c in (Path("E:/fisherwiki-cache"), PATHS.root / "cache"):
        if c.exists():
            return c
    return None


def load_manifest(corpus: str, manifest_path: Path | None = None):
    """Read the corpus manifest parquet into plain dict rows."""
    import pyarrow.parquet as pq

    path = Path(manifest_path or (PATHS.artifacts / corpus / "manifest.parquet"))
    if not path.exists():
        raise SystemExit(
            f"{path} missing - run `python tools/dataset.py build-corpus "
            f"--corpus {corpus}` first"
        )
    table = pq.read_table(path)
    rows = table.to_pylist()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, digest, path


def taxonomy_maps(rows, labels_path: Path):
    """Build class -> genus and class -> family index tensors.

    Genus and family are derived from the accepted scientific name in the
    labels file rather than looked up again, so the mapping is exactly the one
    the corpus was built with.
    """
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    classes = labels["classes"]
    num_classes = labels["num_classes"]

    genus_names, family_names = {}, {}
    genus_of = np.zeros(num_classes, dtype=np.int64)
    family_of = np.zeros(num_classes, dtype=np.int64)

    # Family is not in labels.json; recover it from the taxonomy DB when
    # available, otherwise fall back to genus so the auxiliary head still has a
    # coherent (if coarser-grained) target rather than noise.
    family_lookup = {}
    tax_db = PATHS.work / "taxonomy.duckdb"
    if tax_db.exists():
        import duckdb

        con = duckdb.connect(str(tax_db), read_only=True)
        for name, fam in con.execute(
            "SELECT canonical_name, family_name FROM taxa_joined "
            "WHERE family_name IS NOT NULL"
        ).fetchall():
            family_lookup[name] = fam
        con.close()

    for c in classes:
        cid = c["class_id"]
        sci = c["scientific_name"] or ""
        genus = sci.split(" ")[0] if sci else "?"
        fam = family_lookup.get(sci) or genus
        genus_of[cid] = genus_names.setdefault(genus, len(genus_names))
        family_of[cid] = family_names.setdefault(fam, len(family_names))

    return (
        torch.from_numpy(genus_of),
        torch.from_numpy(family_of),
        len(genus_names),
        len(family_names),
        num_classes,
        classes,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="(default behaviour)")
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--cache-root", default=None,
                    help="pre-resized image cache (see tools/prepare_cache.py)")
    args = ap.parse_args(argv)

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k in TrainConfig().__dict__})

    device_info = env.setup(prefer_gpu=not args.cpu)
    env.seed_everything(cfg.seed)
    if device_info.kind == "cpu" and cfg.amp:
        log("note        : AMP disabled on CPU")
        cfg.amp = False

    rows, dataset_hash, manifest_path = load_manifest(cfg.corpus)
    labels_path = manifest_path.parent / "labels.json"
    genus_of, family_of, n_genera, n_families, num_classes, classes = taxonomy_maps(
        rows, labels_path
    )

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_val:
        val_rows = val_rows[: args.limit_val]

    log(f"corpus      : {cfg.corpus}")
    log(f"manifest    : {manifest_path} (sha256 {dataset_hash[:16]})")
    log(f"classes     : {num_classes} species / {n_genera} genera / {n_families} families")
    log(f"train/val   : {len(train_rows):,} / {len(val_rows):,} images")

    train_cfg = AugmentConfig(size=cfg.image_size)
    eval_cfg = AugmentConfig.eval_only(size=cfg.image_size)

    cache_root = Path(args.cache_root) if args.cache_root else _default_cache()
    if cache_root:
        log(f'image cache : {cache_root}')
    train_ds = FishDataset(train_rows, PATHS.cas, train_cfg, train=True,
                           seed=cfg.seed, cache_root=cache_root)
    val_ds = FishDataset(val_rows, PATHS.cas, eval_cfg, train=False,
                         seed=cfg.seed, cache_root=cache_root)

    sampler = None
    if cfg.balanced_sampling:
        class_ids = np.array([r["class_id"] for r in train_rows], dtype=np.int64)
        sampler = sqrt_balanced_sampler(class_ids, num_classes)

    common = dict(
        num_workers=cfg.workers,
        pin_memory=device_info.is_gpu,
        collate_fn=collate_drop_failures,
        persistent_workers=cfg.workers > 0,
        prefetch_factor=4 if cfg.workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=sampler is None,
        sampler=sampler, drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **common,
    )

    spec = ModelSpec(
        backbone=cfg.backbone,
        num_species=num_classes,
        num_genera=n_genera if cfg.genus_loss_weight > 0 else 0,
        num_families=n_families if cfg.family_loss_weight > 0 else 0,
        embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
        pretrained=cfg.pretrained,
    )
    model = build(spec)

    out_dir = Path(args.out or (PATHS.artifacts / "runs" / f"{cfg.corpus}_{cfg.backbone}"))
    if args.fresh:
        for f in ("checkpoint.pt", "history.jsonl"):
            (out_dir / f).unlink(missing_ok=True)

    manifest = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device_info=device_info,
        out_dir=out_dir,
        num_classes=num_classes,
        genus_of_class=genus_of if spec.num_genera else None,
        family_of_class=family_of if spec.num_families else None,
        dataset_hash=dataset_hash,
        log=log,
    )
    log("")
    log(f"best val top-1: {manifest.get('best_val_top1', 0):.4f}")
    log(f"run directory : {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
