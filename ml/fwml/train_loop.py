"""Training loop with checkpointing, resume and a full run manifest.

Every run records enough to reproduce it: code commit, config, dataset manifest
hash, seed, per-epoch metrics, and the hardware it ran on. That manifest is
later embedded in the pack, so a model in the field can always be traced back to
the exact data and code that produced it.

Resume is real, not aspirational: a run killed mid-epoch restarts from the last
completed epoch with optimizer, scheduler, AMP scaler and RNG state restored.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import env
from .models import FishNetModel, HierarchicalLoss, count_parameters


@dataclass
class TrainConfig:
    corpus: str = "global_v1"
    backbone: str = "mobilenet_v3_large"
    image_size: int = 224
    batch_size: int = 96
    epochs: int = 30
    lr: float = 3e-3
    weight_decay: float = 2e-5
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    genus_loss_weight: float = 0.2
    family_loss_weight: float = 0.1
    embedding_dim: int = 256
    dropout: float = 0.2
    amp: bool = True
    workers: int = 8
    seed: int = 1337
    pretrained: bool = True
    ema_decay: float = 0.0
    grad_clip: float = 1.0
    balanced_sampling: bool = True
    #: Freeze the backbone for this many epochs so the new heads settle first.
    freeze_backbone_epochs: int = 1
    #: Stop if val top-1 has not improved for this many epochs (0 disables).
    early_stop_patience: int = 8

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float = 0.0
    train_top1: float = 0.0
    val_loss: float = 0.0
    val_top1: float = 0.0
    val_top3: float = 0.0
    val_top5: float = 0.0
    val_macro_f1: float = 0.0
    lr: float = 0.0
    seconds: float = 0.0
    images_per_second: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class RunState:
    """Everything needed to resume a run exactly."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.out_dir / "checkpoint.pt"
        self.best_path = self.out_dir / "best.pt"
        self.history_path = self.out_dir / "history.jsonl"
        self.manifest_path = self.out_dir / "run.json"

    def save(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer,
        scheduler,
        scaler,
        best_metric: float,
        config: TrainConfig,
    ) -> None:
        tmp = self.ckpt_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "best_metric": best_metric,
                "config": config.as_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            },
            tmp,
        )
        tmp.replace(self.ckpt_path)

    def load(self):
        if not self.ckpt_path.exists():
            return None
        return torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

    def append_history(self, m: EpochMetrics) -> None:
        with open(self.history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(m.as_dict()) + "\n")


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, ks=(1, 3, 5)):
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return {k: correct[:, : min(k, maxk)].any(dim=1).float().sum().item() for k in ks}


def macro_f1(confusion: np.ndarray) -> float:
    """Macro-averaged F1 from a confusion matrix.

    Reported alongside top-1 because top-1 on an imbalanced corpus is dominated
    by the common species; macro F1 is what tells us whether the tail works.
    """
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = confusion.sum(axis=1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    """AdamW with no weight decay on norms and biases.

    Applying decay to BatchNorm affine parameters and biases is a common,
    silent mistake; it measurably hurts on small-data classes.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Linear warmup then cosine decay, stepped per iteration."""
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_backbone_trainable(model: FishNetModel, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    amp: bool,
    criterion: nn.Module | None = None,
    max_batches: int | None = None,
):
    model.eval()
    totals = {1: 0.0, 3: 0.0, 5: 0.0}
    n = 0
    loss_sum = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

    for i, batch in enumerate(loader):
        if batch is None:
            continue
        if max_batches is not None and i >= max_batches:
            break
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
            logits = out["species"] if isinstance(out, dict) else out
            if criterion is not None:
                loss_sum += float(
                    nn.functional.cross_entropy(logits.float(), y).detach()
                ) * y.size(0)
        acc = accuracy_topk(logits.float(), y)
        for k in totals:
            totals[k] += acc[k]
        n += y.size(0)
        pred = logits.argmax(dim=1)
        np.add.at(confusion, (y.cpu().numpy(), pred.cpu().numpy()), 1)

    if n == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "loss": 0.0,
                "macro_f1": 0.0, "confusion": confusion, "n": 0}
    return {
        "top1": totals[1] / n,
        "top3": totals[3] / n,
        "top5": totals[5] / n,
        "loss": loss_sum / n,
        "macro_f1": macro_f1(confusion),
        "confusion": confusion,
        "n": n,
    }


def train(
    model: FishNetModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device_info: env.DeviceInfo,
    out_dir: Path,
    num_classes: int,
    genus_of_class: torch.Tensor | None = None,
    family_of_class: torch.Tensor | None = None,
    dataset_hash: str = "",
    log=print,
) -> dict:
    device = device_info.torch_device
    state = RunState(out_dir)
    model.to(device)

    criterion = HierarchicalLoss(
        genus_weight=cfg.genus_loss_weight,
        family_weight=cfg.family_loss_weight,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    start_epoch = 0
    best_metric = 0.0
    epochs_without_improvement = 0

    ckpt = state.load()
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        best_metric = ckpt.get("best_metric", 0.0)
        start_epoch = ckpt["epoch"] + 1
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        log(f"resumed from epoch {ckpt['epoch']} (best val top-1 {best_metric:.4f})")

    total_params, trainable_params = count_parameters(model)
    manifest = {
        "config": cfg.as_dict(),
        "model": model.spec.as_dict(),
        "num_classes": num_classes,
        "parameters_total": total_params,
        "parameters_trainable": trainable_params,
        "code_commit": env.git_commit(),
        "dataset_manifest_sha256": dataset_hash,
        "device": device_info.as_dict(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    state.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"model {cfg.backbone}: {total_params/1e6:.2f}M params, {num_classes} classes")

    if genus_of_class is not None:
        genus_of_class = genus_of_class.to(device)
    if family_of_class is not None:
        family_of_class = family_of_class.to(device)

    for epoch in range(start_epoch, cfg.epochs):
        # Must happen before the epoch's first batch: see
        # FishDataset.set_epoch for why every epoch got identical
        # augmentation without this.
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        # Head-only warmup: let the randomly-initialised heads settle before
        # letting gradients disturb pretrained features.
        if cfg.freeze_backbone_epochs:
            set_backbone_trainable(model, epoch >= cfg.freeze_backbone_epochs)

        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0.0
        seen = 0

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                out = model(x)
                g = genus_of_class[y] if genus_of_class is not None else None
                f = family_of_class[y] if family_of_class is not None else None
                loss, _parts = criterion(out, y, g, f)

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = y.size(0)
            running_loss += float(loss.detach()) * bs
            running_correct += (
                out["species"].detach().argmax(dim=1).eq(y).float().sum().item()
            )
            seen += bs

            if step % 100 == 0:
                elapsed = time.time() - t0
                rate = seen / max(elapsed, 1e-6)
                log(
                    f"  epoch {epoch} step {step}/{steps_per_epoch} "
                    f"loss {running_loss/max(seen,1):.4f} "
                    f"top1 {running_correct/max(seen,1):.4f} "
                    f"{rate:.0f} img/s lr {scheduler.get_last_lr()[0]:.2e}"
                )

        train_seconds = time.time() - t0
        val = evaluate(model, val_loader, device, num_classes, cfg.amp, criterion)

        m = EpochMetrics(
            epoch=epoch,
            train_loss=running_loss / max(seen, 1),
            train_top1=running_correct / max(seen, 1),
            val_loss=val["loss"],
            val_top1=val["top1"],
            val_top3=val["top3"],
            val_top5=val["top5"],
            val_macro_f1=val["macro_f1"],
            lr=scheduler.get_last_lr()[0],
            seconds=train_seconds,
            images_per_second=seen / max(train_seconds, 1e-6),
        )
        state.append_history(m)
        log(
            f"epoch {epoch}: train {m.train_top1:.4f} | val top1 {m.val_top1:.4f} "
            f"top3 {m.val_top3:.4f} top5 {m.val_top5:.4f} macroF1 {m.val_macro_f1:.4f} "
            f"({m.images_per_second:.0f} img/s)"
        )

        improved = val["top1"] > best_metric
        if improved:
            best_metric = val["top1"]
            epochs_without_improvement = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "val_top1": best_metric, "config": cfg.as_dict(),
                 "model_spec": model.spec.as_dict()},
                state.best_path,
            )
        else:
            epochs_without_improvement += 1

        state.save(
            epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, best_metric=best_metric, config=cfg,
        )

        if cfg.early_stop_patience and epochs_without_improvement >= cfg.early_stop_patience:
            log(f"early stop: no val improvement for {epochs_without_improvement} epochs")
            break

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["best_val_top1"] = best_metric
    state.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
