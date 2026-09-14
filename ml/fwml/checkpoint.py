"""Crash-safe, mid-epoch checkpointing for multi-night and spot training.

The requirement this exists for
------------------------------
V2 epochs are long. Measured on this machine, EfficientNetV2-S at 384 runs about
76 images/second, so a 3M-image epoch takes roughly eleven hours - about one
night. Waiting for an epoch boundary to checkpoint would mean a session that
stops at 09:00 throws away everything since 22:00, and a crash at hour ten costs
the whole night. So checkpoints are taken **on a wall-clock interval, mid-epoch**,
and carry enough state to continue the same epoch rather than restart it.

What "enough state" means here
------------------------------
Model, optimizer, LR scheduler, AMP scaler, epoch, global step, the number of
samples already consumed in this epoch, every RNG stream (Python, NumPy, torch
CPU, torch CUDA), the resolution schedule position, best-metric bookkeeping and
the config. The data stream is regenerated from ``(seed, epoch, rank)`` and
skipped forward by the consumed count, so it does not need to be serialised -
see :mod:`fwml.shards`.

Two RNG details that are easy to get wrong and silently degrade a run: Python's
``random`` drives augmentation choices and NumPy's drives sampling, so omitting
either makes a resumed run augment differently from an uninterrupted one. V1's
checkpoint saved torch and NumPy but not Python's.

Crash safety
------------
Writes go to a temporary file, are flushed and ``fsync``-ed, then atomically
renamed over the target. A checkpoint is only replaced once its successor is
durable, so a power cut or a spot-instance kill during a save leaves the
previous checkpoint intact. The previous checkpoint is additionally kept as
``.prev`` until the new one has been verified readable, because an atomic rename
still lets you atomically replace good data with a corrupt file if the corruption
happened before the rename.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class TrainerState:
    """Everything that is not a tensor blob, kept together and serialisable."""

    epoch: int = 0
    global_step: int = 0
    samples_this_epoch: int = 0
    samples_total: int = 0
    best_metric: float = float("-inf")
    best_epoch: int = -1
    seconds_trained: float = 0.0
    resolution: int = 0
    finished: bool = False
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "samples_this_epoch": self.samples_this_epoch,
            "samples_total": self.samples_total,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "seconds_trained": self.seconds_trained,
            "resolution": self.resolution,
            "finished": self.finished,
            "history": self.history,
        }

    @staticmethod
    def from_dict(d: dict) -> "TrainerState":
        s = TrainerState()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


def capture_rng() -> dict:
    """Every RNG stream that influences a training step."""
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng(rng: dict) -> None:
    if not rng:
        return
    if "python" in rng:
        # json round-trips tuples as lists; random.setstate demands tuples.
        state = rng["python"]
        if isinstance(state, list):
            state = (state[0], tuple(state[1]), state[2])
        random.setstate(state)
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(_as_byte_tensor(rng["torch"]))
    if "cuda" in rng and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in rng["cuda"]])
        except (RuntimeError, ValueError):
            # A checkpoint moved between machines can carry a different device
            # count. The CUDA stream only drives dropout ordering, so carrying
            # on with a fresh one is right - refusing to resume would not be.
            pass


def _as_byte_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.cpu().to(torch.uint8)
    return torch.tensor(bytearray(x), dtype=torch.uint8)


class CheckpointManager:
    """Atomic checkpoint writes with a verified previous generation kept.

    ``latest.pt`` is what a resume reads. ``best.pt`` is what export and
    evaluation read. ``latest.prev.pt`` exists only to survive the window where
    a new checkpoint turns out to be unreadable.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.latest = self.out_dir / "latest.pt"
        self.previous = self.out_dir / "latest.prev.pt"
        self.best = self.out_dir / "best.pt"
        self.run_json = self.out_dir / "run.json"

    # -- writing --------------------------------------------------------
    def save(self, payload: dict, *, path: Path | None = None,
             keep_previous: bool = True) -> Path:
        target = Path(path or self.latest)
        tmp = target.with_suffix(target.suffix + ".tmp")

        try:
            with open(tmp, "wb") as fh:
                torch.save(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            # Verify before it can replace anything: torch.save can produce a
            # file that fails to load (truncated by a full disk, for instance),
            # and an atomic rename will happily make that the only copy.
            torch.load(tmp, map_location="cpu", weights_only=False)
        except BaseException as exc:
            # Covers a save that raises midway as well as one that writes
            # something unreadable. Either way the partial file must go: leaving
            # it behind is how a later run resumes from a truncated checkpoint.
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"refusing to install unreadable checkpoint: {exc}")

        if keep_previous and target.exists():
            try:
                os.replace(target, self.previous)
            except OSError:
                pass
        os.replace(tmp, target)
        return target

    def save_training_state(
        self,
        *,
        model,
        optimizer,
        scheduler,
        scaler,
        state: TrainerState,
        config: dict,
        extra: dict | None = None,
    ) -> Path:
        payload = {
            "format": 2,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "state": state.as_dict(),
            "rng": capture_rng(),
            "config": config,
            "saved_at": time.time(),
        }
        if extra:
            payload.update(extra)
        return self.save(payload)

    def save_best(self, *, model, state: TrainerState, config: dict) -> Path:
        """Weights only: a best-checkpoint is for export and evaluation, and
        carrying optimizer state would triple its size for no reader."""
        return self.save(
            {
                "format": 2,
                "model": _unwrap(model).state_dict(),
                "state": state.as_dict(),
                "config": config,
                "saved_at": time.time(),
            },
            path=self.best,
            keep_previous=False,
        )

    # -- reading --------------------------------------------------------
    def load(self, map_location: str = "cpu") -> dict | None:
        for path in (self.latest, self.previous):
            if not path.exists():
                continue
            try:
                ckpt = torch.load(path, map_location=map_location, weights_only=False)
                if path is self.previous:
                    print(f"  latest checkpoint unreadable; fell back to {path.name}")
                return ckpt
            except Exception as exc:
                print(f"  checkpoint {path.name} failed to load ({exc}); trying older")
        return None

    def write_run_json(self, meta: dict) -> Path:
        tmp = self.run_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.run_json)
        return self.run_json


def _unwrap(model):
    """State dicts are stored unwrapped so a single-GPU run can load a
    DDP checkpoint and vice versa - otherwise every key gains a `module.`
    prefix and moving a run between one and eight GPUs stops working."""
    return getattr(model, "module", model)


def load_model_state(model, state: dict[str, Any], *, strict: bool = True):
    """Load weights saved from either a wrapped or unwrapped model."""
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    return _unwrap(model).load_state_dict(cleaned, strict=strict)
