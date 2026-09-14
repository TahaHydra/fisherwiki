"""FisherWiki V2 trainer: time-budgeted, mid-epoch resumable, DDP-ready.

    python ml/train_v2.py --shards E:/FisherWiki/shards --hours 11
    python ml/train_v2.py --shards E:/FisherWiki/shards --hours 11 --resume
    torchrun --nproc_per_node=4 ml/train_v2.py --shards /data/shards --hours 8

Designed around one fact: a V2 epoch is roughly a night. Measured here,
EfficientNetV2-S at 384 runs ~76 img/s, so 3M images is ~11 hours. A trainer
that can only stop at epoch boundaries would be unusable - so this one stops on
a wall clock, mid-epoch, and resumes into the same epoch.

Stopping
--------
Three ways, all ending in the same clean shutdown:

* ``--hours N`` - stops when the *next* step would exceed the budget, using a
  running estimate of step time, so it lands under the limit rather than over.
* ``Ctrl+C`` (SIGINT) - the first one requests a stop and finishes the current
  step; a second one within the grace window aborts immediately. Killing a
  trainer mid-``torch.save`` is how checkpoints get corrupted, so the first
  signal never interrupts a write.
* Spot/preemption (SIGTERM) - treated exactly like the first Ctrl+C.

In every case the run exits having written a checkpoint it has already verified
is loadable.

Resume
------
``--resume`` (the default when a checkpoint exists) restores model, optimizer,
scheduler, AMP scaler, all four RNG streams, and the number of samples already
consumed this epoch. The data stream is regenerated from ``(seed, epoch, rank)``
and skipped forward by that count - see :mod:`fwml.shards`. Skipping never opens
a shard, so resuming 80% into an eleven-hour epoch costs milliseconds.

The one honest limitation: resume is exact at *sample* granularity, not at
*gradient* granularity. Samples already consumed are not seen again in that
epoch, but the partially-accumulated batch at the moment of the stop is dropped
rather than reconstructed. That costs at most one batch per stop and keeps the
checkpoint free of optimizer-internal microstate that does not survive a change
in world size.

Multi-GPU
---------
Single-node DDP via ``torchrun``. Each rank streams a disjoint set of shards,
so there is no sampler coordination and no duplicated decode work. Ranks are
given an equal shard count (the remainder is dropped for the epoch) because an
uneven split deadlocks the gradient all-reduce when one rank finishes early.

Three rules keep ranks in lockstep, and all three matter on a preemptible box:

* **The data cursor is per rank.** ``samples_this_epoch`` indexes this rank's
  own slice, so it advances by the local batch, never the global one. Advancing
  it globally makes each rank skip ``world_size`` times too far on resume - at 4
  GPUs, three quarters of every resumed epoch silently never trained on.
* **Stop decisions are reduced across ranks** before anyone acts (see
  :func:`sync_stop`). A SIGTERM lands on one rank first; without the reduce that
  rank leaves the loop while the others block forever in the gradient
  all-reduce - a hang, not a crash.
* **Batches are always full.** A failed decode substitutes another row rather
  than shortening or dropping a batch, because an unequal number of optimizer
  steps between ranks hangs the all-reduce the same way.

Only rank 0 writes checkpoints, between two barriers, so the state it serialises
corresponds to a step every rank has finished and nobody proceeds until the
write is durable.

Checkpoints store unwrapped state dicts, so weights, optimizer, scheduler and
scaler move between 1 and 8 GPUs in either direction. The *data position* does
not: the stream is a function of ``(seed, epoch, world_size, rank)``, so a
cursor taken under one world size names no position under another. When the
world size changes the current epoch restarts - bounded duplication, zero
omission - and everything else carries over. See :func:`world_size_changed`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from fwml import env, models  # noqa: E402
from fwml.checkpoint import (  # noqa: E402
    CheckpointManager,
    TrainerState,
    load_model_state,
    restore_rng,
)
from fwml.shards import ShardIndex, ShardReader, ShardStream  # noqa: E402


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class V2Config:
    shards: str = ""
    val_shards: str = ""
    out: str = ""
    backbone: str = "efficientnet_v2_s"
    num_classes: int = 0
    batch_size: int = 32
    accum_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.05
    epochs: int = 15
    warmup_steps: int = 2000
    label_smoothing: float = 0.1
    seed: int = 1337
    workers: int = 8
    amp: bool = True
    ema_decay: float = 0.0
    #: Progressive resolution. Measured on this machine: 320 -> 104.6 img/s,
    #: 384 -> 76.5, 448 -> 57.6. Spending the early epochs at 320 and only the
    #: last at 448 buys most of the fine-detail benefit for much less time.
    resolution_schedule: list[tuple[float, int]] = field(
        default_factory=lambda: [(0.0, 320), (0.55, 384), (0.85, 448)]
    )
    checkpoint_minutes: float = 20.0
    grad_clip: float = 1.0

    def resolution_for(self, epoch: int) -> int:
        frac = epoch / max(1, self.epochs)
        chosen = self.resolution_schedule[0][1]
        for start, res in self.resolution_schedule:
            if frac >= start:
                chosen = res
        return chosen


# ---------------------------------------------------------------------------
# dataset over shards
# ---------------------------------------------------------------------------
class ShardRowDataset(Dataset):
    """Map-style view over a precomputed list of row indices.

    The stream decides *which* rows and in what order; this only decodes them.
    Keeping those separate is what lets resume be a pure skip over the stream
    without the dataset needing any notion of position.
    """

    def __init__(self, root: Path, rows: list[int], resolution: int,
                 train: bool, seed: int, epoch: int) -> None:
        self.root = Path(root)
        self.rows = rows
        self.resolution = resolution
        self.train = train
        self.seed = seed
        self.epoch = epoch
        self._index: ShardIndex | None = None
        self._reader: ShardReader | None = None

    def _ensure(self) -> None:
        # Opened lazily so each DataLoader worker gets its own handles rather
        # than inheriting a shared file position across a fork.
        if self._index is None:
            self._index = ShardIndex(self.root)
            self._reader = ShardReader(self._index, self.root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import io
        import random as _random

        from PIL import Image

        from fwml.crop_v2 import eval_transform_v2, train_transform_v2
        from fwml.data import AugmentConfig, to_tensor

        self._ensure()
        assert self._index is not None and self._reader is not None

        # A failed decode substitutes a neighbouring row rather than returning
        # None. Dropping a sample would shorten one rank's batch - or empty it
        # entirely - and under DDP that desynchronises the optimizer step count,
        # which hangs the gradient all-reduce rather than failing loudly. The
        # substitute carries its *own* label, so this never mislabels anything;
        # it only means a rare unreadable image is replaced by a readable one.
        row = self.rows[i]
        img = None
        for attempt in range(4):
            candidate = self.rows[(i + attempt * 7919) % len(self.rows)]
            try:
                raw = self._reader.read(candidate)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                row = candidate
                break
            except Exception:
                continue
        if img is None:
            raise RuntimeError(
                f"could not decode any of 4 candidate samples near row {row}; "
                f"the shard set is likely corrupt - run verify_shards()"
            )
        if self.train:
            rng = _random.Random(
                (self.seed * 1_000_003 + self.epoch * 7_919_837 + row) & 0x7FFFFFFF
            )
            cfg = AugmentConfig(size=self.resolution)
            img = train_transform_v2(img, cfg, rng)
        else:
            img = eval_transform_v2(img, self.resolution)
        cls = self._index.table.column("class_id")[row].as_py()
        return to_tensor(img), int(cls)


def collate_batch(batch):
    """Every item is valid by construction - see ShardRowDataset.__getitem__,
    which substitutes rather than returning None, so batches are always full
    and every rank takes the same number of optimizer steps."""
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


# ---------------------------------------------------------------------------
# stop control
# ---------------------------------------------------------------------------
class StopController:
    """Wall-clock budget plus signal handling, with writes never interrupted."""

    def __init__(self, hours: float | None, grace_seconds: float = 5.0) -> None:
        self.deadline = time.time() + hours * 3600 if hours else None
        self.requested = False
        self.reason = ""
        self._last_signal = 0.0
        self._grace = grace_seconds
        self._critical = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass                       # not the main thread; budget still applies

    def _handle(self, signum, _frame):
        now = time.time()
        if self.requested and not self._critical and now - self._last_signal < self._grace:
            print("\n  second signal - aborting now", flush=True)
            raise KeyboardInterrupt
        self._last_signal = now
        self.requested = True
        self.reason = "signal"
        print(f"\n  stop requested (signal {signum}); finishing the current step "
              f"and checkpointing. Press again to abort.", flush=True)

    def critical(self, on: bool) -> None:
        """Mark a region where a second signal must not abort - i.e. a save."""
        self._critical = on

    def should_stop(self, step_seconds: float = 0.0) -> tuple[bool, str]:
        if self.requested:
            return True, self.reason or "signal"
        if self.deadline and time.time() + step_seconds >= self.deadline:
            return True, "time budget"
        return False, ""

    def remaining(self) -> float:
        return self.deadline - time.time() if self.deadline else float("inf")


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------
def ddp_setup() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank); (0, 1, 0) when not distributed."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local)
    return rank, world, local


def is_main(rank: int) -> bool:
    return rank == 0


def advance_counters(state: TrainerState, local_batch: int, world: int) -> None:
    """Advance the two counters, which mean different things.

    ``samples_this_epoch`` is **per rank**: it is the skip offset into this
    rank's own slice of the stream. ``samples_total`` is **global**, and exists
    only for reporting throughput.

    Conflating them is not a rounding error. Incrementing the per-rank cursor by
    the global batch makes each rank skip ``world_size`` times too far on
    resume, so on 4 GPUs three quarters of every resumed epoch is silently never
    trained on - and nothing in the loss curve would show it.
    """
    state.samples_this_epoch += local_batch
    state.samples_total += local_batch * world


def sync_stop(local_stop: bool, world: int, device) -> bool:
    """Agree a stop decision across all ranks.

    Every rank must leave the loop on the same optimizer step. If one rank stops
    while another is still training, the one still training blocks forever in
    the gradient all-reduce, and the one that stopped blocks in the teardown
    barrier - a hang, not a crash, which on a rented interruptible box means
    paying for a dead machine until someone notices.

    A max-reduce means *any* rank requesting a stop stops all of them, which is
    the right polarity: SIGTERM from a preemption may land on one rank first.
    """
    if world <= 1:
        return local_stop
    import torch.distributed as dist

    if not dist.is_initialized():
        return local_stop
    flag = torch.tensor([1.0 if local_stop else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item() > 0)


def barrier(world: int) -> None:
    if world <= 1:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def world_size_changed(state: TrainerState, world: int) -> bool:
    """Whether the stored data cursor describes a different partitioning.

    The stream is a function of ``(seed, epoch, world_size, rank)``, so a cursor
    recorded under one world size names no position under another.
    """
    return int(state.world_size or 1) != int(world)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, cfg: V2Config, total_steps: int):
    import math

    warm = max(1, cfg.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, root: Path, cfg: V2Config, resolution: int, device,
             world: int, rank: int, split: str = "validation",
             max_batches: int | None = None) -> dict:
    """Top-1/top-5 on a held-out split, reduced across ranks.

    Counts are all-reduced rather than gathered: every rank evaluates its own
    shard slice and only the two integers need to meet, which keeps this cheap
    enough to run every epoch. Reporting rank 0's slice alone would silently
    measure an eighth of the split on 8 GPUs.

    `final_test` is refused here. It is not in the open manifest at all, so this
    is belt and braces rather than the seal itself - but a typo should fail
    loudly rather than quietly evaluate nothing.
    """
    if split == "final_test":
        raise ValueError(
            "final_test is sealed; evaluate on validation or dev_test. "
            "See tools/dataset.py v2-release-eval."
        )
    if not root.exists():
        return {}
    index = ShardIndex(root)
    # Shards are written one directory per split, so a directory asked for
    # "validation" may legitimately hold only "dev_test". Fall back to the
    # single split it does contain, and say so - returning an empty result
    # would look identical to "evaluated and scored zero".
    present = sorted({s for s in index.table.column("split").to_pylist()})
    if split not in present:
        if len(present) == 1:
            split = present[0]
        else:
            return {}
    stream = ShardStream(index, split, seed=cfg.seed, shuffle=False,
                         world_size=world, rank=rank)
    rows = list(stream.epoch_rows(0))
    if not rows:
        return {}
    ds = ShardRowDataset(root, rows, resolution, False, cfg.seed, 0)
    loader = DataLoader(ds, batch_size=max(1, cfg.batch_size), shuffle=False,
                        num_workers=cfg.workers, pin_memory=True,
                        collate_fn=collate_batch, drop_last=False)
    model.eval()
    top1 = top5 = seen = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.amp and device.type == "cuda"):
            logits = model(x)["species"].float()
        k = min(5, logits.size(1))
        pred = logits.topk(k, dim=1).indices
        hit = pred.eq(y.view(-1, 1))
        top1 += int(hit[:, 0].sum())
        top5 += int(hit.any(dim=1).sum())
        seen += int(y.numel())
    model.train()

    if world > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            tally = torch.tensor([top1, top5, seen], device=device, dtype=torch.float64)
            dist.all_reduce(tally, op=dist.ReduceOp.SUM)
            top1, top5, seen = (int(v) for v in tally.tolist())
    if not seen:
        return {}
    return {"split": split, "n": seen,
            "top1": round(top1 / seen, 4), "top5": round(top5 / seen, 4)}


def train(cfg: V2Config, hours: float | None, resume: bool, log=print) -> int:
    rank, world, local = ddp_setup()
    info = env.setup(quiet=not is_main(rank))
    device = torch.device(info.torch_device if info.is_gpu else "cpu")
    env.seed_everything(cfg.seed + rank)

    root = Path(cfg.shards)
    # Splits live in sibling directories (shards/train, shards/validation, ...)
    # so final_test is physically separate from anything training reads.
    val_root = Path(cfg.val_shards) if cfg.val_shards else root.parent / "validation"
    if not (val_root / "index.parquet").exists():
        val_root = None
    index = ShardIndex(root)
    if not cfg.num_classes:
        cfg.num_classes = int(max(index.table.column("class_id").to_pylist())) + 1

    ckpt_dir = Path(cfg.out)
    manager = CheckpointManager(ckpt_dir)
    state = TrainerState()

    spec = models.ModelSpec(
        backbone=cfg.backbone, num_species=cfg.num_classes, pretrained=True
    )
    model = models.build(spec)
    # Must happen before DDP wrapping: it rewrites the module tree.
    model = env.bypass_miopen_norm(model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    train_stream = ShardStream(
        index, "train", seed=cfg.seed, world_size=world, rank=rank
    )
    steps_per_epoch = max(1, len(train_stream) // (cfg.batch_size * cfg.accum_steps))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch * cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and info.is_gpu)

    ckpt = manager.load() if resume else None
    if ckpt:
        load_model_state(model, ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        state = TrainerState.from_dict(ckpt["state"])
        restore_rng(ckpt.get("rng", {}))
        if world_size_changed(state, world):
            # Restart the current epoch rather than misinterpret the cursor.
            #
            # The alternative - a world-size-independent global cursor that
            # repartitions on resume - is genuinely more efficient but needs the
            # epoch permutation to be materialised globally and re-split, which
            # is a lot of machinery to save at most one epoch of duplicated
            # work. Model, optimizer, scheduler and scaler all carry over, so
            # what is lost is bounded by the samples already seen in *this*
            # epoch. Nothing is omitted: the epoch is simply replayed in full
            # under the new partitioning.
            if is_main(rank):
                log(f"world size changed {state.world_size} -> {world}; "
                    f"restarting epoch {state.epoch} "
                    f"(discarding a {state.samples_this_epoch:,}-sample "
                    f"per-rank cursor; weights and optimizer are kept)")
            state.samples_this_epoch = 0
        state.world_size = world
        if is_main(rank):
            log(f"resumed: epoch {state.epoch}, step {state.global_step:,}, "
                f"{state.samples_this_epoch:,} samples into this epoch "
                f"(per rank, world={world})")
    elif is_main(rank):
        log("starting a new run")
    state.world_size = world

    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])

    stop = StopController(hours)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    started = time.time()
    step_time = 0.0

    if is_main(rank):
        manager.write_run_json({
            "config": asdict(cfg),
            "device": info.as_dict(),
            "world_size": world,
            "git_commit": env.git_commit(),
            "steps_per_epoch": steps_per_epoch,
            "samples_per_epoch_per_rank": len(train_stream),
        })

    while state.epoch < cfg.epochs and not state.finished:
        resolution = cfg.resolution_for(state.epoch)
        state.resolution = resolution
        rows = list(train_stream.epoch_rows(state.epoch, skip=state.samples_this_epoch))
        if is_main(rank):
            log(f"epoch {state.epoch}  res {resolution}  "
                f"{len(rows):,} samples remaining this epoch")

        if rows:
            ds = ShardRowDataset(root, rows, resolution, True, cfg.seed, state.epoch)
            loader = DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.workers, pin_memory=info.is_gpu,
                collate_fn=collate_batch, drop_last=True,
                persistent_workers=cfg.workers > 0,
                prefetch_factor=4 if cfg.workers else None,
            )
            model.train()
            last_ckpt = time.time()
            for batch in loader:
                t0 = time.time()
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16,
                                        enabled=cfg.amp and info.is_gpu):
                    out = model(x)
                    loss = criterion(out["species"], y)

                scaler.scale(loss / cfg.accum_steps).backward()
                if (state.global_step + 1) % cfg.accum_steps == 0:
                    if cfg.grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                state.global_step += 1
                advance_counters(state, x.size(0), world)
                step_time = 0.9 * step_time + 0.1 * (time.time() - t0) if step_time else \
                    time.time() - t0

                # Every rank must reach the same verdict on the same step, so
                # the decision is reduced across ranks before anyone acts on it.
                # Reducing every step costs one tiny all-reduce alongside the
                # gradient all-reduce already happening; deciding locally costs
                # a hang on a preempted machine.
                local_due = (time.time() - last_ckpt) >= cfg.checkpoint_minutes * 60
                local_stop, reason = stop.should_stop(step_time)
                stopping = sync_stop(local_stop, world, device)
                due = sync_stop(local_due, world, device)

                if due or stopping:
                    # Quiesce before writing: rank 0 serialises state that must
                    # correspond to a step every rank has finished.
                    barrier(world)
                    if is_main(rank):
                        stop.critical(True)
                        state.seconds_trained += time.time() - started
                        started = time.time()
                        manager.save_training_state(
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, state=state, config=asdict(cfg),
                        )
                        stop.critical(False)
                        log(f"  checkpoint @ step {state.global_step:,} "
                            f"({state.samples_this_epoch:,} samples this epoch "
                            f"per rank, loss {loss.item():.3f})")
                    last_ckpt = time.time()
                    # Nobody proceeds until the write is durable, so a
                    # preemption between these two barriers still leaves a
                    # complete checkpoint rather than a half-written one.
                    barrier(world)
                if stopping:
                    if is_main(rank):
                        log(f"stopping: {reason or 'requested by another rank'}")
                    _finish(world)
                    return 0

        # epoch complete
        state.epoch += 1
        state.samples_this_epoch = 0
        barrier(world)
        metrics = {}
        if val_root is not None:
            metrics = evaluate(model, val_root, cfg, resolution, device,
                               world, rank, "validation")
            if metrics and is_main(rank):
                log(f"  {metrics['split']}: top1 {metrics['top1']:.4f}  "
                    f"top5 {metrics['top5']:.4f}  (n={metrics['n']:,})")

        improved = bool(metrics) and metrics["top1"] > state.best_metric
        if is_main(rank):
            state.history.append({
                "epoch": state.epoch, "step": state.global_step,
                "resolution": resolution,
                "seconds": round(state.seconds_trained, 1),
                **({f"val_{k}": v for k, v in metrics.items() if k != "split"}),
            })
            if improved:
                state.best_metric = metrics["top1"]
                state.best_epoch = state.epoch
            manager.save_training_state(
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, state=state, config=asdict(cfg),
            )
            # Only overwrite best.pt on a real improvement: it is what export
            # and evaluation read, and replacing it every epoch would mean the
            # last epoch wins rather than the best one.
            if improved or not manager.best.exists():
                manager.save_best(model=model, state=state, config=asdict(cfg))
            log(f"epoch {state.epoch} complete"
                + (f"  (best top1 {state.best_metric:.4f} @ epoch "
                   f"{state.best_epoch})" if state.best_metric > float('-inf') else ""))
        barrier(world)

    if is_main(rank):
        log("run finished")
    _finish(world)
    return 0


def _finish(world: int) -> None:
    if world > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-shards", default=None,
                    help="defaults to <shards>/../validation")
    ap.add_argument("--hours", type=float, default=None,
                    help="stop cleanly before this many hours elapse")
    ap.add_argument("--backbone", default="efficientnet_v2_s")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--checkpoint-minutes", type=float, default=20.0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or str(Path(args.shards).parent / "runs" / f"v2_{args.backbone}")
    cfg = V2Config(
        shards=args.shards, val_shards=args.val_shards or "",
        out=out, backbone=args.backbone,
        batch_size=args.batch_size, accum_steps=args.accum_steps,
        epochs=args.epochs, lr=args.lr, workers=args.workers, seed=args.seed,
        checkpoint_minutes=args.checkpoint_minutes, amp=not args.no_amp,
    )
    return train(cfg, args.hours, resume=not args.no_resume)


if __name__ == "__main__":
    raise SystemExit(main())
