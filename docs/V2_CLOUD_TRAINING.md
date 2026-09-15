# Training V2 on a rented GPU

Nothing here has been run. This is the workflow the local code already
supports, written down so the first cloud attempt is not also the first time
anyone thinks about spot interruption.

---

## 1. Ship shards, not originals

Upload the prepared shard set, not the CAS.

| what | files | bytes |
|---|---|---|
| CAS originals | 483,271 | 58.4 GB |
| prepared shards @512px | ~240 | ~15 GB |

Measured, not estimated: the 67,584 samples prepared so far occupy 1.9 GB, so
**30 KB/sample**. Two reasons the shards win beyond the 4× size difference:

* **240 files instead of 483,271.** Object stores and rented boxes both charge
  for round trips, and 483k small PUTs is hours of latency before a byte of
  useful work happens.
* **The derivative is the thing being trained on.** Uploading originals means
  re-running detection and preparation on the rented box — GPU-hours spent on
  work already done here.

What to upload, in full:

```
E:\FisherWiki\shards\v2\
    prepare_fingerprint.json     <- the recipe; a resume compares against it
    class_map.json               <- taxon -> class id, append-only
    shards.json                  <- run manifest
    train/       *.tar  *.tar.idx.parquet  index.parquet
    validation/  ...
    dev_test/    ...
```

**Do not upload `final_test/`.** It is sealed. `train_v2.py` refuses the split
by name, but the strongest protection is that the bytes are not on the machine.

---

## 2. Spot interruption

A spot instance gets a short warning and then dies. The three things that
matter are already implemented:

* **Checkpoints are atomic and doubled.** `CheckpointManager.save` writes to a
  temp path, fsyncs, renames, and rotates the previous file to `latest.prev.pt`.
  `load()` falls back to `latest.prev.pt` when `latest.pt` will not deserialise,
  which is exactly the file a preemption mid-write leaves behind.
* **The data cursor is per rank.** `TrainerState.samples_this_epoch` indexes
  this rank's own slice. A global count would make every rank skip `world_size`
  times too far — on 4 GPUs, three quarters of each resumed epoch silently
  dropped.
* **The stream is a pure function of `(seed, epoch, world_size, rank)`.** No
  wall clock, no dict order, no dependence on how far the previous run got. A
  replacement instance regenerates the identical order.

Set `--checkpoint-minutes` from the spot price, not from habit. At 20 minutes
a preemption costs at most 20 minutes of GPU time; at 5 it costs at most 5 but
writes 4× as often to network storage.

Sync `latest.pt`, `latest.prev.pt`, `best.pt` and `run.json` to durable storage
after each write. The local disk of a spot instance is not durable storage.

---

## 3. A different number of GPUs

`world_size_changed()` detects this and **restarts the current epoch** rather
than misreading the cursor. Bounded duplication, zero omission: model,
optimizer, scheduler and scaler all carry over, and what is lost is at most the
samples already seen in that epoch.

This is a deliberate trade. A world-size-independent global cursor would save
that epoch, but it needs the epoch permutation materialised globally and
re-split, which is a lot of machinery to avoid replaying at most one epoch.

`len(ShardStream)` truncates the shard list to a multiple of `world_size`, so
every rank gets the same number of batches. An uneven split deadlocks DDP's
gradient all-reduce when one rank runs out early — that is a hang, not an
error, and it is worth knowing about before it happens at $3/hour.

---

## 4. Dataset identity

Every checkpoint carries a `dataset_fingerprint`: the preprocessing recipe, a
hash of `class_map.json`, and the class count. A resume compares it and stops
with the difference named.

The shard **root path is deliberately not compared**. The same corpus copied
onto a rented box is the same corpus, and comparing paths would refuse every
cloud resume. `tests/test_dataset_fingerprint.py` pins that behaviour.

So: upload the shard directory whole, including `class_map.json` and
`prepare_fingerprint.json`. Uploading only the tars produces a corpus that
looks fine and fingerprints as a different dataset.

---

## 5. NVIDIA vs the local ROCm box

`fwml/env.py` resolves the device. Two differences to expect:

* `env.bypass_miopen_norm()` rewrites the module tree to work around MIOpen
  being unable to JIT-compile BatchNorm kernels on this gfx1101 card. It is a
  no-op on CUDA but must still run **before** DDP wrapping, since it rewrites
  modules.
* AMP is `torch.amp.GradScaler("cuda", ...)` on both — ROCm presents the CUDA
  API — so nothing changes in the training loop.

Pin the torch build in the cloud image. A checkpoint saved under one torch
version and loaded under another is usually fine and occasionally is not, and
finding out during a spot run is the wrong time.

---

## 6. The commands

Upload (rclone shown; any object store works):

```bash
rclone copy "E:/FisherWiki/shards/v2" remote:fisherwiki/shards/v2 \
  --exclude "final_test/**" --transfers 8 --progress
```

Train on the rented box:

```bash
python ml/train_v2.py --shards /data/shards/v2/train \
  --out /data/runs/v2 --backbone efficientnet_v2_s \
  --batch-size 32 --epochs 15 --checkpoint-minutes 10
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 ml/train_v2.py --shards /data/shards/v2/train \
  --out /data/runs/v2 --batch-size 32 --epochs 15 --checkpoint-minutes 10
```

Resume after a preemption — the same command. It finds `latest.pt`, checks the
dataset fingerprint, restores RNG, optimizer, scheduler and scaler, and
continues from the per-rank cursor.

Progress without an interactive session:

```bash
python tools/status.py
```

---

## 7. What is not done

* No instance has been rented, no bytes uploaded, no cloud run started.
* Checkpoint sync to object storage is described here but not scripted; on a
  spot instance it needs to be a background loop or a post-save hook, not a
  manual step.
* The `--hours` budget stop is implemented and tested locally but has never run
  against a real spot preemption signal.
