# V2 training architecture

Every number here was measured on this machine, not estimated. Where a
recommendation differs from the original V2 proposal, the measurement that
changed it is given.

---

## 1. The measurements that decided the design

RX 7800 XT (gfx1101, ROCm 10 on Windows), fp16 AMP, batch at the largest that
fits, 3,000 classes:

| config | batch | img/s | peak VRAM | epoch @2M | epoch @3M |
|---|---|---|---|---|---|
| mobilenet_v3_large @224 *(V1)* | 256 | 508.8 | 6.1 GB | 1.1 h | 1.6 h |
| efficientnet_v2_s @320 | 48 | 104.6 | 8.1 GB | 5.3 h | 8.0 h |
| **efficientnet_v2_s @384** | 32 | **76.5** | 7.8 GB | 7.3 h | **10.9 h** |
| efficientnet_v2_s @448 | 24 | 57.6 | 7.9 GB | 9.6 h | 14.5 h |
| efficientnet_v2_s @512 | 16 | 44.4 | 6.9 GB | 12.5 h | 18.8 h |
| efficientnet_v2_s @640 | 8 | 30.4 | 5.6 GB | 18.3 h | 27.4 h |
| **efficientnet_v2_s @896** | 4 | **15.9** | 5.4 GB | 34.9 h | **52.4 h** |
| efficientnet_v2_m @384 | 16 | 43.5 | 6.8 GB | 12.8 h | 19.2 h |
| convnext_tiny @384 | 24 | 86.9 | 4.9 GB | 6.4 h | 9.6 h |

ONNX Runtime, CPU, 4 threads (desktop Ryzen 7800X3D — a phone is roughly 5–15×
slower, so multiply):

| model | ms/img | fp32 ONNX |
|---|---|---|
| mobilenet_v3_large @224 | 2.5 | 21.9 MB |
| efficientnet_v2_s @384 | 70.8 | 85.3 MB |
| efficientnet_v2_s @448 | 95.6 | 85.3 MB |
| convnext_tiny @384 | 75.0 | 114.2 MB |

---

## 2. Where this disagrees with the proposal

### 896×896 is not affordable, and buys less than cropping

At 896 the GPU does **15.9 img/s**: 52 hours per epoch at 3M images. At 11 hours
a night that is **five nights per epoch**, so a 15-epoch run is 75 nights. Even
on 8×RTX 4090 it is roughly 2 hours/epoch — real money for a change that is not
where the accuracy is.

The deeper objection: **resolution is not what preserves fish detail — framing
is.** If a fish occupies ~25% of the frame, a tight crop at 384 gives it the same
pixel density as the whole frame at ~768, at one quarter of the compute. Raising
input size spends the budget uniformly across background; cropping spends it on
the fish. Do the crop, then 384 is enough.

**Chosen: progressive 320 → 384 → 448**, defaulting to
`[(0.0, 320), (0.55, 384), (0.85, 448)]`. Most epochs run at 320 where the GPU is
37% faster, and the last few at 448 to adapt the model to the resolution it will
be served at. Blended throughput ≈ 86 img/s, ~9.7 h per 3M epoch.

### Derivatives at 1080–1280 px do not fit on this machine

At 3M images, 1080 px ≈ 350 KB each ≈ **1.0 TB**. D: has 535 GB free and E: has
229 GB, so the working set would land on the NAS, which does ~100 files/s — over
**eight hours per epoch in filesystem overhead alone**, before decoding anything.

512 px short edge ≈ 100 KB ≈ **300 GB**, which fits on D: with room to spare, and
leaves headroom above the 448 training maximum for random-resized-crop. Originals
stay archived, so re-deriving larger later is always possible — that is what the
archive is for.

### The backbone choice should be settled by a pilot, not by ImageNet

EfficientNetV2-S is the better *deployment* model: smaller (85 vs 114 MB fp32),
slightly faster on ONNX CPU, better ImageNet accuracy per parameter. ConvNeXt-Tiny
is the better *training* model here: 14% faster, 37% less VRAM, and — because it
uses LayerNorm — it needs no SyncBatchNorm under DDP, which at the small per-GPU
batches this resolution forces is worth another 10–20% on 4–8 GPUs.

Both are in `BACKBONES`. **Start with `efficientnet_v2_s`**; run the other as a
one-epoch pilot on the same shards before committing the full budget.

---

## 3. A real blocker found on this machine

MIOpen on ROCm/Windows **cannot compile its BatchNorm kernels at all**: the
wheels ship no C++ standard library headers, so the runtime JIT fails with
`miopenStatusUnknownError` on every BatchNorm call, in train *and* eval.
Convolution and GroupNorm are unaffected.

`env.py` already worked around this by disabling cuDNN/MIOpen globally. Measured
cost of that blunt fix: **2× on a plain convolution stack** (146 → 72 img/s).
`env.bypass_miopen_norm(model)` is the narrow fix — wrap only the BatchNorm
layers, leave MIOpen on for convolution. End-to-end on EfficientNetV2-S @384 it
is worth ~5% (72.8 → 76.5 img/s), far less than the microbenchmark implies,
because depthwise convolutions dominate that model and MIOpen accelerates them
much less than dense ones. It is applied automatically by the trainer and is a
no-op on CUDA.

This is a local toolchain defect, not an architecture constraint. On NVIDIA it
disappears entirely.

---

## 4. Storage layout

| tier | measured | role |
|---|---|---|
| **NAS** 7 TB | 14.9 MB/s w, 49.7 MB/s r, ~100 files/s | Archive only: originals, raw dumps, finished shard sets, checkpoint history. The training loop never reads it. |
| **D:** 535 GB | 61.6 MB/s w, 120.5 MB/s r | Prepared shards (~300 GB at 3M), checkpoints. Sequential shard reads suit it; random access does not. |
| **E:** 229 GB | 1007 MB/s w, 2203 MB/s r | Hot tier: active DuckDB, index, the shard subset being consumed, scratch. |

The access pattern is deliberately **sequential within shards**, because that is
what this storage rewards. A single GPU at 76–105 img/s × ~100 KB needs only
8–11 MB/s — even D: serves that comfortably, and even the NAS could keep up on
bandwidth if not for its file-open cost. Eight GPUs need ~200 MB/s, which is E:
territory or a cloud NVMe.

---

## 5. Shard format

Plain **tar** + a Parquet sidecar index (`ml/fwml/shards.py`). Tar because it is
the most portable container that exists — readable by WebDataset, by `tar -x`,
and by twenty lines of Python — which matters when the final run may happen on a
rented NVIDIA box. Nothing in it is specific to this GPU, this OS or PyTorch.

The index carries byte offset, length, class id, sha256, split and leak-group id
per sample, which gives three things V1 could not do:

* sampling, split filtering and epoch permutation over ~16 bytes/sample in RAM
  (~48 MB at 3M) instead of opening files;
* **skip-ahead resume** — seeking past consumed samples costs a seek, not a
  decode, so resuming 80% into an 11-hour epoch is instant;
* verification — `verify_shards()` re-hashes against the index, so a bad transfer
  to a rented machine is caught rather than showing up as unexplained accuracy
  loss.

---

## 6. Nightly stop and resume

```powershell
.\train-v2.ps1 -Hours 11
```

Three stop paths, one clean shutdown: the wall-clock budget (stops when the
*next* step would exceed it), `Ctrl+C` (finishes the step, then checkpoints;
press twice to abort), and `SIGTERM` from a spot preemption, treated as the
first `Ctrl+C`. A checkpoint write is never interrupted by the first signal —
that is precisely how checkpoints get corrupted.

Checkpoints are also written every `--checkpoint-minutes` (default 20), so a
crash costs at most that much.

**What is saved:** model, optimizer, scheduler, AMP scaler, epoch, global step,
samples consumed this epoch, all four RNG streams (Python, NumPy, torch CPU,
torch CUDA), resolution position, best metrics, config.

**What makes it correct:** the data stream is a pure function of
`(seed, epoch, world_size, rank)`, so a resume regenerates the identical epoch
and skips forward by the consumed count. Nothing about the order depends on
worker count, dict iteration or how far the last run got.

**Crash safety:** writes go to a temp file, are `fsync`-ed, loaded back to prove
they are readable, and only then atomically renamed. The previous generation is
kept as `latest.prev.pt` and is used automatically if the newest fails to load.
A failed save never destroys the last good checkpoint.

**The one honest limitation:** resume is exact at *sample* granularity, not
*gradient* granularity. The partially-accumulated batch at the moment of the stop
is dropped rather than reconstructed — at most one batch per stop. That keeps the
checkpoint free of optimizer microstate that would not survive a change in world
size, which is what lets a run move between 1 and 8 GPUs.

---

## 7. Multi-GPU and cloud portability

`torchrun --nproc_per_node=N ml/train_v2.py ...` — single-node DDP. Each rank
streams a **disjoint set of shards**, so there is no sampler coordination and no
duplicated decode. Ranks receive an equal shard count (the remainder is dropped
for that epoch) because an uneven split deadlocks the gradient all-reduce when
one rank runs out of batches early.

State dicts are stored **unwrapped**, so a checkpoint moves between 1 and N GPUs
in either direction. The CUDA RNG restore tolerates a changed device count rather
than refusing to resume.

Portability is close to free here because ROCm exposes the `torch.cuda` API, so
there is one code path, not two. The shard format is vendor-neutral tar. What
does *not* transfer exactly: cuDNN vs MIOpen kernel selection makes bitwise
reproducibility impossible across backends — metrics will match to within normal
run-to-run noise, not exactly. That limitation is inherent, not a design choice.

Expected DDP scaling for this model and data path: near-linear to 4 GPUs, ~0.85–0.9
efficiency at 8, with the constraint moving from GPU to JPEG decode. At 8×4090
(~250 img/s each ≈ 2,000 img/s) decode needs roughly 40–60 CPU cores at 512 px —
which is why the derivative size matters as much as the GPU does.

---

## 8. What is not built yet

* **The detector.** The schema hook and crop path exist (`detections` table,
  `--crop-pad`); no boxes have been generated, so shards currently store whole
  frames. This is the single largest expected accuracy gain and the next real
  piece of work.
* **Open-set, calibration, genus fallback** for V2 — V1's mechanisms carry over
  but need refitting on `dev_test`, never `final_test`.
* **On-device detector + pack format changes** for the larger model.
