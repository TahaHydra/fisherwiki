# Training

Everything here is reproducible from this repository. Commands are given for the
reference machine (Windows, AMD GPU); the notes say where a CUDA/Linux box
differs.

---

## 1. Environment

Two virtual environments, deliberately separate so that a broken ML stack cannot
take the data tooling down with it:

```bash
# data tooling
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# training
py -m venv .venv-train
.venv-train/Scripts/python -m pip install \
    --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    --extra-index-url https://pypi.org/simple \
    torch torchvision amd-torch-device-gfx1101 amd-torchvision-device-gfx1101
.venv-train/Scripts/python -m pip install -r requirements-train.txt
```

On CUDA, replace the middle command with the standard PyTorch CUDA wheel. On
CPU, plain `pip install torch torchvision` works and `ml/fwml/env.py` detects it.

### AMD / ROCm on Windows: two things you must know

**1. MIOpen cannot compile its BatchNorm kernels.** Every `torchvision` model
fails with `RuntimeError: miopenStatusUnknownError`. The cause is that MIOpen
JIT-compiles those kernels with HIPRTC and they `#include <type_traits>`; the
ROCm Windows wheels ship no C++ standard library headers at all, including after
installing `rocm-sdk-devel`. `ml/fwml/env.py` sets
`torch.backends.cudnn.enabled = False` automatically on this platform, which
routes convolution and batch-norm to PyTorch's native implementations.

This flag is applied **only** on ROCm/Windows. On CUDA or ROCm/Linux, leaving
cuDNN/MIOpen enabled is much faster and the flag is not set.

**2. fp32 is broken-slow, so AMP is mandatory.** Measured 4096³ matmul:

| precision | time | throughput |
|---|---|---|
| fp32 | 76.6 ms | 1.80 TFLOPS |
| fp16 | 3.0 ms | **45.47 TFLOPS** |

A 25× gap is far beyond the expected 2×; fp32 evidently is not reaching tuned
kernels on gfx1101/Windows. `amp: true` is the default in every config.

The first convolution after a fresh install may fail while MIOpen populates its
kernel database. Re-run; it succeeds. The cache is pinned to a project-local
directory so it is reproducible and disposable.

---

## 2. Storage: the thing that actually determined feasibility

This is the single most consequential performance finding in the project, so it
gets its own section.

Training initially ran at **63 images/second** while the GPU benchmarks at 374.
Profiling the loader showed the decode and augmentation costing ~6 ms per image
but `__getitem__` taking 17.8 ms. The missing 12 ms was **seek latency**: the
data root was on a spinning SATA disk, and the corpus is 307,415 small files
read in random order. No number of DataLoader workers fixes that.

At 63 img/s a 30-epoch run would have taken **32 hours**.

The fix is `tools/prepare_cache.py`, which does two things in one pass:

* re-encodes every corpus image at a **256 px short edge** (training crops to
  224, so everything above that is wasted bytes and wasted JPEG decode), and
* writes the result to an **SSD**.

It also enables `Image.draft()`, which lets libjpeg decode at 1/2, 1/4 or 1/8
scale directly in the DCT domain — free, and meaningful when reading a 500 px
original.

```bash
.venv-train/Scripts/python tools/prepare_cache.py \
    --corpus global_v1 --cache-root E:/fisherwiki-cache --workers 14
```

The cache is **derived data**. The content-addressed store remains the source of
truth; the cache is keyed by the same SHA-256, can be deleted at any time, and
the dataset falls back to the CAS per-image when an entry is missing, so a
partial cache still trains.

`ml/train.py` picks it up from `--cache-root`, `$FISHERWIKI_CACHE`, or a
conventional path, in that order.

> **If you are reproducing this:** put the image corpus on flash. If you cannot,
> build the cache anyway — the 4× decode reduction is worth having on its own.

---

## 3. The pipeline

```bash
# 1. corpus (see docs/DATASETS.md for the data acquisition steps)
.venv/Scripts/python tools/dataset.py build-corpus --corpus global_v1

# 2. fast cache
.venv-train/Scripts/python tools/prepare_cache.py \
    --corpus global_v1 --cache-root E:/fisherwiki-cache

# 3. train
.venv-train/Scripts/python ml/train.py --config ml/configs/global_v1.yaml

# 4. calibrate on val, then report test exactly once
.venv-train/Scripts/python ml/evaluate.py --run <run_dir> --split val --fit-calibration
.venv-train/Scripts/python ml/evaluate.py --run <run_dir> --split test

# 5. export + quantise
.venv-train/Scripts/python ml/export.py --run <run_dir> --quantize int8_static

# 6. pack
.venv/Scripts/python tools/build_pack.py --run <run_dir> \
    --pack-id global_v1 --display-name "Global Angler"
```

A five-minute sanity run over the whole path:

```bash
.venv-train/Scripts/python ml/train.py --config ml/configs/smoke.yaml \
    --limit-train 3000 --limit-val 600
```

---

## 4. Resume

Interrupting and re-running the same command resumes from the last completed
epoch, restoring the optimizer, the LR scheduler, the AMP scaler and both RNG
states. `--fresh` discards the checkpoint and starts over.

Every run directory is self-describing:

```
run.json         config, model spec, parameter counts, code commit,
                 dataset manifest SHA-256, device description, timestamps
history.jsonl    one line per epoch: losses, top-1/3/5, macro F1, LR, img/s
checkpoint.pt    full resumable state
best.pt          best-validation weights only
calibration.json fitted temperature and thresholds
eval_*.json      evaluation reports
```

---

## 5. Training recipe, and why

| Choice | Value | Reason |
|---|---|---|
| Backbone | MobileNetV3-Large | 374 img/s measured vs 191 for EfficientNet-B0 on this GPU, at similar accuracy class. See `MODEL.md`. |
| Input | 224 px | Standard; the pack manifest carries it so a future 288 px model needs no code change. |
| Batch | 96 | Fits in 2.4 GB with AMP, leaving headroom. |
| Optimiser | AdamW, lr 3e-3 | Norms and biases excluded from weight decay — including them measurably hurts small-data classes. |
| Schedule | 2-epoch linear warmup, then cosine | Stepped per iteration. |
| Label smoothing | 0.1 | Species head only. We calibrate explicitly afterwards, so a large value would just blur the signal. |
| Freeze backbone | 1 epoch | Lets the randomly-initialised heads settle before gradients disturb pretrained features. |
| Sampling | sqrt inverse-frequency | Full balancing over-samples a 40-image class ~7× per epoch and overfits exactly the classes least able to afford it. |
| Aux losses | genus 0.2, family 0.1 | Well below the species loss, so they shape features without dominating. |
| Grad clip | 1.0 | |
| Early stop | 8 epochs without val top-1 improvement | |

### Augmentation

Chosen for angler photographs specifically. The full rationale is in
`ml/fwml/data.py`; the two decisions worth repeating here:

* **Saturation and hue jitter are kept very low** (0.15 / 0.02). Colour pattern
  is a primary field mark — the red fins of a rudd versus a roach, the flank
  spots of a brown versus a rainbow trout. The standard ImageNet recipe would
  train the model to ignore the most reliable diagnostic feature available. This
  is the augmentation most worth ablating.
* **No vertical flip.** Fish are photographed dorsal-up essentially always;
  training upside-down fish spends capacity on a pose that never occurs.

Blur, JPEG recompression and random erasing *are* used, because wet lenses,
messaging-app recompression and fingers over the fish are all normal.

---

## 6. Calibration

Fitted on **validation**, never on test. `ml/evaluate.py` refuses
`--fit-calibration --split test` outright, so the rule is enforced by the tool
rather than by discipline.

Temperature is fitted by minimising NLL with a golden-section search over a
single scalar. It cannot change which class wins, so it never costs accuracy —
it only makes the number mean something.

The `unknown_threshold` is then chosen from the **measured coverage/accuracy
curve**: the lowest threshold whose accuracy-when-answered clears 90% while
still answering a reasonable share of photographs. That turns "what should the
threshold be?" into a measurement rather than a guess.

The Kotlin `Calibration.fitTemperature` mirrors the Python implementation so the
on-device and offline behaviour cannot diverge.

---

## 7. Reproducibility

* Seeds set for Python, NumPy and Torch; recorded in `run.json`.
* Split assignment is a SHA-256 of the group key, not a shuffle, so it is stable
  across runs, machines and corpus growth.
* The dataset manifest SHA-256 is recorded in `run.json` and propagated into the
  pack manifest, tying a shipped model to exact training data.
* The code commit is recorded the same way.
* Per-item augmentation RNG is seeded from `(seed, index)`, so output does not
  depend on the number of DataLoader workers.

Exact bitwise reproducibility is **not** claimed: cuDNN/MIOpen kernel selection,
AMP and multi-threaded data loading all introduce non-determinism. Runs are
reproducible in distribution, not to the last decimal.

---

## 8. Full-scale training on other hardware

The pipeline is unchanged; only the environment differs.

```bash
# CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python ml/train.py --config ml/configs/global_v1.yaml
```

On CUDA, `env.setup()` leaves cuDNN enabled and AMP on. Expect materially higher
throughput than the reference machine, both because cuDNN works and because
fp32 is not pathological there.

To scale beyond this corpus, raise `--per-species-cap` in the fetch step and
lower the evidence bar in `build-corpus`; nothing else changes.
