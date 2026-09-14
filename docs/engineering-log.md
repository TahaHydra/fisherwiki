# Engineering log

Reverse-chronological within each day. Records discoveries, decisions and
**failures**, including things that turned out to be dead ends. If a number
appears here it was measured on this machine, not quoted from a paper.

Reference hardware for all benchmarks unless stated otherwise:

> AMD Ryzen 7 7800X3D (8C/16T), 32 GB RAM, **AMD Radeon RX 7800 XT (gfx1101,
> 17.2 GB VRAM)**, Windows 11 26200, Python 3.13.15.

---

## 2026-09-13

### ROCm on Windows: MIOpen JIT-compiles BatchNorm and the compile fails

**Symptom.** Any `torchvision` model died immediately with
`RuntimeError: miopenStatusUnknownError`, while hand-written convolutions of
every shape I tried (plain, depthwise, pointwise, k5, channels_last,
forward-only, forward+backward, batch 96 @ 224px) all passed.

That contradiction is what made it findable: the problem was not convolution at
all. Dumping the MIOpen log showed the real error:

```
MIOpen: Error [BuildHip] HIPRTC status = HIPRTC_ERROR_COMPILATION (6),
        source file: MIOpenBatchNormFwdInferSpatial.cpp
  miopen_type_traits.hpp:151:10: fatal error: 'type_traits' file not found
  1 error generated when compiling for gfx1101.
```

**Root cause.** MIOpen compiles its BatchNorm kernels at runtime with HIPRTC.
Those kernels `#include <type_traits>`. On Linux that resolves against the
system libstdc++. The ROCm **Windows** pip wheels ship no C++ standard library
headers at all - I searched every installed ROCm package for `type_traits`,
`cstddef` and `cstdint` and found none, including after additionally installing
`rocm-sdk-devel`. So every MIOpen BatchNorm call fails, in both train and eval
mode. Convolution works because those kernels are pre-compiled in the device
package (`amd-torch-device-gfx1101`).

**Fix adopted.** `torch.backends.cudnn.enabled = False`, which routes both
convolution and batch-norm away from MIOpen to PyTorch's native implementations.
Measured cost is acceptable (below), so this is not worth fighting further.

**Rejected alternatives.** Supplying libc++ headers into the clang resource dir
was considered and rejected: device-side compilation of a host STL is fragile,
and it would make the build unreproducible for anyone else. Avoiding BatchNorm
architecturally was rejected because it would distort the model comparison.

**Consequence for the project.** `ml/` sets this flag centrally, and
`TRAINING.md` documents it. The flag is a *ROCm-Windows* workaround only; on
CUDA or ROCm-Linux it should be left on.

### Measured training throughput (this is what makes the project feasible)

`torch.backends.cudnn.enabled=False`, batch 96, 224x224, 500 classes,
forward+backward+optimizer step:

| Model | AMP | ms/step | img/s | peak VRAM |
|---|---|---|---|---|
| MobileNetV3-Large | off | 410 | 234 | 4.5 GB |
| **MobileNetV3-Large** | **fp16** | **257** | **374** | **2.4 GB** |
| EfficientNet-B0 | off | 573 | 168 | 8.7 GB |
| EfficientNet-B0 | fp16 | 502 | 191 | 4.5 GB |

At 374 img/s a 100k-image epoch takes ~4.5 minutes, so a 40-epoch run is ~3
hours. Full training on this machine is therefore realistic, not aspirational.

### fp32 GEMM on this stack is broken-slow; fp16 is fine

| op | time | throughput |
|---|---|---|
| 4096³ matmul fp32 | 76.6 ms | 1.80 TFLOPS |
| 4096³ matmul fp16 | 3.0 ms | **45.47 TFLOPS** |

A 25x gap is far beyond the expected 2x. fp32 is evidently not hitting tuned
kernels on gfx1101/Windows. Practical consequence: **AMP is mandatory here**,
not an optimisation. Any benchmark I report without AMP should be read as a
lower bound caused by the stack, not by the architecture.

### MIOpen cold-cache failures are transient and misleading

The very first convolution after install failed with the same
`miopenStatusUnknownError`, then succeeded on a later run with no relevant
change. Re-testing showed conv works with or without `MIOPEN_USER_DB_PATH`
once the kernel DB exists. I set `MIOPEN_USER_DB_PATH` and
`MIOPEN_CUSTOM_CACHE_DIR` to a project-local directory anyway, so the cache is
reproducible and disposable rather than hidden in a user profile. Worth knowing
that this error code covers at least two unrelated faults.

### S3 `Content-Encoding: gzip` silently corrupts resumable downloads

The iNaturalist bulk objects are stored with `Content-Encoding: gzip` metadata,
so `requests`/`urllib3` transparently gunzip the body. First symptom was a size
check failing: got 190,368,572 bytes for a file whose `Content-Length` is
39,765,976.

The size mismatch was the *lucky* failure. The dangerous one is resume: a
`Range` request offsets into the **compressed** object, but the client had been
appending **decompressed** bytes, so a resumed download would splice two
different byte streams together and produce a plausible-looking, silently
corrupt file. `download_file` now streams `resp.raw` with
`decode_content=False` so the bytes on disk are always the exact stored object,
and the SHA-256 matches what the provider published.

This is the reason the downloader verifies size *and* digest rather than
trusting a completed transfer.

### Parallel range requests: 8 MB/s -> 40-60 MB/s

Single-stream download from the iNaturalist S3 mirror ran at 5-8 MB/s. Twelve
concurrent 64 MB range requests sustain 40-60 MB/s, turning the 33 GB metadata
pull from ~75 minutes into ~12. Chunks are written individually so an
interrupted run resumes at chunk granularity. This is static object storage
designed for parallel bulk reads, so this is appropriate use, not abuse; the
per-host token bucket still applies.

### Licensing: FishNet cannot be used, and this was not obvious

The brief named FishNet first, so I checked it first. Findings:

* The project page states only that *the website* is CC BY-SA 4.0. No licence
  is stated for the dataset.
* The GitHub repository has **no LICENSE file**. Confirmed via the GitHub API:
  `license: None`.
* Images are distributed as a single Google Drive archive with no per-image
  licence metadata.
* The images derive from FishBase, where photographs are individually
  copyrighted by their contributors and the **default when no licence is
  selected is All Rights Reserved**.

So for any given FishNet image we cannot establish the licence, which is exactly
the case our own policy forbids. **FishNet is excluded from every corpus**,
including the research one - our `research_nc` policy requires a *known*
licence, not merely a non-commercial one. FishNet remains useful as a published
benchmark to compare against, and its taxonomy/trait *facts* can inform our
schema, but none of its bytes ship or train.

This is the single most consequential decision so far, and it inverts the
priority implied by the brief: **iNaturalist Open Data becomes the primary
corpus**, because it is the only large fish-image source with a
machine-verifiable per-photograph licence.

### iNaturalist fish clade roots verified against the real export

`Fish` is paraphyletic, so the root set is a product decision. Verified against
the 2026-08-27 `taxa.csv.gz`:

| clade | taxon_id | active species |
|---|---|---|
| Actinopterygii | 47178 | 35,771 |
| Chondrichthyes | 196614 | 1,331 |
| Myxini | 49099 | 89 |
| Petromyzonti | 49231 | 49 |
| Sarcopterygii | 85497 | 8 |

Total ~37,248 active species / 52,813 taxa, consistent with the ~35,000
described-species figure.

I checked `Sarcopterygii` explicitly rather than assuming: in iNaturalist's
backbone its only children are `Ceratodontiformes`, `Coelacanthiformes` and
`Lepidosireniformes`, and tetrapod classes such as `Aves` hang directly off
`Vertebrata` as siblings. Including it therefore adds lungfish and coelacanths
without dragging in birds and mammals. Had iNaturalist used a strictly
monophyletic backbone, this root would have pulled in every tetrapod.

### Multi-photo fusion: I had the combination rule exactly backwards

Expert ID fuses several photographs of one fish. I implemented an
entropy-weighted **log-space** (geometric) mean and justified it in the
docstring as being both principled under independence and robust to a single
confidently-wrong frame.

A unit test disagreed, and the test was right. Worked example, three frames,
two agreeing and one confidently wrong - the realistic failure being a frame
that caught the water, the net or the angler's hand rather than the fish:

```
frame A/B  p = [0.988, 0.007, 0.002, 0.002]   -> class 0
frame C    p = [1e-5,  1.000, 1e-5,  1e-5 ]   -> class 1, very loudly

product rule (log space)  -> [-4.16, -3.28, -8.08, -8.08]  argmax = 1   WRONG
sum rule     (arithmetic) -> [0.647, 0.350, 0.002, 0.002]  argmax = 0   correct
```

The product rule gives **any single frame a veto**: one near-zero probability
drives that class's log to about -12, and no amount of agreement elsewhere
recovers it. So the geometric mean is not merely no better here, it is
actively the fragile choice for exactly the case I claimed it protected
against. This is the classic sum-vs-product result from Kittler et al.,
*On Combining Classifiers* (PAMI 1998), which finds the sum rule markedly more
resilient to individual estimation errors.

Switched to an entropy-weighted arithmetic mean. The fused probabilities are
handed back to the ranker as `ln(p)` with temperature 1, since `softmax(ln p)`
is exactly `p` and temperature was already applied per frame.

Worth recording as a process point as much as a technical one: the plausible
mathematical story ("correct under independence") was doing the arguing, and it
was only a concrete adversarial test case that checked it.

### iNaturalist attaches one photo to several observations

`photo_id` is not unique in `photos.csv`: 6,739 photos in the 2026-08 snapshot
belong to two or more observations. Besides violating our candidate primary
key, this is a **train/test leak**: splitting is done by observation group, so
one identical image could land on both sides through two different group keys.
Deduplicated at extract time and again at select time.

### DuckDB row-at-a-time inserts cap a whole pipeline

Two separate stalls came from the same cause. Registering 308k candidate rows
with `executemany` had not finished after four minutes. Later, per-image
`mark_stored` calls capped the *download* at ~10 images/second no matter how
many threads were running, because each insert costs ~10 ms and they serialise
on one connection lock.

Measured, after moving both to Arrow columnar inserts with buffered flushes:

| operation | before | after |
|---|---|---|
| insert 308k candidates | >4 min (unfinished) | 6.6 s |
| `mark_stored` throughput | 101 rows/s | 75,042 rows/s |
| image download throughput | 10 img/s | ~90 img/s |

The lesson generalises: in a columnar store, a per-item write in a worker loop
is not a small inefficiency, it is an architectural ceiling.

### Licence cleanliness costs 88% of the available data

Of 4,168,455 licensed fish photographs in the iNaturalist export:

| licence | images | in production corpus |
|---|---|---|
| CC BY-NC | 3,356,078 | no |
| **CC BY** | **429,365** | **yes** |
| CC BY-NC-ND | 151,232 | no |
| CC BY-NC-SA | 101,234 | no |
| **CC0** | **75,982** | **yes** |
| CC BY-SA | 45,033 | opt-in only |
| CC BY-ND | 9,531 | no |

The commercially-distributable corpus is **505,347 images, 12.1%** of what is
available. That single number explains most of the project's shape: 1,963
species clear the evidence bar instead of the 15,466 that have any photograph
at all. It is worth stating plainly rather than burying, because a project that
quietly trained on the CC BY-NC pile would look four times better on paper and
could not be shipped.

### The data root was on a spinning disk, and that decided everything

Training launched at **63 images/second** against a GPU that benchmarks at 374.
The obvious suspects were both wrong: profiling the loader showed decode +
augmentation costing ~6 ms per image, while `__getitem__` took 17.8 ms.

The missing ~12 ms was seek latency. `Get-PhysicalDisk` settles it:

```
DeviceId MediaType   BusType     drive
0, 1     HDD         SATA        D:   <- 37.9 GB corpus lived here
2        SSD         NVMe        C:   (62 GB free)
3        SSD         NVMe        E:   (122 GB free)
```

307,415 small files read in random order from a 7200 rpm disk. No number of
DataLoader workers fixes that; it is the one workload spinning rust is worst at.
At 63 img/s a 30-epoch run is **32 hours**.

`tools/prepare_cache.py` does two things in one pass: re-encodes every image at
a 256 px short edge (training crops to 224, so anything larger is wasted bytes
*and* wasted JPEG decode), and writes it to the unused NVMe. It also enables
`Image.draft()`, which lets libjpeg decode at 1/2, 1/4 or 1/8 scale in the DCT
domain - free, and worth having even on fast storage.

The cache is derived data: keyed by the same SHA-256, deletable, and the dataset
falls back to the content-addressed store per image when an entry is missing, so
a partial cache still trains.

**Measured result** after the cache was built (302,693 images, 8.2 GB on NVMe,
down from 37.9 GB on the HDD):

| | images/second |
|---|---|
| corpus on HDD, full-size JPEGs | 63 |
| corpus on SSD, 256 px cache | **650** |

A 10.3x improvement, turning a 32-hour run into roughly 5 hours.

Note that 650 img/s is *above* the 374 img/s I measured for the model alone.
That is not a contradiction: epoch 0 trains with the backbone frozen
(`freeze_backbone_epochs: 1`), so there is no backward pass through it. Later
epochs settle to the model-bound rate. Worth stating so the number is not
quoted out of context as a steady-state figure.

Two things worth keeping from this:

1. **I profiled the wrong layer first.** I reached for worker counts and
   augmentation cost - the things I had written - before asking what the storage
   was. The component timings summed to a third of the measured time and I
   should have chased that gap immediately instead of tuning around it.
2. **The obvious fix was not the whole fix.** Moving to SSD addresses the seeks;
   re-encoding addresses the decode. Doing only the first would have left a 4x
   saving on the table, and doing only the second would not have helped at all.

### A note on how many of these were caught by a refusal, not a metric

Running total of bugs found in this project by something *refusing to proceed*
rather than by a number looking wrong:

| guard | caught |
|---|---|
| `verify_no_leakage` refusing to export | 3 distinct leaks (photo in two observations, stratification on the wrong grouping, byte-identical re-uploads) |
| strict CSV parsing (`ignore_errors=false`) | GBIF `nullstr` silently dropping 97% of rows |
| size + digest verification on download | S3 `Content-Encoding: gzip` corrupting resumable downloads |
| licence policy gate in `ImageStore.put` | the `License`-subclasses-`str` bug, which refused 199 of 200 valid images |
| a unit test with a concrete adversarial case | multi-photo fusion using the product rule instead of the sum rule |

None of these would have produced a visibly wrong accuracy number. Several would
have produced a *better-looking* one.

### Preprocessing parity: my resize was aliasing, and only a cross-language test found it

I had written in three docstrings that Kotlin/Python preprocessing parity was
"pinned by a test". It was not — I had not written the test. When I did, it
failed:

```
max |diff| 0.617   (normalised units, ~36 grey levels)
rms        0.033
```

The shape of the failure is what identified the cause: a large *peak*
disagreement with a small *RMS*, concentrated at high-contrast edges. That is
aliasing, not a structural error — a swapped channel or a wrong crop offset
would have produced differences of order 1.0 everywhere.

**Cause.** My `Preprocessor` did textbook 2-tap bilinear sampling. That is
correct for upscaling and wrong for downscaling: going 500 px → 256 px, a 2-tap
read skips most source pixels entirely. PIL — and therefore the training
pipeline — scales the filter support by the downscale factor so every source
pixel contributes. That is what `antialias=True` means in
`torchvision.transforms.Resize`, and it has been torchvision's default for a
while precisely because getting it wrong is so easy.

So the model would have trained on antialiased images and served on aliased
ones. Textbook train/serve skew, invisible at runtime, no error anywhere.

**Fix.** A separable triangle filter with support scaled by the downscale
factor, weights precomputed per axis, computing only the cropped output window.

| implementation | max diff | rms |
|---|---|---|
| naive 2-tap bilinear | 0.617 | 0.0331 |
| **antialiased triangle** | **0.017** | **0.0063** |

A 36× improvement in peak agreement. The test bound is now set just above the
achieved value, so any regression to an unfiltered resample fails immediately.

The fixtures are generated by the **real** training code path
(`ml/fwml/data.py` `eval_transform` + `to_tensor`), not reimplemented for the
test, so this compares shipping code against shipping code. The source image is
non-square (500×333) with an asymmetric bright marker, so a transposed axis or
an RGB/BGR swap cannot coincidentally pass.

Worth noting for its own sake: I wrote the claim before the test, and the claim
was wrong. The docstrings now describe what the test actually checks.

### ONNX Runtime does not load under the JetBrains Runtime

The end-to-end engine test failed for every case with:

```
UnsatisfiedLinkError: onnxruntime.dll: A dynamic link library (DLL)
                      initialization routine failed
```

Python's `onnxruntime` 1.30 ran the *same model file* fine, which ruled out the
model and the machine. A minimal probe isolated it to the JVM:

| JVM | `OrtEnvironment.getEnvironment()` |
|---|---|
| JetBrains Runtime 21.0.10 (bundled with Android Studio) | **fails** |
| Temurin 25.0.4 | works |

Same jar, same DLL, same machine. JBR is a patched OpenJDK and something in it
breaks the native load.

Scope matters here: this affects the **test JVM only**. The Android app uses
`onnxruntime-android` on the device runtime and is unaffected. But since JBR is
what ships with Android Studio, it is exactly the JDK a contributor is most
likely to have, so `:core/build.gradle.kts` now prefers a non-JBR Adoptium
toolchain for the test task and falls back silently when none is discoverable.

### End-to-end: the offline path is verified working

The whole chain now runs in a JVM test against a real `.fwpack`:

```
verify manifest -> hardened extract -> per-file SHA-256 -> open ONNX session
  -> preprocess -> infer -> calibrate -> geo prior -> rank -> species lookup
```

The fixture pack is genuine in every part: a real ONNX graph exported by torch,
a real SQLite database built by the shipping `SpeciesDatabaseBuilder`, a real
geo prior written by the shipping writer, real SHA-256 hashes in a real
manifest. It is 7.1 KB only because the model is deliberately tiny.

The model is hand-weighted so each class responds to one colour channel, which
is the point: the tests assert *which species comes back* for a red image versus
a blue one, not merely that nothing threw. That includes the behaviours that
matter most — that a tampered model is rejected on its hash, that the geographic
prior shifts ranking without overturning clear visual evidence, and that one
wrong frame out of three does not overturn the majority.

122 Kotlin tests, 146 Python tests.

### Store verification: all checks pass, plus one small finding

`dataset.py verify --sample 1500` over the full corpus:

```
files referenced by provenance : 310,393  (0 missing)
files on disk                  : 309,133  (0 without provenance)
hashes verified                :   1,500  (0 mismatched)
stored by licence              : CC-BY 265,249 · CC0 45,144
```

Two things worth reading closely.

**The licence gate held across 310,393 images.** Only CC-BY and CC0 appear in
the store. Nothing else ever got written, which is the property the whole
provenance design exists to guarantee.

**310,393 rows but 309,127 distinct hashes.** That is content addressing working
as intended: 1,266 provenance rows share a file with another row. One photograph
appears under **14 separate candidate ids** - the same image uploaded as
fourteen different observations. It is stored once, and the corpus builder
deduplicates by SHA-256, so it also trains once.

**The small finding:** 309,133 files for 309,127 distinct hashes. Six extra
files exist because the CAS path embeds the extension
(`<sha>.jpg` vs `<sha>.jpeg`), so identical bytes with different declared
extensions land in two files. Harmless - dedup downstream is by hash, not by
path - but it does mean the store is not perfectly content-addressed. Worth
fixing by normalising the extension from the decoded format rather than the
source's declaration; logged rather than done, since it is six files.

### INT8 is the wrong default for MobileNetV3, and fp16 is the right one

I had set the pack default to INT8 on the standard reasoning that INT8 is what
you ship to phones. Measuring it showed that is wrong here, twice over.

**First:** *dynamic* INT8 is 25x **slower** than fp32.

| variant | latency | size |
|---|---|---|
| fp32 | 2.2 ms | 14.9 MB |
| int8 dynamic | 66.2 ms | 4.0 MB |

Dynamic quantisation targets matmul-heavy models - transformers, RNNs - where
the weight matrices dominate. A depthwise-separable convnet pays a
quantise/dequantise round trip at every layer boundary and ONNX Runtime has no
fast INT8 kernel for many depthwise shapes, so it falls back. Wrong tool.

**Second:** *static* INT8 is fast, but wrecks accuracy. Measured on 1,500
held-out validation images:

| variant | top-1 | vs fp32 | agrees with fp32 | size |
|---|---|---|---|---|
| fp32 | 0.5173 | - | 100.0% | 14.9 MB |
| **fp16** | **0.5180** | **+0.0007** | **99.9%** | **7.5 MB** |
| int8 (percentile) | 0.4347 | -0.0827 | 58.8% | 4.3 MB |
| int8 (min-max) | 0.3187 | -0.1987 | 38.9% | 4.3 MB |

Nearly 20 points with default min-max calibration. Switching to percentile
calibration halves the damage, which is itself worth knowing - min-max is
outlier-driven and MobileNetV3 has layers with very wide activation ranges
because of hard-swish. But 8 points is still an unacceptable price for 3 MB in
an app whose entire value proposition is not being confidently wrong.

The "agrees with fp32" column is the one that makes this concrete: under
min-max INT8, **61% of predictions change**. That is not a quantised model, it
is a different model.

**fp16** halves the size for +0.0007 top-1 and 99.9% identical predictions.
It is now the default for both `ml/export.py` and `tools/build_pack.py`.

Caveat recorded honestly: ONNX Runtime's CPU provider has no native fp16
kernels for most ops, so on CPU this is a size win rather than a speed win. The
size is what a pack download cares about. On an ARM device with an NNAPI or GPU
delegate fp16 is typically also faster, but this machine cannot measure that.

Two process notes:

1. **I ran this against a mid-training checkpoint deliberately.** Waiting for
   the finished model to discover that the export default was wrong would have
   cost six hours. The absolute accuracy numbers above are from epoch 5 and are
   not the model's final accuracy - but the *relative* comparison between
   variants is exactly what was needed, and it is valid at any checkpoint.
2. This is the third time in this project that a widely-repeated default turned
   out to be wrong for this specific case, after the multi-photo fusion rule and
   the resampling filter. The pattern is consistent enough to be worth stating:
   the conventional choice was defensible in the abstract and wrong in the
   particular, and only a measurement distinguished the two.

---

## The headline accuracy was inflated by photographer overlap

*2026-09-14, after the release run*

While writing `MODEL.md` I claimed, in the limitations section, that a
"257-image unseen-photographer subset" was the nearest available proxy for
angler-style photographs. Then I went to check the number and discovered I had
never computed it. The `unseen_observer` flag existed in the manifest — the
splitter has written it since the leakage work — but nothing downstream ever
read it.

This is the second time in this project I have written a documentation sentence
describing a measurement that did not exist, after the Kotlin/Python
preprocessing parity test. Both times the sentence was *plausible*: the
mechanism existed, the number was clearly computable, and describing it felt
like description rather than invention. The failure mode is not laziness, it is
that documentation prose runs ahead of verification without any signal that it
has.

### Making the join sound first

Computing it needed care. `collate_drop_failures` removes images that fail to
decode, so the model returns fewer predictions than the split has rows, and
position *i* in the output is not row *i* in the manifest. Zipping the two to
attribute predictions to observers would be correct only while nothing fails to
decode — and would be silently wrong otherwise, because the shapes still
broadcast and the numbers still look plausible.

In this run zero images failed, so the naive version would have produced the
right answer today and a subtly wrong one later. I added `emit_index` to
`FishDataset` and `collate_drop_failures_indexed`, so predictions carry their
manifest row index, and pinned it with `tests/test_eval_alignment.py`.

Those tests need torch, which lives in `.venv-train`, which had no pytest — so
`pytest.importorskip("torch")` would have made them skip forever while
displaying as a green suite. Installed pytest into the training environment;
both suites are now listed in `MODEL.md` because running only the data one
under-reports.

### The result

| | n | top-1 |
|---|---|---|
| photographer also in train | 28,686 | 0.5356 |
| photographer never seen | 257 | **0.3113** |

A 22-point gap, 95% CI [0.255, 0.368].

The split had no *observation* leakage — that was fixed carefully and verified
by a guard that refuses to export a leaking split. But observation grouping does
not stop a photographer with 200 observations from having some in train and some
in test, and the model evidently learned camera, lighting, background and
handling style along with the fish.

I suspected the subset was just harder — that one-off contributors photograph
rarer species. Measured: they do not. Median training support is 233 for the
unseen-observer subset against 192 for the rest, and it has *fewer* images in
the sparse 0–60 bucket. The subset is slightly easier by that measure, so the
gap is if anything understated.

### What it does to the product claims

At the shipped 0.80 threshold, on unseen photographers: species precision 71.8%
rather than 93.0%, useful answers 24.5% rather than 46.7%, wrong answers 11.3%
rather than 7.2%, silence 64.2% rather than 46.0%.

The precision figure is the one that stings, because §4 of `MODEL.md` uses 93%
to justify the threshold choice and that justification is weaker than it looked.

But the shape of the degradation is the thing the whole uncertainty stack was
built for, and it held: faced with inputs it genuinely could not handle, the
model got quieter rather than confidently wrong. Silence absorbed most of the
lost accuracy — the wrong-answer rate rose 4 points while accuracy fell 22.
Confidence dropped too (0.404 against 0.535), which is why. If temperature
scaling and the three-signal rejection had not been there, those 22 points would
have surfaced as false identifications.

### Consequences

- `MODEL.md` §5 now leads with this, and the headline number carries a warning.
- The fix is more *photographers* per species, not more images per species —
  which inverts the conclusion from the accuracy-vs-support table, where images
  per class looked like the binding constraint. Both are true; they are
  different axes and I had only been measuring one.
- The right split for the *next* corpus build is probably observer-disjoint by
  construction. The earlier attempt at that produced 284 classes with no
  validation images, which is why it was abandoned for observation grouping. The
  real answer is likely a hybrid: observer-disjoint where a class has enough
  distinct observers to allow it, observation-grouped where it does not, and the
  evaluation reporting the two populations separately rather than pooling them
  into one flattering average.

---

## The safety warning was attached to the wrong thing

*2026-09-14*

Reading the confusion table to write `MODEL.md`, one row did not belong with the
others:

    Trachinus draco  ->  Mullus barbatus   8  (30.8%)

Every other frequent confusion was within genus and harmless — two mantas, two
*Dascyllus*, two *Heniochus*. This one is a weeverfish, which has venomous
dorsal and opercular spines and buries itself in sand, being called a red
mullet, which is dinner.

I went to check that the app handled it and found it did not. `SafetyCard` was
driven by `detail.safetyWarnings`, and `detail` was fetched for
`identification.best` alone. When the model is wrong in the direction of
"harmless", the user sees no warning at all. Every safety test in the suite
passed, because every one of them asked "does a venomous species show its
warning?" — never "does a harmless *prediction* hide a venomous possibility?"

### The obvious fix, and why it was not enough

Collect warnings across all candidates, not just the winner. That is right, and
it is now `hazardsForCandidates`, deduplicated on (kind, summary) so three
congeners from one venomous family do not produce three identical cards.

Then I measured whether it actually covers the case. Of the 26 test images of
*Trachinus draco*, 19 were misidentified; of those, the weever remained in the
top five in only **6 — 32%**. The worst case was *Mullus barbatus* at 0.974
confidence with *Trachinus* at rank 14, probability 0.0002.

So the candidate-scan fix covers about a third of the exposure. Shipping it
alone and calling the problem solved would have been the kind of half-measure
that is worse than none, because it looks like coverage.

### What actually covers it

A static cross-reference, built from the same confusion matrix that revealed
the problem. `similar_species` had been in the schema since the beginning —
including a `confusion_rate REAL, -- measured on our held-out test set` column
— and had never had a single row written to it.

`ml/evaluate.py` now exports the full matrix (1,275 pairs seen twice or more,
against 30 in the human-readable report), and `tools/fwdata/confusion.py` turns
it into 2,348 rows. 53 pairs cross a safety boundary. 49 species that carry no
danger warning of their own now reach one.

Three decisions inside that worth recording:

1. **Asymmetric admission floor.** Pairs normally need ≥2 occurrences and a ≥2%
   rate. When exactly one side carries a `danger` warning, one occurrence is
   enough. The costs of the two errors are not comparable, so the thresholds
   should not be either.
2. **Nothing asserts morphology.** `difference` is `NOT NULL`, and the
   temptation was to write something plausible about fin spines. It holds an
   explicit statement that no distinguishing feature is recorded, and the rows
   cite a source whose own citation text says it describes model behaviour
   rather than biology. A reader can exclude one source id and be left with
   only biological claims.
3. **Both directions, keyed for the user.** The row that protects someone is
   the one keyed on the species the app *displays*, since that is the name on
   screen when they reach for the fish.

### A bug the tests caught

Look-alike hazards inherit the *displayed* species' candidate rank, which is 0.
Sorting by (severity, rank) therefore ranked a static look-alike above a species
the model had actually proposed, and the deduplication then discarded the
stronger claim. Two tests failed on exactly this. Fixed with an explicit
`reason` term in the sort: a proposed candidate always beats a static
cross-reference carrying the same warning.

Two further tests failed because extending the fixture invalidated their
premise — one asserted a species had no hazards when it now had a dangerous
look-alike, another counted `similar_species` rows. Both were rewritten to
assert the property rather than a count, which is what they should have done
originally.

### The general point

The bug was not in the safety code. The safety code was careful: every warning
sourced, severity-ordered, family-rank expansion, a test asserting that absence
of a warning is not a claim of safety. It was correct about the species it was
given, and the species it was given was the one the model guessed.

For a system whose central claim is that it is often uncertain, attaching
safety-critical information to the top-1 prediction was the wrong shape from the
start. The uncertainty was modelled everywhere in the ranking, the calibration
and the UI copy — and then quietly discarded at the one point where being wrong
has physical consequences.

Worth noting how it was found: not by a test, not by review, but by reading a
metrics table for a documentation file and noticing one row that did not look
like its neighbours.

---

## The inference library was asking for internet access

*2026-09-14*

While checking that my changes to the result screen had not disturbed anything,
I grepped the app for network code and found none: no HTTP client, no
networking dependency, nothing. The manifest nonetheless requested `INTERNET`,
with a comment saying it was "used ONLY for pack downloads".

There are no pack downloads. `PackManager` installs from a `File` or a content
`Uri`; the downloader was never built and there is no server to download from.
`PRIVACY.md`, `SECURITY.md` and `README.md` all described a network feature
that does not exist — `SECURITY.md` went as far as "all pack URLs are HTTPS",
which is true only vacuously.

Removing the permission is strictly better than documenting it: an app without
`INTERNET` cannot open a socket, and the platform enforces that rather than the
author promising it. So I removed both network permissions, rewrote the four
documents to say packs are installed from a file, and stated the missing
downloader as a gap.

### Then I checked the APK

The permissions were still there.

`onnxruntime-android` declares `INTERNET` and `ACCESS_NETWORK_STATE` in its own
manifest, and the manifest merger unions dependency manifests into the app's.
The inference library — the single component with the least business talking to
a server — grants network access to every application that links it.

So for the whole life of this project the shipped APK has had internet
permission, while the source manifest and the privacy documentation said
otherwise. Both were written in good faith and both were wrong, because they
described the input to a build step rather than its output.

Fixed with explicit `tools:node="remove"` entries. Verified against the built
artefact: 6 permissions, none of them network.

### The check that follows from it

`scripts/check_apk_permissions.py` reads the permission set out of the binary
manifest inside an APK and fails on anything not on an explicit allow-list —
so a future dependency reintroducing one is a build failure rather than a
discovery. It rejects unknown permissions rather than only known-bad ones,
because the failure mode here was precisely something nobody had thought to
deny.

It parses AXML by hand, which is the kind of code that can silently return
nothing and look like a pass, so: it fails explicitly when it parses zero
permissions, and `tests/test_apk_permissions.py` cross-checks its output
against Gradle's own merged manifest XML. They agree exactly.

### What I take from this

Three of the four things in this log that turned out to be wrong were wrong in
the same way: a claim about the *system* verified against the *source*. The
Kotlin resampling parity test that did not exist, the taxonomy tool three
documents referenced, the unseen-photographer number I described before
computing — and now a permission set that was correct in the file I wrote and
incorrect in the file users would install.

The rule that would have caught all four: **verify the artefact, not the
intention.** Read the APK, not the manifest. Run the test, do not describe it.
Compute the metric before writing the sentence that cites it.

The privacy claim is also now stronger than it was when I believed it. It went
from an assurance about what the code does to a property of the package that a
sceptical reader can confirm in ten seconds without trusting me at all — which
is what a privacy claim in an open-source app should be.
