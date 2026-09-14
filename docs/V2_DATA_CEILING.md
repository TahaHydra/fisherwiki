# The V2 dataset ceiling

The V2 plan assumed a target of 2–3 million images. Measured against the actual
sources, that target is not reachable under this project's licence policy. This
document records the numbers, because the target shapes every other decision —
model size, epoch count, cloud budget — and should not be carried forward on an
assumption.

---

## 1. iNaturalist is the ceiling, and it is 505,347

Measured directly from the extracted candidate pool
(`work/candidates_inaturalist.parquet`, 4,168,455 rows):

| licence | images | admissible under `production`? |
|---|---:|---|
| CC-BY-NC-4.0 | 3,356,078 | no |
| **CC-BY-4.0** | **429,365** | **yes** |
| CC-BY-NC-ND-4.0 | 151,232 | no |
| CC-BY-NC-SA-4.0 | 101,234 | no |
| **CC0-1.0** | **75,982** | **yes** |
| CC-BY-SA-4.0 | 45,033 | no (`production_sa` only) |
| CC-BY-ND-4.0 | 9,531 | no |

**Admissible total: 505,347 images over 8,115 species.** That is not a cap we
chose — it is every CC0/PD/CC-BY fish photograph iNaturalist has.

80.5% of iNaturalist's fish photography is non-commercial-licensed and therefore
unusable for a product that redistributes a trained model commercially. This was
already documented in [`DATASET_LICENSES.md`](DATASET_LICENSES.md); what is new
here is the consequence for the V2 target.

### Depth, not just breadth

| | species |
|---|---:|
| in the admissible pool | 8,115 |
| with ≥ 25 admissible images | 2,775 |
| with ≥ 40 admissible images | 2,190 |
| with ≥ 100 admissible images | 1,195 |
| with ≥ 300 admissible images | 382 |

Taking every admissible image with no per-species cap yields 505,347. A cap of
300 yields 369,617; a cap of 1,000 yields 472,503. The curve is flat past ~600
because only 382 species have that many images at all.

---

## 2. What other sources realistically add

| source | scale | licence reality |
|---|---|---|
| **Wikimedia Commons** | tens of thousands of fish images | mostly CC-BY-SA — `production_sa`, not `production` |
| **FathomNet** | ~84k images / 176k localizations (2022) | **CC-BY-NC-ND**, with an explicit data-use grant for ML training including commercial. Deep-sea ROV imagery — mostly invertebrates and bathyal fish, little overlap with what an angler catches. |
| **GBIF media** | large | largely re-aggregates iNaturalist and museum specimens; specimen photographs are a different domain from live fish |
| **FishNet** | ~100k | **excluded** — no licence at all. See `DATASET_LICENSES.md`; this remains excluded. |

FathomNet's position deserves a decision rather than an assumption: the images
are CC-BY-NC-ND, but the published data-use policy explicitly permits use for
"training and development of machine learning algorithms for commercial …
purposes". Training on them while never redistributing them is plausibly within
that grant — packs ship `ATTRIBUTIONS.csv`, not pixels. That is a legal judgment,
not an engineering one, and this project excluded FishNet over a weaker version
of the same question.

**Realistic all-in ceiling under the current `production` policy: roughly
550,000–650,000 images.** Four to five times short of the 2–3M target.

---

## 3. The options

1. **Accept ~500–650k.** Roughly double the V1 corpus, with the crop pipeline
   worth more than the extra images anyway (+18.9 points measured, versus a
   doubling of data which historically buys far less). This needs no policy
   change and no legal review.
2. **Relax to `production_sa`** (add CC-BY-SA). +45,033 images from iNaturalist,
   plus most of Wikimedia Commons. Share-alike obligations attach to derivative
   works, which for a trained model is an unsettled question — worth a decision,
   not a default.
3. **Relax to include CC-BY-NC** (`research_nc`). This is the only route to
   millions: +3.36M images from iNaturalist alone. It forecloses commercial
   redistribution of the model, which is a product decision, and it would make
   the resulting weights unshippable under the project's current terms.
4. **Add FathomNet under its ML-training grant.** ~30k relevant fish images,
   pending the legal read above.

Options 1 and 4 preserve the project's licence posture. Options 2 and 3 change
what FisherWiki can be.

---

## 4. What this means for the training plan

Nothing about the V2 *architecture* changes. What changes is scale:

* At ~600k images and ~86 img/s blended, an epoch is **under two hours** locally,
  not eleven. A 15-epoch run is a couple of nights, not a month.
* The cloud multi-GPU plan becomes largely unnecessary for the first V2 model.
  It stays worth keeping for a later corpus, and the code is already written and
  tested, but renting 8×4090 to train on 600k images would be spending money to
  wait less.
* The measured bottleneck stays where it was: **photographers, not photographs.**
  V1's unseen-photographer accuracy (0.3113 against 0.5356) is the number that
  most needs moving, and it responds to *diversity* of contributors rather than
  volume. 2,775 species with ≥25 admissible images is the honest scope of a
  species list for V2.
