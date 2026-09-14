# Data provenance

Every image in this project is traceable to the record it came from, the person
who took it and the licence it carries. This document describes how that is
stored, how it is enforced, and how to reproduce or audit it.

The licensing *analysis* is in [`DATASET_LICENSES.md`](DATASET_LICENSES.md).
This is the mechanism.

---

## 1. The rule

> **There is exactly one way to put an image on disk: `ImageStore.put()`, and it
> requires a fully-populated provenance record whose licence the active policy
> admits.**

There is no helper anywhere in the codebase that writes image bytes without
provenance, and the check happens *before* the filesystem is touched. This is
enforced by API surface rather than by developer discipline, because discipline
is exactly what fails at 2 a.m. six months from now.

Two consequences worth stating:

* A file in the content-addressed store with no provenance row is an **error**,
  not a warning — `dataset.py verify` reports it as such. An image with no
  provenance row is an image with no recorded licence.
* Changing the licence policy re-selects rows but never re-downloads pixels,
  because storage is content-addressed.

---

## 2. What is recorded

One row per candidate photograph, written **before** anything is downloaded:

| field | meaning |
|---|---|
| `candidate_id` | `<source>:<record_id>`, primary key |
| `source_dataset` | `inaturalist`, `inaturalist-nonfish`, ... |
| `source_record_id` | the provider's own id (iNaturalist `photo_id`) |
| `image_url` | the exact URL the bytes came from |
| `source_url` | human-facing page for the record |
| `source_taxon_id` | taxon id **as asserted by the source** |
| `original_scientific_name` | name as asserted by the source |
| `accepted_scientific_name` | after taxonomy reconciliation |
| `taxon_id` | our canonical `fw_taxon_id` |
| `license` | normalised, from the closed vocabulary |
| `license_raw` | **exactly** as the source gave it |
| `license_url` | canonical licence URL |
| `creator`, `copyright_holder` | photographer |
| `attribution` | rendered attribution string |
| `group_key` | observation UUID — the leakage-safe split unit |
| `observer_key` | photographer id — the second grouping level |
| `latitude`, `longitude`, `positional_accuracy` | as published |
| `observed_on`, `quality_grade` | |
| `position_in_observation` | 0 = the first photo of a sighting |
| `declared_width`, `declared_height`, `ext` | as the source states them |
| `context_tag`, `notes` | e.g. `negative:Amphibia`, anomaly score |
| `discovered_at` | when we first considered it |

Then, once bytes exist, a second row records what only measurement can know:

| field | meaning |
|---|---|
| `sha256` | of the exact bytes; also the storage path |
| `bytes`, `width`, `height` | measured, not declared |
| `phash`, `dhash` | perceptual hashes for dedupe and leakage checks |
| `cas_path` | location in the content-addressed store |
| `download_timestamp` | |

`license` and `license_raw` are both kept deliberately. The normalised enum
collapses CC version numbers (`CC BY 2.0` → `CC-BY-4.0`) because policy does not
depend on the version, but attribution does, so the original string is
preserved.

---

## 3. Storage layout

```
$FISHERWIKI_DATA/
  raw/                    provider bulk dumps, exactly as published
    inaturalist/          observations, photos, taxa, observers (.csv.gz)
    gbif/                 backbone-simple-2023-08-28.txt.gz
  cas/                    content-addressed images
    ab/cd/abcd…ef.jpg
  provenance.duckdb       candidates, stored, failures, corpus_members, flags
  work/                   intermediates: candidate parquet, taxonomy, caches
  artifacts/              corpus manifests, training runs, built packs
```

Content addressing (`cas/<sha[0:2]>/<sha[2:4]>/<sha>.<ext>`) means the same
photograph obtained twice is stored once, a corpus is just a list of hashes, and
deleting a corpus never deletes pixels.

---

## 4. Failures are recorded too

`fetch_failures` holds why a candidate never became an image, classified as:

* `permanent` — 403/404/410, or undecodable. Never retried; a re-run skips it.
* `transient` — network or filesystem trouble. Retried on the next run.

This matters for honesty about coverage: 803 of 308,227 selected photographs had
been deleted at the source by the time we fetched them. That is a documented gap
rather than a silent one, and `verify` can explain any discrepancy between what
was selected and what exists.

The distinction was worth getting right. An earlier version counted filesystem
races as licence refusals, which made 9 disk errors look like 9 licence
violations — the wrong alarm entirely.

---

## 5. Quality facts, recorded not enforced

Every stored image carries measured facts: dimensions, perceptual hashes,
Laplacian-variance blur score, clipped-pixel fraction, saturation spread, and
any flags raised (`near_greyscale`, `very_dark`, `too_small`, `extreme_aspect`,
`heavily_clipped`, `very_bright`).

These are **descriptive**. Thresholds are applied later, per corpus, because the
right threshold is a product decision. A dark, blurry photo of a fish in a
landing net at dusk is a valuable training example, not garbage.

Only genuinely unusable images are hard-rejected: undecodable, zero-sized, or
single-colour.

---

## 6. Generated artefacts

```bash
.venv/Scripts/python tools/dataset.py attributions --corpus global_v1
```

produces:

* **`ATTRIBUTIONS.csv`** — one row per image: id, taxon, source, record id,
  source URL, image URL, creator, licence, licence URL, attribution string.
  This ships **inside every pack**, so a released model carries the credits for
  the photographs that made it.
* **`ATTRIBUTIONS.summary.json`** — image counts per licence.

---

## 7. Auditing

```bash
.venv/Scripts/python tools/dataset.py verify --sample 5000
```

checks four things:

1. every provenance row's file exists and matches its recorded size;
2. **no file in the store lacks a provenance row** (treated as an error);
3. sampled files still hash to their own filename — the path *is* the hash, so
   this detects bit-rot and tampering alike;
4. no stored image carries an inadmissible licence, optionally against a
   specific policy.

```bash
.venv/Scripts/python tools/dataset.py dedupe --threshold 6
```

reports exact duplicates, near-duplicates within a class, near-duplicates
*across* classes (label-quality suspects rather than duplicates), and
cross-split leakage.

---

## 8. Reproducing a corpus exactly

A corpus is reproducible from:

* the pinned provider snapshots (iNaturalist 2026-08-27, GBIF 2023-08-28 with
  its SHA-256 recorded),
* the committed taxon registry (`data/taxonomy/taxon_registry.tsv`),
* the licence policy name,
* the split salt (the corpus name),
* the code commit.

Split assignment is a hash of the group key rather than a shuffle, so it is at
least **stable across runs and machines**: rebuilding the same corpus from the
same data always reproduces the same split, deterministically (pinned by
`test_rebuilding_gives_the_same_split`).

**Stability under corpus growth is weaker than that, and this document
previously overstated it.** The OBSERVER strategy genuinely is immune to
growth: it assigns each group a fixed hash fraction, independent of every
other group, so adding data can only ever change the *new* groups' splits.
The OBSERVATION strategy - the one actually used to build `global_v1`, chosen
because the observer-independent version left 284 classes with no validation
images at all - instead *ranks* each class's groups by hash and keeps the
top `ceil(n × 0.8)` as train, where `n` is that class's current group count.
Adding a group to a class can shift both an existing group's rank among its
now-larger class and the boundary itself, so an existing observation's split
*can* change. `test_growth_can_move_an_existing_observation` demonstrates this
directly against the real corpus builder, not just in theory.

Practically: a corpus rebuild after adding data is not guaranteed to leave
every previous train/val/test assignment untouched for classes built with the
OBSERVATION strategy. "Never tune against the test set" is still enforced
*within one corpus build* - the held-out test split is read once per run,
which is the actual mechanism the promise depends on - but is not yet a
promise that survives rebuilding the corpus with more data. See `task.md` for
the two ways to close this gap: version corpus builds so an old test split can
be pinned and re-evaluated exactly, or make split assignment for small classes
growth-stable by some other means without reintroducing the 284-classes
problem. **This gap is closed for V2 — see §10**, which keeps V1 exactly as
described above so its reported numbers stay reproducible.

Every training run records the corpus manifest's SHA-256 and the code commit in
`run.json`, and both are propagated into the pack manifest — so a model in the
field can always be traced back to the exact data and code that produced it.

---

## 9. Removing an image

Because everything is keyed by SHA-256, a takedown request, a licence change or
a mislabelled image is handled precisely:

```sql
-- find it
SELECT * FROM provenance WHERE source_record_id = '12345678';
-- or by content
SELECT * FROM provenance WHERE sha256 = 'abcd…';
```

Delete the CAS file and the `stored` row, leave the `candidates` row with a
`fetch_failures` entry recording why, and rebuild the corpus. The candidate row
is kept deliberately: the record of *what was considered and then excluded* is
part of the provenance, not noise.

---

## 10. V2 splits: immutable, persisted, four-way

Section 8 describes a gap: V1's OBSERVATION strategy re-derives splits on every
build and can move an existing observation when the corpus grows. V2 closes it,
and is implemented separately in [`tools/fwdata/splits_v2.py`](../tools/fwdata/splits_v2.py)
so that V1 stays byte-for-byte reproducible.

### The four splits

| split | used for |
|---|---|
| `train` | what the model fits |
| `validation` | model selection, early stopping, temperature/calibration fitting |
| `dev_test` | **all** product development: confusion analysis, `similar_species`, safety cross-references, threshold tuning, geo-prior evaluation |
| `final_test` | sealed. Release evaluation only, through one recorded command |

`dev_test` exists because of a measured failure, not a theory. V1 claimed the
test split was "read exactly once", and that stopped being true - not by
carelessness, but because legitimate product work (the *Trachinus draco*
confusion, `class_metrics_test.json` feeding `similar_species`) had nowhere
else to happen. Giving that work its own split is what makes the sealed one
credible.

### Where assignment lives

Assignments are **stored, not derived** — in the same `provenance.duckdb`,
alongside V1, scoped by `dataset_version` so the two cannot be confused:

```
split_groups         (dataset_version, group_id) -> split, taxon, observer,
                     rule, batch, assigned_at        -- the decision
split_group_members  (dataset_version, sha256) -> group_id
                                                    -- image -> leak group
split_quarantine     images refused because accepting them would have leaked
final_test_access    one row per unseal, forever
```

V1's `corpus_members` is untouched and is not read by V2. A V1 split is a record
of how that model was built; it is not a claim about where an image belongs now,
so it is deliberately **not** imported
(`test_v1_corpus_members_are_neither_read_nor_written`).

### Why growth cannot move an assignment

Not because the hash is cleverer. Because `assign` only ever **inserts rows for
groups it has never seen** and never issues an `UPDATE` against an existing one.
That makes the guarantee structural rather than a property of the arithmetic,
and it is what lets the assignment rule consult current state (for coverage)
without endangering anything.

First assignment of a new group uses a fixed-band hash, not a rank-and-cut:

```
fraction = sha256(f"v2:{taxon_id}" + "\0" + group_id)  ->  [0, 1)
train 0.00-0.70 | validation 0.70-0.80 | dev_test 0.80-0.90 | final_test 0.90-1.00
```

The per-class salt preserves the stratification V1 needed (every class spread
across all four splits) while removing the thing that made V1 unstable: the
class's current group count is not an input, so no boundary moves when it
changes.

### Leakage grouping hierarchy

The unit that must never span two splits is the **leak group** — a connected
component over these edges, strongest first:

1. **Exact bytes.** Membership is keyed by `sha256`, so the same photograph under
   two candidate ids is one node *by construction*. 1,102 hashes in the current
   store arrive under more than one candidate id; V1 tie-broke a winner and the
   losers went on to contaminate the open-set "unseen species" pool.
2. **Near duplicates** — `dhash` Hamming ≤ 6, computed **per taxon** with the
   exact search in [`tools/fwdata/dedupe.py`](../tools/fwdata/dedupe.py).

   "Exact" is load-bearing and was not true at first. The original index banded
   the 64-bit hash into four 16-bit bands and required one band to match
   exactly. By pigeonhole that finds every pair only when the threshold is
   *below* the band count — its own comment said so — and the threshold is 6
   against 4 bands. Two hashes differing by one bit in each band are 4 apart and
   share no band. On the V1 CAS that dropped **451 of 2,960 genuine same-taxon
   pairs, 15%**, each one a chance for the same photograph to sit in train and
   in the sealed holdout. It now uses blocked brute force below 8,192 hashes and
   multi-index hashing above, both verified against brute force.

   Searching per taxon rather than globally makes cross-class unions impossible
   by construction instead of computed and discarded. That restriction matters:
   globally 210,381 pairs clear the threshold and only ~3,000 are same-class,
   the rest coming from a few thousand images whose dhash carries almost no
   information — dark or uniform frames, 29 hashing to all zeroes — which
   collide with everything. `dataset.py dedupe` still reports those, as the
   hash-quality signal they are rather than as label disputes.
3. **Observation / media group** — every image of one `group_key`.
4. **Photographer** — deliberately **not** a union edge. Making it one is V1's
   OBSERVER strategy, measured and rejected: ~5,500 photographers over 1,978
   classes left 284 classes with no validation images. Instead a new group
   *inherits* the split any of its photographers already holds within that
   class — **unconditionally**.

   It was briefly gated on the class having ≥ 8 photographers, which growth
   breaks: a class assigned while it had 5 got no inheritance, so one
   photographer's groups landed on both sides of the boundary, and when the
   class later reached 8 the rule switched on but immutability meant those
   assignments could never be repaired. The gate also made the outcome depend on
   the *current* photographer count — the coupling V2 exists to remove. Thin
   classes are protected by `MIN_GROUPS_FOR_HOLDOUT` and the coverage top-up
   instead, which is where that protection belongs.

   Because inheritance ties a photographer's groups together, the coverage
   top-up moves a whole **photographer** into an empty split, never a single
   group — moving one group of a photographer who has others would buy coverage
   with exactly the leak the rule prevents. `split_groups.rule` records which
   rule decided each row.

   One more thing the multi-source future needs: `group_key` and `observer_key`
   are **namespaced by source** at this layer. They are raw provider ids —
   iNaturalist's `observer_key` is a bare integer, 308,227 of 308,227 of them —
   so a second source reusing an id would otherwise be read as the same
   photographer, or merge two species into one leak group.

When growth bridges two already-assigned groups sitting in different splits
(a new image near-duplicating both), neither existing assignment moves and the
new image is **quarantined** — recorded in `split_quarantine` with the groups it
bridged, excluded from every manifest, and counted in the report. Silently
dropping it would be how the corpus rots.

### How final_test is sealed

Structurally, not by convention:

* `v2-export` writes `manifest.parquet` containing **only** train/validation/
  dev_test. Training and evaluation load that file, so they *cannot* read
  final_test — the rows are not there.
* final_test goes to `sealed/final_test.parquet` next to a `SEALED.md`.
* `ml/evaluate.py --split final_test` refuses with an explanation and a pointer
  to `dev_test`.
* Unsealing requires `v2-release-eval --i-am-releasing` (or
  `FISHERWIKI_RELEASE_EVAL=1`) **and** a `--reason`, and appends a row to
  `final_test_access` with the timestamp, reason, git commit and row count.
  "It was read once" becomes a query, not a memory:

```sql
SELECT * FROM final_test_access ORDER BY accessed_at;
```

### Migration from V1, and the exact commands

V1 assets are read-only inputs throughout; nothing below writes to
`artifacts/global_v1`, `artifacts/runs/*` or `corpus_members`. The V1 CAS is
reused as acquisition batch zero — full-quality originals, not the derived
`E:\fisherwiki-cache`, which is a decoded/resized training cache and never a
source.

```bash
# 0. one-time safety copy (already done if you followed the V2 bring-up)
#    D:\fisherwiki-data\v2\snapshots\provenance_v1.duckdb

# 1. assign V2 splits to every eligible image already in the CAS.
#    Safe to interrupt and re-run; it only ever adds rows.
.venv/Scripts/python tools/dataset.py v2-assign --batch batch0_v1_cas

# 2. structural checks (no group or hash spans two splits, no orphans)
.venv/Scripts/python tools/dataset.py v2-verify

# 3. see what you got, including the assignment-rule mix
.venv/Scripts/python tools/dataset.py v2-status

# 4. write the open manifest + the sealed final_test manifest
.venv/Scripts/python tools/dataset.py v2-export --corpus global_v2
```

Every later acquisition batch is the same one command, which assigns only what
is new and cannot disturb what exists:

```bash
.venv/Scripts/python tools/dataset.py v2-assign --batch batch1_<source>
.venv/Scripts/python tools/dataset.py v2-export --corpus global_v2
```

And, once, at release:

```bash
.venv/Scripts/python tools/dataset.py v2-release-eval --corpus global_v2 \
    --reason "v2.0 release evaluation" --i-am-releasing
```

`v2-assign` prints a fingerprint — a SHA-256 over every `(group_id, split)`
pair. Recording it in a training run is how a later rebuild proves it changed
nothing.

### What batch zero actually produces

Measured by running the real assignment against copies of the live provenance
database (about 6 s end to end, so this is cheap to re-run), at both candidate
ratios:

| | 70/10/10/10 | 80/10/5/5 |
|---|---|---|
| eligible images | 305,569 | 305,569 |
| train images | 207,731 | **235,934** |
| validation / dev_test / final_test | 32,915 / 32,611 / 32,312 | 33,180 / 18,535 / 17,920 |
| photographers in final_test | 2,266 | 1,660 |
| classes clearing the corpus bar | 1,852 | **1,899** |
| …of those, missing validation / dev_test / final_test | 10 / 11 / 15 | 8 / 11 / 14 |
| …of those, missing **train** | 0 | 0 |
| photographer/class pairs spanning splits | 0 | 0 |
| quarantined | 0 | 0 |
| leakage (groups / hashes / orphans) | 0 / 0 / 0 | 0 / 0 / 0 |

**Recommended freeze for a 2–3M corpus: 80/10/5/5.** It is better on every axis
measured here — 13.6% more training images, *more* classes clearing the bar
(because the bar counts train images), equal or better holdout coverage, and
identical leakage guarantees. At 2–3M images a 5% release holdout is still
100,000–150,000 images, far more than a stable release estimate needs; spending
10% there buys precision nobody reads and costs training data that shows up in
accuracy. The default is left at 70/10/10/10 until you confirm, because the
ratio is frozen for the life of the corpus once the first batch is assigned.

Holdouts run slightly rich against target (validation 10.9% at a 10% setting)
because the top-up moves whole photographers out of train. That is the intended
trade: a class that cannot be measured is worse than a split a fraction of a
point off nominal.

**One deliberate difference from V1, worth a decision rather than a default.**
V2's class bar counts **train** images (`--min-images`, default 40) rather than
all images, so that whether a class is learnable does not depend on how much
holdout it happened to receive. That is stricter than it looks: 40 train images
is roughly 57 total, and it admits **1,654** classes where V1's bar admitted
1,977. Measured on batch zero:

| `--min-images` (train) | classes |
|---|---|
| 40 (default) | 1,654 |
| 30 | 1,871 |
| 25 | 1,963 |
| V1's rule (40 *total*) | 1,976 |

`--min-images 25` reproduces V1's coverage under V2's cleaner definition. Which
to ship is a product call about how thin a class may be and still earn a
species-level claim, so it is a flag rather than a silent default.
