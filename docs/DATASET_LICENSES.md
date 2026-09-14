# Dataset and media licensing

This document records **what we verified**, **when**, and **what we decided**.
It is the authority for which images may enter which corpus. Where a licence
could not be established, that is stated as a finding rather than smoothed over.

All checks below were performed on **2026-09-13** against live sources.

> **Not legal advice.** This is an engineering record of licence metadata and
> the policy we derived from it. Anyone shipping this commercially should have
> a lawyer review the conclusions, particularly on §5 (ShareAlike) and §6
> (training as a derivative use).

---

## 1. The rule the pipeline enforces

Three corpus policies exist, defined in [`tools/fwdata/licenses.py`](../tools/fwdata/licenses.py):

| Policy | Admits | Commercial-safe | Purpose |
|---|---|---|---|
| `production` *(default)* | CC0, Public Domain, CC BY | yes | Any model shipped in a release pack |
| `production_sa` | + CC BY-SA | yes, with obligations | Opt-in only; see §5 |
| `research_nc` | + CC BY-NC, CC BY-NC-SA | **no** | Ablations only; never shipped |

Enforced properties:

* **Empty means All Rights Reserved, not public domain.** On iNaturalist an
  empty licence column is ARR. This is the single most dangerous mis-parse and
  is covered by a test.
* **Unparseable means `UNKNOWN`, and `UNKNOWN` is admitted by nothing.**
  `CC-BY-MAYBE` does not resolve to CC BY; it resolves to `UNKNOWN`.
* **NoDerivatives media never enters any corpus**, including the research one
  (see §6).
* `ImageStore.put()` is the only way to write image bytes, and it refuses a
  record whose licence the active policy does not admit — the check happens
  before the filesystem is touched.
* `assert_no_forbidden_licenses()` re-checks a built corpus before training.

---

## 2. iNaturalist Open Data — **primary corpus**

| | |
|---|---|
| Source | `s3://inaturalist-open-data` (us-east-1), HTTPS mirror `https://inaturalist-open-data.s3.amazonaws.com/` |
| Registry | <https://registry.opendata.aws/inaturalist-open-data/> |
| Docs | <https://github.com/inaturalist/inaturalist-open-data> |
| Snapshot used | **2026-08-27** (monthly regeneration) |
| Licence granularity | **Per photograph**, in `photos.csv.gz` |
| Verdict | **Accepted.** Licence is machine-verifiable per image. |

This is the only large fish-image source we found where the licence of each
individual photograph is published as structured data. That single property is
why it became the primary corpus rather than FishNet.

iNaturalist explicitly directs large-scale ML users at this bulk export rather
than the public API, so this project never calls `api.inaturalist.org`.

**What the export contains.** Only photos under CC0 or a Creative Commons
licence are included; All-Rights-Reserved photos are excluded upstream. We
still normalise defensively rather than assuming.

**Attribution.** Built from `observers.name` falling back to `observers.login`,
in the form iNaturalist specifies:

* CC0 → `<name>, no rights reserved (CC0)`
* other CC → `© <name>, some rights reserved (<licence>)`

Emitted per image into `ATTRIBUTIONS.csv`.

**Caveat we accept:** the licence enum collapses CC version numbers
(`CC BY 2.0` → `CC-BY-4.0`). The exact original string is preserved in
`license_raw` in the provenance store, so attribution can cite the real version.

---

## 3. GBIF Backbone Taxonomy — **accepted, taxonomy only**

| | |
|---|---|
| Dataset key | `d7dddbf4-2cf0-4f39-9b2a-bb099caae36c` |
| Licence | `http://creativecommons.org/licenses/by/4.0/legalcode` — **CC BY 4.0** |
| How verified | `GET https://api.gbif.org/v1/dataset/d7dddbf4-...` → `"license"` field |
| Version pinned | `2023-08-28` (latest published; `current/` resolves to it) |
| File | `simple.txt.gz`, 489 MB, SHA-256 `fde017e1315b4ae6fc1e1bae79f9cfd234b8ba40f6f4fb5ac031084a3b1763f0` |
| Verdict | **Accepted** for taxonomy and synonymy. Commercially usable with attribution. |

Required attribution, carried into every pack that embeds this taxonomy:

> GBIF Secretariat (2023). GBIF Backbone Taxonomy. Checklist dataset
> <https://doi.org/10.15468/39omei>. Licensed CC BY 4.0.

**Important scope limit.** The CC BY 4.0 licence covers the *backbone
checklist*. It does **not** license occurrence records or any media reachable
through GBIF. An occurrence record's licence is not its photograph's licence.
Any future GBIF media harvesting must read the media-level licence.

---

## 4. FishNet (ICCV 2023) — **rejected from every corpus**

The brief named FishNet first, so it was checked first. It cannot be used.

| Check | Finding |
|---|---|
| Project page licence | States only that *the website* is CC BY-SA 4.0. **No dataset licence stated.** |
| Repository licence | **None.** `GET https://api.github.com/repos/faixan-khan/FishNet` → `"license": null`. No LICENSE file in the tree. |
| Image distribution | A single Google Drive archive. **No per-image licence metadata.** |
| Upstream image source | FishBase, where photographs are individually copyrighted by contributors and the **default when no licence is selected is All Rights Reserved**. |

**Verdict: excluded from `production`, `production_sa` *and* `research_nc`.**

Note the second part especially: FishNet is excluded even from the research
corpus. `research_nc` relaxes *non-commercial*, not *unknown*. A licence we
cannot establish is exactly the case the policy exists to refuse, and "it's only
for research" is not a licence.

FishNet remains useful and is credited as:

* a **published benchmark** to compare our numbers against, and
* prior art whose taxonomy and functional-trait *schema* informed ours.

No FishNet bytes are redistributed, trained on, or shipped.

```
@InProceedings{Khan_2023_ICCV,
  author    = {Khan, Faizan Farooq and Li, Xiang and Temple, Andrew J. and Elhoseiny, Mohamed},
  title     = {FishNet: A Large-scale Dataset and Benchmark for Fish Recognition,
               Detection, and Functional Trait Prediction},
  booktitle = {ICCV}, year = {2023}, pages = {20496--20506}
}
```

### FishBase

FishBase's **database content is CC BY-NC 4.0**, and its photographs are
individually copyrighted with ARR as the default.

Consequence: FishBase cannot supply species facts for a commercially
distributable app, which is inconvenient because it is the canonical fish-facts
resource. Our species database is therefore built from CC0/CC BY sources
(Wikidata, GBIF) and fields we cannot source that way are left **absent rather
than invented**. See [`SPECIES_DATA.md`](SPECIES_DATA.md).

---

## 5. CC BY-SA: why it is quarantined behind an opt-in

ShareAlike requires adaptations to be licensed under compatible terms. The
unsettled question is whether **model weights trained on a CC BY-SA image are
an "adapted work"** of that image.

* If **no** (weights are a statistical summary, not an adaptation), then CC BY-SA
  media is as usable as CC BY and only attribution is owed.
* If **yes**, shipping those weights would oblige us to license the weights
  under CC BY-SA — which is a legitimate choice for an open-source project but
  must be a deliberate one, not an accident of a data-loading glob.

Because the question is genuinely open, the default `production` policy
**excludes** CC BY-SA, and `production_sa` exists to enable it explicitly. The
resulting pack manifest records which policy built the model, so a released
artefact always states its own provenance.

The same reasoning applies more strongly to CC BY-NC-SA, which is
non-commercial *and* copyleft.

---

## 6. Why NoDerivatives is excluded even from research

ND permits redistribution of the *unmodified* work. Training involves
resizing, cropping and augmenting — plainly modification — and the resulting
model is at minimum arguably derivative.

Rather than take a position on it, `License.allows_derivatives` returns `False`
for ND and no policy admits ND media. The cost is small (ND is a small fraction
of iNaturalist media) and the downside of being wrong is large.

---

## 7. Sources evaluated and not (yet) used

| Source | Status | Reason |
|---|---|---|
| **Wikimedia Commons** | Planned, supplementary | Per-file licence is machine-readable via the API, so it satisfies the policy. Deferred: iNaturalist alone already yields more images than we can train on, and Commons' fish images skew heavily toward museum specimens and aquarium shots rather than angler-held fish. |
| **GBIF occurrence media** | Deferred | Usable in principle, but media licences are heterogeneous and a large share of records resolve back to iNaturalist anyway, so it would mostly duplicate what we already have. |
| **WildFish / WildFish++** | Rejected | No clear redistribution licence located for the imagery. |
| **Fish4Knowledge** | Rejected for training | Underwater CCTV footage from Taiwanese reefs. Domain is far from angler photographs and the species set is narrow. |
| **FathomNet** | Deferred | Deep-sea ROV imagery, CC BY for much of it, but the taxa are almost entirely species a recreational angler will never encounter. |
| **The Nature Conservancy Fisheries Monitoring** | Rejected | Kaggle competition terms restrict use to the competition. |
| **Museum / institutional collections** | Deferred | Preserved specimens differ so much in colour and posture from live fish that they risk teaching the model the wrong features. Potentially useful later as an explicit `specimen` context tag. |

---

## 8. Generated artefacts

| File | Contents |
|---|---|
| `ATTRIBUTIONS.csv` | One row per image in a corpus: id, taxon, source, URL, creator, licence, licence URL, attribution string |
| `ATTRIBUTIONS.summary.json` | Image counts per licence for the corpus |
| [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) | Schema and rebuild procedure for the provenance store |

Rebuild a corpus under a different licence filter with:

```bash
.venv/Scripts/python tools/dataset.py build-corpus --policy production
```

Because storage is content-addressed, changing the policy re-selects rows but
never re-downloads pixels already held.
