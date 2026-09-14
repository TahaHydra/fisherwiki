# Datasets

Every number in this document was measured on the actual data, not quoted.
Licensing analysis lives in [`DATASET_LICENSES.md`](DATASET_LICENSES.md);
this document is about *what the data is like* and *what we did to it*.

---

## 1. Sources actually used

| Source | Role | Size | Licence |
|---|---|---|---|
| **iNaturalist Open Data** | the image corpus | 33 GB of metadata → 307,415 images | per-photo CC0 / CC BY |
| **GBIF Backbone Taxonomy** | accepted names, synonymy | 489 MB, pinned `2023-08-28` | CC BY 4.0 |
| **Wikidata** | common names, IUCN status, cross-ids | 14 SPARQL queries | CC0 |

Nothing was scraped. iNaturalist and GBIF are official bulk exports; Wikidata is
its documented query service, used for a 1,963-name subset in 14 batched
requests.

**FishNet is not used.** It was checked first because the brief named it first,
and it has no licence at all — see
[`DATASET_LICENSES.md §4`](DATASET_LICENSES.md).

---

## 2. Funnel, with real numbers

```
iNaturalist 2026-08-27 snapshot
        │
        ▼  fish clades (Actinopterygii, Chondrichthyes, Myxini,
        │              Petromyzonti, Sarcopterygii)
        │  research grade only
 2,745,412 fish observations
        │
        ▼  join to photos
 4,168,455 licensed fish photographs   ·  16,112 taxa · 15,466 species
        │
        ▼  licence policy: CC0 / PD / CC BY only
   505,347 production-safe images          ← 12.1% survives
        │
        ▼  evidence bar: ≥40 images AND ≥25 independent observations
     1,963 species-level classes
        │
        ▼  cap 300 per species, diversity-ordered selection
   308,227 selected
        │
        ▼  download (803 gone from the source, 0 undecodable)
   307,415 stored   ·  37.9 GB  ·  2,066 taxa
        │
        ▼  dedupe by SHA-256, re-check the evidence bar
     1,978 classes
        │
        ▼  observation-grouped, per-class stratified split
   243,469 train · 30,589 val · 28,943 test     leakage: 0 / 0
```

### The number that shapes the whole project

**88% of the available fish photographs cannot be used.**

| licence | images | usable |
|---|---|---|
| CC BY-NC | 3,356,078 | no |
| **CC BY** | **429,365** | **yes** |
| CC BY-NC-ND | 151,232 | no |
| CC BY-NC-SA | 101,234 | no |
| **CC0** | **75,982** | **yes** |
| CC BY-SA | 45,033 | opt-in (`production_sa`) |
| CC BY-ND | 9,531 | no |

This is why the model covers 1,963 species rather than the 15,466 that have at
least one photograph. Training on the CC BY-NC pile would produce a model that
looks roughly four times better on paper and cannot be shipped.

---

## 3. Class distribution

Among the 1,963 species clearing the bar:

| images per species | species | total images |
|---|---|---|
| 40–100 | 782 | 52,127 |
| 100–300 | 804 | 136,974 |
| 300–1,000 | 334 | 171,686 |
| 1,000+ | 43 | 75,844 |

Capping at 300 per species reduces 436,631 images to 308,227 and pulls the
imbalance ratio from 25:1 down to 7.5:1. Training additionally uses
**square-root** inverse-frequency sampling rather than full balancing: full
balancing would over-sample a 40-image class about 7× per epoch and overfit
exactly the classes least able to afford it.

### Regional coverage

Species-level classes with real observation support per region (regions overlap,
so the column does not sum to 1,963):

| region | species classes | observations |
|---|---|---|
| Indo-Pacific | 841 | 108,554 |
| Australia / NZ | 680 | 57,481 |
| North America Atlantic | 573 | 95,036 |
| North America freshwater | 554 | 79,770 |
| Africa | 277 | 23,027 |
| South America | 223 | 20,214 |
| Europe freshwater | 211 | 32,074 |
| Europe Atlantic | 190 | 20,342 |
| North America Pacific | 187 | 19,208 |
| Mediterranean | 151 | 19,300 |

These numbers are what settled the one-global-model decision: the per-region
class lists sum to 3,887 because of overlap, so regional models would
collectively be larger than a single global model while each seeing less data.
See [`ARCHITECTURE.md §4`](ARCHITECTURE.md).

---

## 4. Selection, not just filtering

Which 300 images per species matters as much as how many. A species whose 300
photographs come from 4 observations is well-covered on paper and generalises
terribly.

Selection orders by:

1. `position_in_observation` — the first photo of each observation first,
2. hashed observation id — spread across observations,
3. hashed photographer id — spread across photographers,
4. hashed photo id — deterministic tie-break.

So the first 300 taken for a species are drawn from as many distinct sightings
and as many distinct people as the data allows.

---

## 5. Quality signals, and what we deliberately keep

Every stored image carries measured quality facts: dimensions, perceptual
hashes, Laplacian-variance blur score, clipped-pixel fraction, saturation
spread. Flags raised on the corpus:

| flag | images |
|---|---|
| `near_greyscale` | 5,713 |
| `heavily_clipped` | 350 |
| `very_dark` | 182 |
| `too_small` | 57 |
| `extreme_aspect` | 52 |
| `very_bright` | 22 |

**These are recorded, not deleted.** Users photograph fish in hands, in landing
nets, on grass, in boats, at night under a headtorch, wet and reflective and
bleeding and half out of frame. A dark, blurry photo of a fish in a net at dusk
is a *valuable* training example, not garbage — it is the hard case the app has
to handle.

Only genuinely unusable images are hard-rejected: undecodable, zero-sized, or
single-colour. Even that check had to be fixed: the first version rejected
anything with a low standard deviation, which threw out real night photographs
simply for being dark. It now uses peak-to-peak range, which separates "dark but
textured" from "one solid colour".

Truncated JPEGs are decoded as far as they go rather than discarded.

---

## 6. Leakage prevention

The split refuses to export if it leaks, and that check found three real
problems that no accuracy metric would have revealed:

1. **Same photo, two observations.** iNaturalist attaches one `photo_id` to more
   than one observation — 6,739 cases. Besides breaking the candidate primary
   key, that put one image in two groups and therefore two splits.
2. **Stratification applied to the wrong grouping.** Per-class stratification is
   sound only when a group maps to one class. Applied to photographer groups
   (who span many species) it produced 748 groups spanning splits.
3. **Byte-identical re-uploads.** 338 images with the same SHA-256 in two
   splits: the same photograph uploaded again as a new observation.

Additionally, 582 images carry **conflicting species labels** — identical bytes
submitted under two different taxa. These are counted and reported as a
label-quality signal rather than silently resolved.

---

## 7. Taxonomy reconciliation

iNaturalist drives the taxon set because our images are labelled with its ids,
and because GBIF's backbone does not link fish orders to class
`Actinopterygii` at all (`Perciformes`'s parent is phylum Chordata and
`classKey` is NULL throughout — verified against the live API), so fish cannot
be selected from GBIF by class.

GBIF is joined by canonical name to supply what iNaturalist's export lacks:

* **43,448 of 43,579** active fish taxa matched (99.7%)
* **66,754 synonyms** recovered — including the renames anglers actually
  remember, like `Stizostedion lucioperca` → `Sander lucioperca`
* **4,701 status disagreements** recorded rather than silently resolved

Wikidata then contributes, for the 1,963-species class list:

* 1,906 matched (97.1%)
* 1,289 English common names, 599 French, **127 languages** total
* 1,648 IUCN statuses

Coverage is uneven — 674 species have no English common name — but a missing
name is a blank field, not a wrong one.

---

## 8. Reproducing this

```bash
.venv/Scripts/python tools/dataset.py discover
.venv/Scripts/python tools/dataset.py download --source inaturalist   # 33 GB
.venv/Scripts/python tools/dataset.py download --source gbif          # 489 MB
.venv/Scripts/python tools/taxonomy.py reconcile
.venv/Scripts/python tools/dataset.py extract                         # ~8 min
.venv/Scripts/python tools/dataset.py plan --policy production
.venv/Scripts/python scripts/fetch_wikidata.py
.venv/Scripts/python tools/dataset.py fetch --policy production \
    --per-species-cap 300 --min-observations 25 --workers 64          # ~66 min
.venv/Scripts/python tools/dataset.py build-corpus --corpus global_v1
.venv/Scripts/python tools/dataset.py verify
```

Every step is resumable. `fetch` skips what is already stored and does not retry
URLs that failed permanently.

Wall-clock on the reference machine: about 2.5 hours end to end, dominated by
the 37.9 GB image download at ~80 images/second.

---

## 9. Sources evaluated and rejected

| Source | Why not |
|---|---|
| FishNet | No licence, anywhere. See `DATASET_LICENSES.md §4`. |
| FishBase | CC BY-NC — unusable for a distributable app. |
| WildFish / WildFish++ | No clear redistribution licence for the imagery. |
| Fish4Knowledge | Underwater CCTV from Taiwanese reefs; wrong domain, narrow species set. |
| FathomNet | Mostly CC BY, but deep-sea ROV imagery of taxa no angler will meet. |
| TNC Fisheries Monitoring | Kaggle competition terms restrict use to the competition. |
| Wikimedia Commons | Licence *is* machine-verifiable, so it passes policy. Deferred only because iNaturalist already yields more than we can train on, and Commons' fish images skew to museum specimens and aquaria. A genuine future addition. |
| Museum collections | Preserved specimens differ enough in colour and posture from live fish to risk teaching the wrong features. Possible later behind an explicit `specimen` context tag. |
