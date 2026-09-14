"""Append-only registry of canonical internal taxon IDs.

Why not just use GBIF or iNaturalist IDs?
----------------------------------------
Because both change. Taxa get merged, split, deprecated and re-keyed; a source
that deprecates an ID and reuses the name elsewhere would silently repoint a
model class at a different animal. And a model trained against one source's IDs
cannot be re-evaluated against another's.

So we mint our own. ``fw_taxon_id`` is assigned once, never reused, and never
changes meaning. The registry file lives in the repository (it is small -
roughly 37k fish taxa - and it is reviewable in a diff), so two developers who
rebuild the corpus from scratch get identical class identities.

Invariants
----------
1. **Append-only.** An ID is never deleted or reassigned. A taxon that turns out
   to be a synonym is marked ``merged_into`` rather than removed, so old packs
   remain interpretable.
2. **Name-keyed on the canonical form**, not the display name, so authorship and
   casing differences cannot mint duplicate IDs.
3. **Deterministic order.** New IDs are assigned in sorted name order within a
   run, so a rebuild from the same inputs produces the same file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

from ..config import REPO_ROOT
from .names import canonical_form

DEFAULT_REGISTRY = REPO_ROOT / "data" / "taxonomy" / "taxon_registry.tsv"

#: IDs below this are reserved for non-species sentinels (unknown, background).
FIRST_TAXON_ID = 1000

#: Sentinel classes that every model shares.
RESERVED: dict[int, str] = {
    0: "__unknown__",       # open-set rejection target
    1: "__not_a_fish__",    # negative class where a detector/classifier uses one
}

FIELDS = [
    "fw_taxon_id",
    "canonical_name",
    "rank",
    "gbif_taxon_id",
    "inat_taxon_id",
    "worms_aphia_id",
    "wikidata_qid",
    "merged_into",
    "created_on",
]


@dataclass
class TaxonRecord:
    fw_taxon_id: int
    canonical_name: str
    rank: str = "species"
    gbif_taxon_id: int | None = None
    inat_taxon_id: int | None = None
    worms_aphia_id: int | None = None
    wikidata_qid: str | None = None
    merged_into: int | None = None
    created_on: str = field(default_factory=lambda: date.today().isoformat())

    def row(self) -> dict[str, str]:
        return {
            "fw_taxon_id": str(self.fw_taxon_id),
            "canonical_name": self.canonical_name,
            "rank": self.rank,
            "gbif_taxon_id": "" if self.gbif_taxon_id is None else str(self.gbif_taxon_id),
            "inat_taxon_id": "" if self.inat_taxon_id is None else str(self.inat_taxon_id),
            "worms_aphia_id": ""
            if self.worms_aphia_id is None
            else str(self.worms_aphia_id),
            "wikidata_qid": self.wikidata_qid or "",
            "merged_into": "" if self.merged_into is None else str(self.merged_into),
            "created_on": self.created_on,
        }


def _to_int(v: str | None) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


class TaxonRegistry:
    """Load/mutate/save the canonical taxon ID registry."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DEFAULT_REGISTRY)
        self.by_id: dict[int, TaxonRecord] = {}
        self.by_name: dict[str, TaxonRecord] = {}
        self._next_id = FIRST_TAXON_ID
        if self.path.exists():
            self.load()

    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                rec = TaxonRecord(
                    fw_taxon_id=int(row["fw_taxon_id"]),
                    canonical_name=row["canonical_name"],
                    rank=row.get("rank") or "species",
                    gbif_taxon_id=_to_int(row.get("gbif_taxon_id")),
                    inat_taxon_id=_to_int(row.get("inat_taxon_id")),
                    worms_aphia_id=_to_int(row.get("worms_aphia_id")),
                    wikidata_qid=row.get("wikidata_qid") or None,
                    merged_into=_to_int(row.get("merged_into")),
                    created_on=row.get("created_on") or "",
                )
                self.by_id[rec.fw_taxon_id] = rec
                self.by_name[rec.canonical_name] = rec
                self._next_id = max(self._next_id, rec.fw_taxon_id + 1)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tsv.tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                               lineterminator="\n")
            w.writeheader()
            for tid in sorted(self.by_id):
                w.writerow(self.by_id[tid].row())
        tmp.replace(self.path)

    # -- lookup ------------------------------------------------------------
    def get(self, name: str) -> TaxonRecord | None:
        return self.by_name.get(canonical_form(name))

    def resolve(self, fw_taxon_id: int) -> TaxonRecord | None:
        """Follow ``merged_into`` chains to the surviving record."""
        seen: set[int] = set()
        rec = self.by_id.get(fw_taxon_id)
        while rec is not None and rec.merged_into is not None:
            if rec.fw_taxon_id in seen:
                raise ValueError(f"merge cycle at fw_taxon_id={rec.fw_taxon_id}")
            seen.add(rec.fw_taxon_id)
            rec = self.by_id.get(rec.merged_into)
        return rec

    # -- mutation ----------------------------------------------------------
    def ensure(self, name: str, rank: str = "species", **ids) -> TaxonRecord:
        """Return the record for ``name``, minting a new ID if unseen."""
        canon = canonical_form(name)
        if not canon:
            raise ValueError(f"unparseable taxon name: {name!r}")
        rec = self.by_name.get(canon)
        if rec is None:
            rec = TaxonRecord(self._next_id, canon, rank=rank)
            self._next_id += 1
            self.by_id[rec.fw_taxon_id] = rec
            self.by_name[canon] = rec
        for k, v in ids.items():
            if v not in (None, "") and getattr(rec, k, "missing") is None:
                setattr(rec, k, v)
        return rec

    def ensure_many(self, names: Iterable[tuple[str, str]]) -> dict[str, TaxonRecord]:
        """Mint IDs for many ``(name, rank)`` pairs in deterministic order."""
        out: dict[str, TaxonRecord] = {}
        for name, rank in sorted(set(names)):
            canon = canonical_form(name)
            if not canon:
                continue
            out[canon] = self.ensure(name, rank)
        return out

    def merge(self, loser_id: int, winner_id: int) -> None:
        """Record that ``loser_id`` is a synonym of ``winner_id``."""
        if loser_id == winner_id:
            return
        loser = self.by_id[loser_id]
        if winner_id not in self.by_id:
            raise KeyError(f"unknown winner fw_taxon_id={winner_id}")
        loser.merged_into = winner_id

    def __len__(self) -> int:
        return len(self.by_id)

    @property
    def active(self) -> list[TaxonRecord]:
        return [r for r in self.by_id.values() if r.merged_into is None]
