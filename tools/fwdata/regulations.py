"""Fishing regulation rule packs - deliberately separate from species data.

Why this is its own thing
-------------------------
Biological facts are approximately timeless: a pike has teeth, a weeverfish has
venomous spines. Fishing regulations are the opposite - a size limit, a closed
season or a bag limit can change between one trip and the next, and an angler
relying on a stale copy can be fined or can kill a fish they should have
returned.

So regulations are **never** mixed into the species database. They live in their
own pack, with their own version, their own jurisdiction, and their own validity
dates, and the app is required to say how old its copy is.

Three rules this module enforces
--------------------------------
1. **Every rule carries a jurisdiction, a source URL and a retrieval date.**
   A rule without a citable source does not load. This is the same bar as
   safety warnings and for the same reason: being confidently wrong is worse
   than being silent.
2. **Every rule carries `valid_from`, and `valid_until` where the source states
   one.** :meth:`RuleSet.staleness` reports how old the data is so the UI can
   show it prominently rather than burying it.
3. **Nothing is inferred.** No rule is generated, interpolated between
   jurisdictions, or guessed from a similar region. If a jurisdiction is not
   covered, it is absent, and the app says it has no data rather than offering
   a neighbouring county's limits.

Status
------
**Scaffolding only.** The schema, loader and validation are implemented and
tested; no jurisdiction data ships yet, because sourcing regulations under a
licence that permits redistribution requires per-jurisdiction review that has
not been done. The app treats a missing regulation pack as normal and the
feature is entirely optional - it does not block identification, which is the
product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

VALID_RULE_KINDS = (
    "minimum_size",
    "maximum_size",
    "slot_limit",
    "bag_limit",
    "closed_season",
    "licence_required",
    "gear_restriction",
    "catch_and_release_only",
    "protected",
    "other",
)

VALID_UNITS = ("cm", "mm", "inch", "kg", "g", "lb", "count", "none")


class RegulationDataError(ValueError):
    """The rule pack is malformed. Never downgraded to a warning."""


@dataclass(frozen=True)
class RuleSource:
    source_id: str
    title: str
    authority: str
    url: str
    retrieved_on: str
    licence: str
    notes: str | None = None


@dataclass(frozen=True)
class Rule:
    """One regulation, always tied to a jurisdiction and a source."""

    jurisdiction: str            # ISO-ish code, e.g. 'GB-ENG', 'US-FL', 'FR'
    kind: str
    #: Taxon this applies to, by canonical scientific name, or a rank-level
    #: name (family/genus). Null means it applies to all species.
    taxon: str | None
    taxon_rank: str
    value_num: float | None
    unit: str
    text: str
    valid_from: str
    valid_until: str | None
    source_id: str
    waters: str | None = None    # 'freshwater' | 'marine' | named water body

    @property
    def is_currently_valid(self) -> bool:
        today = date.today().isoformat()
        if self.valid_from > today:
            return False
        return self.valid_until is None or self.valid_until >= today


@dataclass
class RuleSet:
    jurisdiction: str
    version: int
    retrieved_on: str
    sources: list[RuleSource] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)

    def staleness(self, today: date | None = None) -> int:
        """Days since this data was retrieved. The UI must surface this."""
        t = today or date.today()
        try:
            got = datetime.strptime(self.retrieved_on, "%Y-%m-%d").date()
        except ValueError:
            return 10_000
        return (t - got).days

    def is_stale(self, threshold_days: int = 180) -> bool:
        """Regulations change at least annually in most jurisdictions.

        Six months is deliberately conservative: an angler shown a stale rule
        confidently is worse off than one told the app does not know.
        """
        return self.staleness() >= threshold_days

    def for_taxon(self, scientific_name: str, genus: str | None = None,
                  family: str | None = None) -> list[Rule]:
        """Rules applying to a taxon, most specific first.

        Rules with no taxon apply to everything and come last, so a
        species-specific size limit is shown above a general licence
        requirement.
        """
        out: list[Rule] = []
        for rank, value in (("species", scientific_name), ("genus", genus),
                            ("family", family)):
            if not value:
                continue
            out += [
                r for r in self.rules
                if r.taxon_rank == rank and r.taxon == value and r.is_currently_valid
            ]
        out += [r for r in self.rules if r.taxon is None and r.is_currently_valid]
        return out


def load(path: Path) -> RuleSet:
    """Parse and validate a rule pack. Raises on any problem."""
    p = Path(path)
    if not p.exists():
        raise RegulationDataError(f"{p} not found")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    for field_name in ("jurisdiction", "version", "retrieved_on"):
        if not doc.get(field_name):
            raise RegulationDataError(f"rule pack missing {field_name!r}")

    sources: list[RuleSource] = []
    seen: set[str] = set()
    for s in doc.get("sources") or []:
        for f in ("source_id", "title", "authority", "url", "retrieved_on", "licence"):
            if not s.get(f):
                raise RegulationDataError(f"regulation source missing {f!r}: {s}")
        if s["source_id"] in seen:
            raise RegulationDataError(f"duplicate source_id {s['source_id']!r}")
        seen.add(s["source_id"])
        sources.append(RuleSource(**{k: s.get(k) for k in
                                     ("source_id", "title", "authority", "url",
                                      "retrieved_on", "licence", "notes")}))
    if not sources:
        raise RegulationDataError("a rule pack must declare at least one source")

    rules: list[Rule] = []
    for r in doc.get("rules") or []:
        for f in ("jurisdiction", "kind", "text", "valid_from", "source_id"):
            if not r.get(f):
                raise RegulationDataError(f"rule missing {f!r}: {r}")
        if r["kind"] not in VALID_RULE_KINDS:
            raise RegulationDataError(
                f"unknown rule kind {r['kind']!r}; must be one of {VALID_RULE_KINDS}"
            )
        unit = r.get("unit", "none")
        if unit not in VALID_UNITS:
            raise RegulationDataError(f"unknown unit {unit!r}")
        if r["source_id"] not in seen:
            raise RegulationDataError(
                f"rule cites unknown source {r['source_id']!r}"
            )
        # A numeric rule without a number is a rule that cannot be applied.
        if r["kind"] in ("minimum_size", "maximum_size", "bag_limit") \
                and r.get("value_num") is None:
            raise RegulationDataError(
                f"{r['kind']} rule has no value_num: {r['text'][:60]}"
            )
        rules.append(
            Rule(
                jurisdiction=r["jurisdiction"],
                kind=r["kind"],
                taxon=r.get("taxon"),
                taxon_rank=r.get("taxon_rank", "species"),
                value_num=r.get("value_num"),
                unit=unit,
                text=" ".join(str(r["text"]).split()),
                valid_from=str(r["valid_from"]),
                valid_until=str(r["valid_until"]) if r.get("valid_until") else None,
                source_id=r["source_id"],
                waters=r.get("waters"),
            )
        )

    return RuleSet(
        jurisdiction=doc["jurisdiction"],
        version=int(doc["version"]),
        retrieved_on=str(doc["retrieved_on"]),
        sources=sources,
        rules=rules,
    )


def available_packs(root: Path | None = None) -> list[Path]:
    """Rule packs present on disk. Empty is a normal, supported state.

    Files whose name begins with ``_`` are templates and documentation, not
    data. Without this filter ``_template.yaml`` loads as a real jurisdiction
    pack - and it contains an illustrative pike size limit of 0 cm, which would
    be shown to a user as though it were the law. Caught by a test; the failure
    mode is exactly the kind this module exists to prevent.
    """
    from .config import REPO_ROOT

    d = Path(root or (REPO_ROOT / "data" / "regulations"))
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.yaml") if not p.name.startswith("_"))
