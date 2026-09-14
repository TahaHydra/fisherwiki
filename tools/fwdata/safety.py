"""Loads and validates the curated safety-warning table.

Handling-safety facts get the highest evidentiary bar in this project, for an
asymmetric reason: a missing warning is bad, but a *wrong* warning erodes trust
in the warnings that matter, and the ones that matter concern stonefish and
stingrays.

So this module refuses to load anything malformed rather than skipping it. A
typo in a family name would otherwise mean a venom warning silently never
reaches the user, which is the failure mode with the worst consequences and the
lowest chance of being noticed.

Warnings are expressed at a taxonomic **rank** (family, genus, order, class) and
expanded to every species in the pack that sits under that rank. That matches
how the evidence actually exists - Smith & Wheeler (2006) establish venom for
lineages, not for individual species - and it means a species with thin data
still inherits its family's warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import REPO_ROOT
from .speciesdb import Source

DEFAULT_PATH = REPO_ROOT / "data" / "safety" / "safety_warnings.yaml"

VALID_SEVERITIES = ("info", "caution", "danger")
VALID_RANKS = ("species", "genus", "family", "order", "class")


class SafetyDataError(ValueError):
    """The curated table is malformed. Never downgraded to a warning."""


@dataclass(frozen=True)
class SafetyWarning:
    rank: str
    taxon: str
    kind: str
    severity: str
    summary: str
    detail: str | None
    source_id: str


def load(path: Path | None = None) -> tuple[list[Source], list[SafetyWarning]]:
    """Parse and validate the table. Raises on any problem."""
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        raise SafetyDataError(f"{p} not found")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    raw_sources = doc.get("sources") or []
    if not raw_sources:
        raise SafetyDataError("no sources declared")
    sources: list[Source] = []
    seen_ids: set[str] = set()
    for s in raw_sources:
        for field in ("source_id", "title", "license", "retrieved_on", "citation"):
            if not s.get(field):
                raise SafetyDataError(f"source missing {field!r}: {s}")
        if s["source_id"] in seen_ids:
            raise SafetyDataError(f"duplicate source_id {s['source_id']!r}")
        seen_ids.add(s["source_id"])
        sources.append(
            Source(
                source_id=s["source_id"],
                title=" ".join(str(s["title"]).split()),
                publisher=s.get("publisher"),
                url=s.get("url"),
                license=s["license"],
                license_url=s.get("license_url"),
                retrieved_on=str(s["retrieved_on"]),
                citation=" ".join(str(s["citation"]).split()),
                notes=" ".join(str(s["notes"]).split()) if s.get("notes") else None,
            )
        )

    warnings: list[SafetyWarning] = []
    seen: set[tuple[str, str, str]] = set()
    for w in doc.get("warnings") or []:
        for field in ("rank", "taxon", "kind", "severity", "summary", "source_id"):
            if not w.get(field):
                raise SafetyDataError(f"warning missing {field!r}: {w}")
        if w["rank"] not in VALID_RANKS:
            raise SafetyDataError(f"invalid rank {w['rank']!r} for {w['taxon']!r}")
        if w["severity"] not in VALID_SEVERITIES:
            raise SafetyDataError(
                f"invalid severity {w['severity']!r} for {w['taxon']!r}; "
                f"must be one of {VALID_SEVERITIES}"
            )
        if w["source_id"] not in seen_ids:
            raise SafetyDataError(
                f"warning for {w['taxon']!r} cites unknown source "
                f"{w['source_id']!r}"
            )
        key = (w["rank"], w["taxon"], w["kind"])
        if key in seen:
            raise SafetyDataError(f"duplicate warning {key}")
        seen.add(key)
        warnings.append(
            SafetyWarning(
                rank=w["rank"],
                taxon=w["taxon"],
                kind=w["kind"],
                severity=w["severity"],
                summary=" ".join(str(w["summary"]).split()),
                detail=" ".join(str(w["detail"]).split()) if w.get("detail") else None,
                source_id=w["source_id"],
            )
        )

    if not warnings:
        raise SafetyDataError("no warnings declared")
    return sources, warnings


def expand(
    warnings: list[SafetyWarning],
    taxa: list[dict],
    log=print,
) -> tuple[list[dict], dict]:
    """Expand rank-level warnings onto the concrete taxa in a pack.

    ``taxa`` rows need ``fw_taxon_id`` plus whichever of ``genus``, ``family``,
    ``order``, ``class`` are known.

    Returns ``(rows, report)``. The report names any warning that matched
    **nothing**, because a family name that matches no species in the pack is
    almost always a typo, and the consequence of that typo is a venom warning
    that silently never appears.
    """
    by_rank: dict[str, dict[str, list[SafetyWarning]]] = {r: {} for r in VALID_RANKS}
    for w in warnings:
        by_rank[w.rank].setdefault(w.taxon, []).append(w)

    rows: list[dict] = []
    matched: dict[str, int] = {}

    for t in taxa:
        fields = {
            "species": t.get("scientific_name"),
            "genus": t.get("genus"),
            "family": t.get("family"),
            "order": t.get("order"),
            "class": t.get("class"),
        }
        # Most specific first, so a species-level override beats its family.
        applied: set[str] = set()
        for rank in VALID_RANKS:
            value = fields.get(rank)
            if not value:
                continue
            for w in by_rank[rank].get(value, []):
                if w.kind in applied:
                    continue
                applied.add(w.kind)
                key = f"{w.rank}:{w.taxon}:{w.kind}"
                matched[key] = matched.get(key, 0) + 1
                rows.append(
                    {
                        "fw_taxon_id": t["fw_taxon_id"],
                        "kind": w.kind,
                        "severity": w.severity,
                        "summary": w.summary,
                        "detail": w.detail,
                        "applies_to": w.rank,
                        "source_id": w.source_id,
                    }
                )

    unmatched = [
        f"{w.rank}:{w.taxon}:{w.kind}"
        for w in warnings
        if f"{w.rank}:{w.taxon}:{w.kind}" not in matched
    ]
    report = {
        "warnings_defined": len(warnings),
        "rows_generated": len(rows),
        "species_with_a_warning": len({r["fw_taxon_id"] for r in rows}),
        "unmatched_warnings": unmatched,
        "matched": matched,
    }
    if unmatched:
        log(
            f"  note: {len(unmatched)} curated warnings matched no species in "
            f"this pack: {', '.join(unmatched[:8])}"
            + ("..." if len(unmatched) > 8 else "")
        )
    return rows, report
