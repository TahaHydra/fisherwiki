#!/usr/bin/env python
"""FisherWiki taxonomy command line.

    python tools/taxonomy.py reconcile
    python tools/taxonomy.py lookup "Sander lucioperca"
    python tools/taxonomy.py lookup "Stizostedion lucioperca"   # a synonym
    python tools/taxonomy.py synonyms --species "Esox lucius"
    python tools/taxonomy.py stats
    python tools/taxonomy.py disagreements --limit 20

``reconcile`` is the only subcommand that writes: it rebuilds the canonical
taxonomy from the iNaturalist and GBIF bulk exports and appends any new taxa to
the committed registry. The rest are read-only and exist because a taxonomy you
cannot interrogate is a taxonomy you cannot trust.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fwdata.config import PATHS  # noqa: E402
from fwdata.taxonomy.names import canonical_form, parse_scientific_name  # noqa: E402
from fwdata.taxonomy.registry import TaxonRegistry  # noqa: E402


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _taxonomy_db():
    import duckdb

    p = PATHS.work / "taxonomy.duckdb"
    if not p.exists():
        raise SystemExit(
            f"{p} not found - run `python tools/taxonomy.py reconcile` first"
        )
    return duckdb.connect(str(p), read_only=True)


# ---------------------------------------------------------------------------
def cmd_reconcile(args: argparse.Namespace) -> int:
    from fwdata.taxonomy.reconcile import reconcile

    stats = reconcile(save=not args.dry_run, log=_log)
    _log("")
    for k, v in stats.as_dict().items():
        _log(f"  {k:24} {v:,}" if isinstance(v, int) else f"  {k:24} {v}")
    if args.dry_run:
        _log("\n(dry run - registry and taxonomy database not written)")
    else:
        reg = TaxonRegistry()
        _log(f"\nregistry: {reg.path} ({len(reg):,} taxa)")
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    """Resolve a name, following synonyms to the accepted taxon."""
    query = " ".join(args.name)
    parsed = parse_scientific_name(query)
    canon = parsed.canonical
    _log(f"query      : {query!r}")
    _log(f"canonical  : {canon!r}  (rank {parsed.rank})")
    if parsed.authorship:
        _log(f"authorship : {parsed.authorship}")
    if parsed.uncertain:
        _log("note       : open nomenclature marker present (sp./cf./aff.)")
    if parsed.hybrid:
        _log("note       : hybrid marker present")
    _log("")

    reg = TaxonRegistry()
    rec = reg.get(canon)
    if rec is not None:
        resolved = reg.resolve(rec.fw_taxon_id)
        _log(f"fw_taxon_id: {rec.fw_taxon_id}")
        _log(f"rank       : {rec.rank}")
        if resolved and resolved.fw_taxon_id != rec.fw_taxon_id:
            _log(f"merged into: {resolved.fw_taxon_id} ({resolved.canonical_name})")
        for field in ("gbif_taxon_id", "inat_taxon_id", "worms_aphia_id",
                      "wikidata_qid"):
            v = getattr(rec, field)
            if v:
                _log(f"{field:11}: {v}")
    else:
        _log("not in the canonical registry as an accepted name")

    con = _taxonomy_db()
    try:
        rows = con.execute(
            "SELECT accepted_canonical, synonym_status, source FROM synonyms "
            "WHERE synonym_canonical = ?",
            [canon],
        ).fetchall()
        if rows:
            _log("")
            _log("this name is a SYNONYM of:")
            for acc, status, source in rows:
                _log(f"  {acc}   ({status}, per {source})")

        ladder = con.execute(
            "SELECT class_name, order_name, family_name, genus_name, active, "
            "       gbif_taxon_key, gbif_status "
            "FROM taxa_joined WHERE canonical_name = ?",
            [canon],
        ).fetchone()
        if ladder:
            cls, order, fam, gen, active, gbif_key, gbif_status = ladder
            _log("")
            _log("classification:")
            for label, value in (("class", cls), ("order", order),
                                 ("family", fam), ("genus", gen)):
                if value:
                    _log(f"  {label:7}: {value}")
            _log(f"  active : {active}")
            if gbif_key:
                _log(f"  gbif   : {gbif_key} ({gbif_status})")
    finally:
        con.close()
    return 0


def cmd_synonyms(args: argparse.Namespace) -> int:
    canon = canonical_form(args.species)
    con = _taxonomy_db()
    try:
        rows = con.execute(
            "SELECT synonym_canonical, synonym_scientific_name, synonym_status "
            "FROM synonyms WHERE accepted_canonical = ? ORDER BY synonym_canonical",
            [canon],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        _log(f"no synonyms recorded for {canon!r}")
        return 0
    _log(f"{len(rows)} synonyms of {canon}:")
    for c, sci, status in rows[: args.limit]:
        _log(f"  {c:42} {status:22} {sci or ''}")
    if len(rows) > args.limit:
        _log(f"  ... and {len(rows) - args.limit} more")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    reg = TaxonRegistry()
    _log(f"registry file   : {reg.path}")
    _log(f"canonical taxa  : {len(reg):,}")
    _log(f"  active        : {len(reg.active):,}")
    _log(f"  merged away   : {len(reg) - len(reg.active):,}")
    by_rank: dict[str, int] = {}
    with_gbif = 0
    for r in reg.by_id.values():
        by_rank[r.rank] = by_rank.get(r.rank, 0) + 1
        if r.gbif_taxon_id:
            with_gbif += 1
    _log("  by rank       : " + ", ".join(
        f"{k}={v:,}" for k, v in sorted(by_rank.items(), key=lambda kv: -kv[1])
    ))
    _log(f"  with GBIF id  : {with_gbif:,} "
         f"({100 * with_gbif / max(1, len(reg)):.1f}%)")

    report = PATHS.work / "reconcile_report.json"
    if report.exists():
        _log("")
        _log("last reconciliation:")
        for k, v in json.loads(report.read_text(encoding="utf-8")).items():
            _log(f"  {k:24} {v:,}" if isinstance(v, int) else f"  {k:24} {v}")
    return 0


def cmd_disagreements(args: argparse.Namespace) -> int:
    """Taxa iNaturalist treats as accepted that GBIF treats as synonyms.

    Recorded rather than resolved. Both are curated authorities that update on
    different schedules, and silently picking one would hide a real ambiguity.
    """
    con = _taxonomy_db()
    try:
        rows = con.execute(
            """
            SELECT t.canonical_name, s.accepted_canonical, s.synonym_status
            FROM taxa_joined t
            JOIN synonyms s ON s.synonym_canonical = t.canonical_name
            WHERE t.active AND t.rank = 'species'
            ORDER BY t.canonical_name
            """
        ).fetchall()
    finally:
        con.close()
    _log(f"{len(rows):,} taxa are active in iNaturalist but a synonym in GBIF")
    _log("(recorded, not resolved - we follow iNaturalist because the images "
         "are labelled with its ids)")
    _log("")
    for inat_name, gbif_accepted, status in rows[: args.limit]:
        _log(f"  {inat_name:38} -> GBIF: {gbif_accepted:38} ({status})")
    if len(rows) > args.limit:
        _log(f"  ... and {len(rows) - args.limit:,} more")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="taxonomy.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("reconcile", help="rebuild canonical taxonomy from bulk exports")
    p.add_argument("--dry-run", action="store_true",
                   help="report without writing the registry or database")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("lookup", help="resolve a name, following synonyms")
    p.add_argument("name", nargs="+")
    p.set_defaults(func=cmd_lookup)

    p = sub.add_parser("synonyms", help="list synonyms of an accepted name")
    p.add_argument("--species", required=True)
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_synonyms)

    p = sub.add_parser("stats", help="registry and reconciliation summary")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("disagreements", help="iNaturalist/GBIF status conflicts")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_disagreements)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
