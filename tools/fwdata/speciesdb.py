"""Builds the offline SQLite species database shipped inside a pack.

Sourcing rule
-------------
**Every factual row carries a source id, and nothing is written without one.**
There is no code path in this module that inserts a biological fact with a NULL
``source_id``; the schema enforces it with ``NOT NULL`` and the loaders enforce
it by requiring a source key argument.

That constraint is what keeps the app honest. FishBase, the obvious place to get
fish facts, is CC BY-NC and therefore unusable for a commercially distributable
app (see ``docs/DATASET_LICENSES.md``). Rather than paraphrase it, or worse
generate plausible-sounding descriptions with a language model, fields we cannot
source from CC0/CC-BY data are simply **absent**. The app renders an absent
field as absent. An angler deciding whether a spine is venomous is not served by
a confident guess.

Schema overview
---------------
``taxa``                 canonical taxon rows keyed by ``fw_taxon_id``
``taxon_synonyms``       historical / alternative scientific names
``common_names``         multilingual vernacular names, ``lang`` is BCP-47
``regions``              region definitions carried for display
``taxon_regions``        which taxa occur in which region, with evidence counts
``habitats``             freshwater / marine / brackish and finer habitat notes
``traits``               typed key-value measurements (max length, weight, ...)
``diagnostic_features``  human-verifiable field marks
``similar_species``      confusable pairs with the specific difference
``safety_warnings``      venom / toxicity / handling, high evidentiary bar
``sources``              every source cited by any row above
``model_classes``        model class index -> fw_taxon_id
``model_metadata``       one row: model + calibration provenance
``pack_metadata``        one row: pack identity and build provenance

Indices are created for the two queries the app actually runs hot: lookup by
class index after inference, and search by name prefix.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA_PATH = Path(__file__).with_name("species_schema.sql")

#: DDL for a pack's species database. Kept in a .sql file so the Kotlin
#: test suite can build fixtures from the identical schema rather than a
#: hand-maintained copy that would silently drift.
SCHEMA_SQL = SCHEMA_PATH.read_text(encoding="utf-8")


@dataclass(frozen=True)
class Source:
    """A citable source. Every fact row references one of these."""

    source_id: str
    title: str
    license: str
    retrieved_on: str
    citation: str
    publisher: str | None = None
    url: str | None = None
    license_url: str | None = None
    notes: str | None = None

    def row(self) -> tuple:
        return (
            self.source_id, self.title, self.publisher, self.url, self.license,
            self.license_url, self.retrieved_on, self.citation, self.notes,
        )


class SpeciesDatabaseBuilder:
    """Creates and populates a pack's species database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path))
        self.con.executescript(SCHEMA_SQL)
        self._sources: set[str] = set()

    def close(self) -> None:
        self.con.commit()
        # A shipped database should be compact and have no side files.
        self.con.execute("PRAGMA optimize")
        self.con.execute("VACUUM")
        self.con.commit()
        self.con.close()

    def __enter__(self) -> "SpeciesDatabaseBuilder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- sources -----------------------------------------------------------
    def add_source(self, source: Source) -> str:
        self.con.execute(
            "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?)", source.row()
        )
        self._sources.add(source.source_id)
        return source.source_id

    def _require_source(self, source_id: str) -> None:
        if source_id not in self._sources:
            raise ValueError(
                f"unknown source {source_id!r}: register it with add_source() before "
                "inserting facts that cite it"
            )

    # -- taxa --------------------------------------------------------------
    def add_taxa(self, rows: list[dict], source_id: str) -> int:
        self._require_source(source_id)
        self.con.executemany(
            "INSERT OR REPLACE INTO taxa (fw_taxon_id, scientific_name, rank, genus, "
            "family, taxon_order, class, authorship, gbif_taxon_id, inat_taxon_id, "
            "worms_aphia_id, wikidata_qid, source_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    r["fw_taxon_id"], r["scientific_name"], r.get("rank", "species"),
                    r.get("genus"), r.get("family"), r.get("order"), r.get("class"),
                    r.get("authorship"), r.get("gbif_taxon_id"), r.get("inat_taxon_id"),
                    r.get("worms_aphia_id"), r.get("wikidata_qid"), source_id,
                )
                for r in rows
            ],
        )
        return len(rows)

    def add_synonyms(self, rows: list[tuple[int, str, str]], source_id: str) -> int:
        self._require_source(source_id)
        self.con.executemany(
            "INSERT OR REPLACE INTO taxon_synonyms VALUES (?,?,?,?)",
            [(t, n, k, source_id) for t, n, k in rows],
        )
        return len(rows)

    def add_common_names(
        self, rows: list[tuple[int, str, str, int]], source_id: str
    ) -> int:
        self._require_source(source_id)
        self.con.executemany(
            "INSERT OR REPLACE INTO common_names VALUES (?,?,?,?,?)",
            [(t, lang, name, pref, source_id) for t, lang, name, pref in rows],
        )
        return len(rows)

    def add_regions(self, regions: list[dict]) -> int:
        self.con.executemany(
            "INSERT OR REPLACE INTO regions VALUES (?,?,?,?)",
            [
                (r["id"], r["name"], r["water"], json.dumps(r["boxes"]))
                for r in regions
            ],
        )
        return len(regions)

    def add_taxon_regions(
        self, rows: list[tuple[int, str, int, float]], source_id: str
    ) -> int:
        self._require_source(source_id)
        self.con.executemany(
            "INSERT OR REPLACE INTO taxon_regions VALUES (?,?,?,?,?)",
            [(t, r, obs, share, source_id) for t, r, obs, share in rows],
        )
        return len(rows)

    def add_habitats(self, rows: list[tuple[int, str, str | None]], source_id: str) -> int:
        self._require_source(source_id)
        self.con.executemany(
            "INSERT OR REPLACE INTO habitats VALUES (?,?,?,?)",
            [(t, w, d, source_id) for t, w, d in rows],
        )
        return len(rows)

    def add_traits(self, rows: list[dict], source_id: str) -> int:
        self._require_source(source_id)
        self.con.executemany(
            "INSERT OR REPLACE INTO traits VALUES (?,?,?,?,?,?)",
            [
                (
                    r["fw_taxon_id"], r["key"], r.get("value_num"),
                    r.get("value_text"), r.get("unit"), source_id,
                )
                for r in rows
            ],
        )
        return len(rows)

    def add_diagnostic_features(self, rows: list[dict], source_id: str) -> int:
        self._require_source(source_id)
        self.con.executemany(
            "INSERT OR REPLACE INTO diagnostic_features VALUES (?,?,?,?,?)",
            [
                (r["fw_taxon_id"], r["ordinal"], r["feature"],
                 r.get("body_part"), source_id)
                for r in rows
            ],
        )
        return len(rows)

    def add_similar_species(self, rows: list[dict], source_id: str) -> int:
        self._require_source(source_id)
        self.con.executemany(
            "INSERT OR REPLACE INTO similar_species VALUES (?,?,?,?,?)",
            [
                (r["fw_taxon_id"], r["other_fw_taxon_id"], r["difference"],
                 r.get("confusion_rate"), source_id)
                for r in rows
            ],
        )
        return len(rows)

    def add_safety_warnings(self, rows: list[dict], source_id: str) -> int:
        """Safety facts. Highest evidentiary bar in the schema.

        A wrong warning is bad; a wrong *absence* of a warning is worse. Both
        are avoided the same way: only insert what a named, citable source
        actually says, and never infer.
        """
        self._require_source(source_id)
        for r in rows:
            if r.get("severity") not in ("info", "caution", "danger"):
                raise ValueError(f"invalid severity {r.get('severity')!r}")
            if not r.get("summary"):
                raise ValueError("safety warning needs a summary")
        self.con.executemany(
            "INSERT OR REPLACE INTO safety_warnings VALUES (?,?,?,?,?,?,?)",
            [
                (r["fw_taxon_id"], r["kind"], r["severity"], r["summary"],
                 r.get("detail"), r.get("applies_to", "species"), source_id)
                for r in rows
            ],
        )
        return len(rows)

    def add_model_classes(self, rows: list[dict]) -> int:
        self.con.executemany(
            "INSERT OR REPLACE INTO model_classes VALUES (?,?,?,?,?,?,?)",
            [
                (
                    r["class_index"], r["fw_taxon_id"], r["train_images"],
                    r["train_observations"], r.get("test_precision"),
                    r.get("test_recall"), r.get("test_support"),
                )
                for r in rows
            ],
        )
        return len(rows)

    def set_metadata(self, table: str, values: dict) -> None:
        if table not in ("model_metadata", "pack_metadata"):
            raise ValueError(table)
        self.con.executemany(
            f"INSERT OR REPLACE INTO {table} VALUES (?,?)",
            [(k, json.dumps(v) if not isinstance(v, str) else v)
             for k, v in values.items()],
        )

    # -- verification ------------------------------------------------------
    def verify(self) -> dict:
        """Check the invariants that make the database trustworthy."""
        cur = self.con.cursor()
        problems: list[str] = []

        cur.execute("PRAGMA foreign_key_check")
        fk = cur.fetchall()
        if fk:
            problems.append(f"{len(fk)} foreign key violations")

        # Every fact table must reference a registered source.
        for table in (
            "taxa", "taxon_synonyms", "common_names", "taxon_regions", "habitats",
            "traits", "diagnostic_features", "similar_species", "safety_warnings",
        ):
            cur.execute(
                f"SELECT count(*) FROM {table} t "
                "LEFT JOIN sources s ON s.source_id = t.source_id "
                "WHERE s.source_id IS NULL"
            )
            n = cur.fetchone()[0]
            if n:
                problems.append(f"{table}: {n} rows with an unregistered source")

        # Every model class must resolve to a taxon.
        cur.execute(
            "SELECT count(*) FROM model_classes m "
            "LEFT JOIN taxa t ON t.fw_taxon_id = m.fw_taxon_id "
            "WHERE t.fw_taxon_id IS NULL"
        )
        n = cur.fetchone()[0]
        if n:
            problems.append(f"model_classes: {n} classes with no taxon row")

        # Class indices must be dense from 0: a gap would silently shift labels.
        cur.execute("SELECT count(*), min(class_index), max(class_index) FROM model_classes")
        cnt, lo, hi = cur.fetchone()
        if cnt and (lo != 0 or hi != cnt - 1):
            problems.append(
                f"model_classes indices are not dense 0..{cnt - 1} (got {lo}..{hi})"
            )

        counts = {}
        for table in (
            "sources", "taxa", "taxon_synonyms", "common_names", "regions",
            "taxon_regions", "habitats", "traits", "diagnostic_features",
            "similar_species", "safety_warnings", "model_classes",
        ):
            cur.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cur.fetchone()[0]

        return {"ok": not problems, "problems": problems, "counts": counts}
