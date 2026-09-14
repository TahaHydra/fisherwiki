"""The SQL and Python canonicalisers must agree, or IDs silently diverge.

``reconcile.py`` canonicalises names in SQL (fast, runs over millions of GBIF
rows) while ``registry.py`` canonicalises in Python when minting ``fw_taxon_id``.
If those two implementations disagree for any name, the registry mints an ID
under one spelling while the corpus joins on another, and a class quietly loses
its images. These tests pin them together.
"""

from __future__ import annotations

import duckdb
import pytest

from fwdata.taxonomy.names import canonical_form
from fwdata.taxonomy.reconcile import _register_udfs

#: Names drawn from the shapes that actually occur in the iNaturalist and GBIF
#: exports (both supply authorship-free names).
REAL_NAMES = [
    "Perca fluviatilis",
    "Esox lucius",
    "Sander lucioperca",
    "Stizostedion lucioperca",
    "Micropterus salmoides salmoides",
    "Salmo trutta",
    "Oncorhynchus mykiss",
    "Thunnus thynnus",
    "Scophthalmus maximus",
    "Gasterosteus aculeatus aculeatus",
    "Perca (Perca) fluviatilis",
    "PERCA FLUVIATILIS",
    "perca fluviatilis",
    "  Esox   lucius  ",
    "Actinopterygii",
    "Percidae",
    "Sebastes",
    "Coregonus lavaretus maraena",
    "Squalius cephalus",
    "Barbus barbus",
]


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    _register_udfs(c)
    yield c
    c.close()


@pytest.mark.parametrize("name", REAL_NAMES)
def test_sql_canon_matches_python_canon(con, name):
    sql_result = con.execute("SELECT canon(?)", [name]).fetchone()[0]
    assert sql_result == canonical_form(name), (
        f"canon mismatch for {name!r}: SQL={sql_result!r} Python={canonical_form(name)!r}"
    )


def test_sql_canon_handles_null_and_empty(con):
    assert con.execute("SELECT canon(NULL)").fetchone()[0] == ""
    assert con.execute("SELECT canon('')").fetchone()[0] == ""
    assert con.execute("SELECT canon('   ')").fetchone()[0] == ""


def test_sql_canon_strips_subgenus(con):
    assert con.execute("SELECT canon('Salmo (Salmo) trutta')").fetchone()[0] == (
        "Salmo trutta"
    )


def test_sql_canon_is_idempotent(con):
    for name in REAL_NAMES:
        once = con.execute("SELECT canon(?)", [name]).fetchone()[0]
        twice = con.execute("SELECT canon(?)", [once]).fetchone()[0]
        assert once == twice
