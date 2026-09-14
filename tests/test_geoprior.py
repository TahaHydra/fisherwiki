"""geoprior.build's use_all_observations must actually reach beyond training.

The docstring promises the histogram covers "all research-grade observations
of each taxon, not only the images we downloaded", specifically so the prior
describes the species' real range rather than this project's own download
sampling. The query used to join `corpus_members USING (candidate_id)`, which
restricts to exactly the candidates selected into the training corpus - the
sampling the docstring says it exists to escape. These tests build a
provenance store where a taxon has a handful of *downloaded* observations in
one place and many more *discovered-but-never-downloaded* observations
somewhere else, and check which location(s) the built prior actually reflects.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import duckdb  # noqa: E402

from fwdata import geoprior  # noqa: E402

TRAIN_CELL = geoprior.pack_cell(52.0, 5.0, 2.0)     # Netherlands-ish
DISCOVERED_ONLY_CELL = geoprior.pack_cell(-33.0, 151.0, 2.0)  # Sydney-ish


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "provenance.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""
        CREATE TABLE corpus_members (
            corpus VARCHAR, candidate_id VARCHAR, sha256 VARCHAR,
            class_id BIGINT, taxon_id BIGINT, split VARCHAR, weight DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE candidates (
            candidate_id VARCHAR, taxon_id BIGINT, latitude DOUBLE,
            longitude DOUBLE, group_key VARCHAR, quality_grade VARCHAR
        )
    """)
    # Only needed for the use_all_observations=False path, which reads the
    # `provenance` view (stored JOIN candidates) rather than `candidates`
    # directly - a downloaded image gets a `stored` row, a merely-discovered
    # candidate does not.
    con.execute("CREATE TABLE stored (candidate_id VARCHAR)")
    con.execute("""
        CREATE VIEW provenance AS
        SELECT c.*, s.candidate_id AS _stored_id
        FROM candidates c JOIN stored s USING (candidate_id)
    """)

    # 5 downloaded, trained-on observations - all at TRAIN_CELL's location.
    for i in range(5):
        con.execute(
            "INSERT INTO corpus_members VALUES ('c', ?, ?, 0, 1000, 'train', 1.0)",
            [f"trained:{i}", f"sha{i}"],
        )
        con.execute(
            "INSERT INTO candidates VALUES (?, 1000, 52.0, 5.0, ?, 'research')",
            [f"trained:{i}", f"obs-trained-{i}"],
        )
        con.execute("INSERT INTO stored VALUES (?)", [f"trained:{i}"])

    # 20 discovered candidates of the *same taxon*, never downloaded (no
    # corpus_members row), at a different location entirely.
    for i in range(20):
        con.execute(
            "INSERT INTO candidates VALUES (?, 1000, -33.0, 151.0, ?, 'research')",
            [f"discovered:{i}", f"obs-discovered-{i}"],
        )
    con.close()
    return db


def test_use_all_observations_reaches_discovered_but_undownloaded_candidates(tmp_path):
    db = _make_db(tmp_path)
    out = tmp_path / "prior.bin"

    geoprior.build(
        out_path=out, corpus="c", provenance_db=db, num_classes=1,
        use_all_observations=True, log=lambda m: None,
    )

    _cell_deg, _n, maps, totals = geoprior.read(out)
    cells = maps[0]

    assert TRAIN_CELL in cells, "the trained-on location should still be present"
    assert DISCOVERED_ONLY_CELL in cells, (
        "a location with only discovered-but-undownloaded candidates is missing - "
        "use_all_observations is still gated by corpus_members somewhere"
    )
    # 25 observations total (5 trained + 20 discovered-only), all counted.
    assert totals[0] == 25


def test_use_all_observations_false_sees_only_the_training_corpus(tmp_path):
    db = _make_db(tmp_path)
    out = tmp_path / "prior.bin"

    geoprior.build(
        out_path=out, corpus="c", provenance_db=db, num_classes=1,
        use_all_observations=False, log=lambda m: None,
    )

    _cell_deg, _n, maps, totals = geoprior.read(out)
    cells = maps[0]

    assert TRAIN_CELL in cells
    assert DISCOVERED_ONLY_CELL not in cells, (
        "use_all_observations=False should see only what was downloaded"
    )
    assert totals[0] == 5


def test_use_all_observations_is_scoped_to_the_requested_corpus(tmp_path):
    """class_id numbering is corpus-specific; a taxon must only contribute to
    the class_id *this* corpus assigned it, not to whatever class_id happens
    to share a number in an unrelated corpus."""
    db = _make_db(tmp_path)
    con = duckdb.connect(str(db))
    # A second corpus maps the same taxon to a *different* class_id.
    con.execute(
        "INSERT INTO corpus_members VALUES ('other', 'trained:0', 'sha0', 7, 1000, 'train', 1.0)"
    )
    con.close()

    out = tmp_path / "prior.bin"
    geoprior.build(
        out_path=out, corpus="c", provenance_db=db, num_classes=1,
        use_all_observations=True, log=lambda m: None,
    )
    _cell_deg, n, maps, _totals = geoprior.read(out)
    assert n == 1  # num_classes as requested for corpus 'c', not corpus 'other'
    assert TRAIN_CELL in maps[0]
