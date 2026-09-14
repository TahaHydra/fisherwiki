"""Fetch a licensed non-fish negative set for open-set evaluation.

    python scripts/fetch_nonfish_negatives.py --limit 3000

Why from iNaturalist rather than a generic image set
----------------------------------------------------
The point of a negative set is to test the *decision boundary*, and a boundary
is only meaningfully tested by things that could plausibly be confused. Generic
web photographs differ from our corpus in lighting, framing, camera and
photographer behaviour, so a model rejecting them would prove little.

These negatives come from the same source, the same photographers and the same
conditions as the positives — they simply are not fish. A heron standing in
water, a frog on a rock, a dragonfly on a reed: exactly the things an angler
photographs by accident.

Note the classes chosen. Amphibians, reptiles and aquatic invertebrates are
deliberately over-represented because they share habitat and posture with fish;
a model that rejects a butterfly but accepts a newt has not learned what a fish
is.

Stored with ``source_dataset = 'inaturalist-nonfish'`` so they can never be
mistaken for training data: the corpus builder selects on
``species_taxon_id IS NOT NULL``, which these rows deliberately leave null.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import duckdb  # noqa: E402

from fwdata import net  # noqa: E402
from fwdata.config import PATHS  # noqa: E402
from fwdata.images import ImageRejected, measure  # noqa: E402
from fwdata.licenses import PRODUCTION, normalize  # noqa: E402
from fwdata.provenance import ImageProvenance, ImageStore, ProvenanceDB  # noqa: E402
from fwdata.sources import inaturalist as inat  # noqa: E402

#: iNaturalist clade roots for the negative set, with a rough share of the
#: budget. Habitat-sharing groups are weighted up on purpose.
NEGATIVE_CLADES: dict[str, tuple[int, float]] = {
    # name: (iNat taxon_id, share of the sample)
    "Amphibia": (20978, 0.16),      # frogs, newts - share water and posture
    "Reptilia": (26036, 0.12),      # turtles, water snakes
    "Mollusca": (47115, 0.10),      # squid, octopus, shells
    "Arthropoda": (47120, 0.14),    # crabs, crayfish, insects
    "Aves": (3, 0.16),              # herons, kingfishers, gulls
    "Mammalia": (40151, 0.08),      # otters, seals
    "Plantae": (47126, 0.12),       # weed, reeds - what you actually hook
    "Cnidaria": (47534, 0.06),      # jellyfish
    "Echinodermata": (47549, 0.06), # starfish, urchins
}


def log(m: str = "") -> None:
    print(m, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--memory-gb", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8,
                    help="DuckDB threads; lower this to avoid "
                         "starving a concurrent training run")
    args = ap.parse_args(argv)

    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_gb}GB'")
    tmp = PATHS.work / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")

    log(f"selecting {args.limit:,} non-fish negatives from "
        f"{len(NEGATIVE_CLADES)} clades")

    # Resolve each clade to its descendant taxa via the ancestry path.
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW taxa AS SELECT * FROM {inat.read_csv_sql('taxa')}"
    )
    clade_sql = []
    for name, (tid, share) in NEGATIVE_CLADES.items():
        clade_sql.append(
            f"SELECT taxon_id, '{name}' AS clade, {share} AS share FROM taxa "
            f"WHERE rank = 'species' AND active = 'true' "
            f"AND (ancestry = '{tid}' OR ancestry LIKE '%/{tid}' "
            f"     OR ancestry LIKE '%/{tid}/%')"
        )
    con.execute(
        "CREATE OR REPLACE TABLE neg_taxa AS " + " UNION ALL ".join(clade_sql)
    )
    log(f"  {con.execute('SELECT count(*) FROM neg_taxa').fetchone()[0]:,} "
        f"candidate taxa")

    # Sample down *during* the observation scan. The unfiltered join matches
    # ~151M research-grade observations - essentially all of iNaturalist - and
    # materialising that to pick 3,000 images makes the subsequent photo join
    # enormous for no benefit. Reservoir-style sampling on a hash keeps the
    # selection deterministic while bounding the build side of the join.
    per_clade_pool = max(2000, args.limit * 4)
    log(f"scanning observations.csv.gz (sampling {per_clade_pool:,} per clade)")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE neg_obs AS
        SELECT * FROM (
            SELECT o.observation_uuid, o.observer_id, o.taxon_id,
                   t.clade, t.share,
                   row_number() OVER (
                       PARTITION BY t.clade ORDER BY hash(o.observation_uuid)
                   ) AS rn
            FROM {inat.read_csv_sql('observations')} o
            JOIN neg_taxa t ON t.taxon_id = o.taxon_id
            WHERE o.quality_grade = 'research'
              AND o.latitude IS NOT NULL
        ) WHERE rn <= {per_clade_pool}
        """
    )
    log(f"  {con.execute('SELECT count(*) FROM neg_obs').fetchone()[0]:,} "
        f"research-grade observations ({time.time()-t0:.0f}s)")

    log("scanning photos.csv.gz")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE neg_photos AS
        SELECT p.photo_id, p.observation_uuid, p.observer_id, p.extension,
               p.license, o.taxon_id, o.clade, o.share
        FROM {inat.read_csv_sql('photos')} p
        JOIN neg_obs o ON o.observation_uuid = p.observation_uuid
        WHERE p.position = 0
          AND p.license IN ('CC0', 'CC-BY')
        """
    )
    total = con.execute("SELECT count(*) FROM neg_photos").fetchone()[0]
    log(f"  {total:,} licensed candidate photos ({time.time()-t0:.0f}s)")

    # Sample per clade to the requested share, one photo per observation and at
    # most one per photographer per clade so the set is not dominated by a few
    # prolific users.
    rows = con.execute(
        f"""
        WITH ranked AS (
            SELECT *, row_number() OVER (
                       PARTITION BY clade
                       ORDER BY hash(observer_id), hash(photo_id)
                   ) AS rn
            FROM (
                SELECT *, row_number() OVER (
                           PARTITION BY clade, observer_id ORDER BY hash(photo_id)
                       ) AS per_observer
                FROM neg_photos
            ) WHERE per_observer <= 3
        )
        SELECT photo_id, extension, license, clade, taxon_id
        FROM ranked
        WHERE rn <= CAST({args.limit} * share AS INTEGER)
        """
    ).fetchall()
    con.close()

    log(f"selected {len(rows):,} photos ({time.time()-t0:.0f}s)")
    by_clade: dict[str, int] = {}
    for _, _, _, clade, _ in rows:
        by_clade[clade] = by_clade.get(clade, 0) + 1
    log("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_clade.items())))

    # --- download ----------------------------------------------------------
    db = ProvenanceDB()
    store = ImageStore(db, PRODUCTION)

    provs = []
    for photo_id, ext, lic, clade, taxon_id in rows:
        provs.append(
            ImageProvenance(
                source_dataset="inaturalist-nonfish",
                source_record_id=str(photo_id),
                image_url=inat.photo_url(photo_id, ext or "jpg"),
                source_url=f"https://www.inaturalist.org/photos/{photo_id}",
                source_taxon_id=str(taxon_id),
                original_scientific_name="",
                # taxon_id stays NULL on purpose: the corpus builder selects on
                # species_taxon_id IS NOT NULL, so these can never be trained on.
                accepted_scientific_name=None,
                license=normalize(lic),
                license_raw=lic or "",
                ext=(ext or "jpg").lower(),
                context_tag=f"negative:{clade}",
                notes=f"open-set negative, clade={clade}",
            )
        )
    db.add_candidates(provs)
    log(f"registered {len(provs):,} negative candidates")

    done = {
        r[0] for r in db.con.execute("SELECT candidate_id FROM stored").fetchall()
    }
    provs = [
        p for p in provs
        if db.candidate_id(p.source_dataset, p.source_record_id) not in done
    ]

    from concurrent.futures import ThreadPoolExecutor
    import threading

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "bytes": 0}
    t1 = time.time()

    def one(prov: ImageProvenance) -> None:
        cid = db.candidate_id(prov.source_dataset, prov.source_record_id)
        try:
            data = net.fetch_bytes(prov.image_url)
            facts = measure(data)
        except (net.DownloadError, ImageRejected) as exc:
            db.mark_failure(cid, "permanent", str(exc)[:200])
            with lock:
                stats["fail"] += 1
            return
        try:
            store.put(data, prov, sha256=facts.sha256, width=facts.width,
                      height=facts.height, phash=facts.phash, dhash=facts.dhash)
        except Exception as exc:
            db.mark_failure(cid, "transient", str(exc)[:200])
            with lock:
                stats["fail"] += 1
            return
        with lock:
            stats["ok"] += 1
            stats["bytes"] += facts.nbytes
            n = stats["ok"] + stats["fail"]
            if n % 500 == 0:
                el = time.time() - t1
                log(f"  {n:,}/{len(provs):,} {n/max(el,1e-6):.0f} img/s "
                    f"{stats['bytes']/1e6:.0f} MB")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, provs))

    db.flush()
    db.close()
    log("")
    log(f"stored {stats['ok']:,} negatives ({stats['bytes']/1e6:.0f} MB), "
        f"{stats['fail']:,} failed, {time.time()-t0:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
