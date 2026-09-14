"""Geographic regions used for pack scoping and for the on-device geo prior.

Two different jobs, deliberately kept apart:

**Pack scoping** asks "which species belong in the Europe Freshwater pack?".
That is answered empirically - a species is in a region if enough of its
*observations* fall inside the region - not from a hand-written species list.
Bounding boxes are good enough for this because the question is coarse.

**The on-device geo prior** must be finer than a bounding box, so it does not
use these shapes at all. It uses a per-species occurrence histogram over equal
-area grid cells built from real coordinates (see :mod:`fwdata.geoprior`).
A bounding box would tell the app that a brown trout is plausible in the middle
of the Sahara because the box covering Europe includes North Africa.

Boxes are ``(lat_min, lat_max, lon_min, lon_max)`` in WGS84 degrees. Antimeridian
crossings are expressed as two boxes rather than wrapped longitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

WaterType = Literal["freshwater", "marine", "brackish", "any"]


@dataclass(frozen=True)
class Region:
    """A candidate pack scope."""

    id: str
    name: str
    boxes: tuple[tuple[float, float, float, float], ...]
    water: WaterType = "any"
    #: Short note on why the boundary is drawn where it is.
    rationale: str = ""

    def contains(self, lat: float | None, lon: float | None) -> bool:
        if lat is None or lon is None:
            return False
        for la0, la1, lo0, lo1 in self.boxes:
            if la0 <= lat <= la1 and lo0 <= lon <= lo1:
                return True
        return False

    def sql_predicate(self, lat_col: str = "latitude", lon_col: str = "longitude") -> str:
        """SQL that is true when a row's coordinates fall inside this region."""
        parts = [
            f"({lat_col} BETWEEN {la0} AND {la1} AND {lon_col} BETWEEN {lo0} AND {lo1})"
            for la0, la1, lo0, lo1 in self.boxes
        ]
        return "(" + " OR ".join(parts) + ")"


#: Candidate regions. These are *proposals*; `fwdata.planning` measures how much
#: licensed training data each actually has and the pack split is decided from
#: that, per the project brief.
REGIONS: dict[str, Region] = {
    r.id: r
    for r in [
        Region(
            id="europe_freshwater",
            name="Europe - freshwater",
            boxes=((35.0, 71.5, -11.0, 40.0),),
            water="freshwater",
            rationale=(
                "Continental Europe plus Britain/Ireland and Scandinavia. "
                "Eastern edge at 40E keeps the Caspian basin out, whose fauna "
                "differs enough to belong with a separate pack."
            ),
        ),
        Region(
            id="europe_atlantic",
            name="Europe - Atlantic & North Sea",
            boxes=((35.0, 71.5, -25.0, 13.0),),
            water="marine",
            rationale=(
                "NE Atlantic shelf: Bay of Biscay, Celtic Sea, North Sea, "
                "Norwegian coast. Excludes the Mediterranean, which has a "
                "substantially different species set."
            ),
        ),
        Region(
            id="mediterranean",
            name="Mediterranean & Black Sea",
            boxes=((30.0, 47.5, -6.0, 42.0),),
            water="marine",
            rationale=(
                "Semi-enclosed basin with high endemism and heavy Lessepsian "
                "immigration; kept separate from the Atlantic deliberately."
            ),
        ),
        Region(
            id="north_america_freshwater",
            name="North America - freshwater",
            boxes=((24.0, 72.0, -170.0, -52.0),),
            water="freshwater",
            rationale=(
                "Canada, USA and northern Mexico. The largest and best "
                "photographed freshwater angling fauna on iNaturalist."
            ),
        ),
        Region(
            id="north_america_atlantic",
            name="North America - Atlantic & Gulf",
            boxes=((8.0, 60.0, -98.0, -52.0),),
            water="marine",
            rationale="US/Canadian east coast plus the Gulf of Mexico and Caribbean approaches.",
        ),
        Region(
            id="north_america_pacific",
            name="North America - Pacific",
            boxes=((22.0, 62.0, -170.0, -105.0),),
            water="marine",
            rationale="Baja through Alaska; a cold-temperate fauna quite unlike the Atlantic coast.",
        ),
        Region(
            id="australia_nz",
            name="Australia & New Zealand",
            boxes=((-50.0, -8.0, 108.0, 180.0), (-50.0, -8.0, -180.0, -175.0)),
            water="any",
            rationale=(
                "Two boxes because New Zealand's Chatham Islands cross the "
                "antimeridian. High endemism justifies its own pack."
            ),
        ),
        Region(
            id="indo_pacific",
            name="Indo-Pacific (tropical)",
            boxes=((-30.0, 30.0, 30.0, 180.0), (-30.0, 30.0, -180.0, -120.0)),
            water="marine",
            rationale=(
                "Coral Triangle and surrounding tropical seas. Enormous species "
                "richness; expect this pack to need genus/family fallbacks."
            ),
        ),
        Region(
            id="south_america",
            name="South America",
            boxes=((-56.0, 13.0, -82.0, -34.0),),
            water="any",
            rationale="Amazon/Orinoco/Parana basins plus both coasts.",
        ),
        Region(
            id="africa",
            name="Africa",
            boxes=((-35.0, 37.0, -18.0, 52.0),),
            water="any",
            rationale="Continental Africa including the Rift Valley lakes.",
        ),
    ]
}


def region(region_id: str) -> Region:
    try:
        return REGIONS[region_id]
    except KeyError:
        raise SystemExit(f"unknown region {region_id!r}; have {sorted(REGIONS)}")


def regions_for_point(lat: float, lon: float) -> list[Region]:
    """All regions whose box contains a point (regions deliberately overlap)."""
    return [r for r in REGIONS.values() if r.contains(lat, lon)]


def all_ids() -> list[str]:
    return sorted(REGIONS)


def as_dicts(ids: Iterable[str] | None = None) -> list[dict]:
    """Serialisable form, embedded into pack manifests."""
    ids = list(ids or all_ids())
    return [
        {
            "id": REGIONS[i].id,
            "name": REGIONS[i].name,
            "water": REGIONS[i].water,
            "boxes": [list(b) for b in REGIONS[i].boxes],
            "rationale": REGIONS[i].rationale,
        }
        for i in ids
    ]
