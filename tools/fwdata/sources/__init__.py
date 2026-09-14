"""Per-source acquisition adapters.

Each adapter is responsible for turning an *official bulk distribution* of a
data provider into rows of the common provenance schema.  Adapters must not
scrape HTML, must not use search engines, and must not page through a public
API when the provider publishes a bulk export.
"""

from __future__ import annotations

SOURCES = ("inaturalist", "gbif", "fishnet", "wikimedia")
