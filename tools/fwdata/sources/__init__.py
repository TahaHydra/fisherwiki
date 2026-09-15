"""Per-source acquisition adapters.

Prefer official bulk distributions when a provider publishes one.  Providers
that expose media only through a documented public API are paged conservatively
through the shared rate-limited HTTP layer.  HTML scraping is not used.
"""

from __future__ import annotations

SOURCES = (
    "inaturalist",
    "gbif",
    "gbif-media",
    "wikimedia-commons",
    "fathomnet",
    "fishnet",
)
