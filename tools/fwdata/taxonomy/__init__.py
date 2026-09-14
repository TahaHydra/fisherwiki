"""Canonical taxonomy: stable internal IDs, synonym reconciliation, names.

Folder names and source-provided IDs are deliberately *not* used as class
identities anywhere in this project. A species can be split, merged, moved
between genera or renamed, and any of those events would silently reassign
meaning to a model's output index if the class identity were a name or a
foreign ID.

Instead every taxon gets a ``fw_taxon_id`` from an append-only registry that is
committed to the repository. Model classes map to ``fw_taxon_id``; packs carry
that mapping; the mobile database joins on it.
"""

from __future__ import annotations

from .names import (  # noqa: F401
    NameParts,
    canonical_form,
    is_binomial,
    normalize_name,
    parse_scientific_name,
)
