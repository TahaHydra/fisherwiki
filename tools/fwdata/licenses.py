"""Licence normalisation and corpus policy.

Why this module exists
----------------------
The project may eventually be distributed commercially.  That makes the licence
of every individual photograph a hard constraint, not a footnote.  Three
mistakes are easy to make and all three are guarded against here:

1. Confusing a *dataset/record* licence with the *media* licence.  A GBIF
   occurrence record may be CC0 while the attached photograph is
   All-Rights-Reserved.  Callers must pass the media licence.
2. Treating "no licence field" as permissive.  On iNaturalist an empty licence
   column means **All Rights Reserved**, not public domain.
3. Silently mixing NonCommercial or ShareAlike media into the production
   training corpus.

The vocabulary is closed: anything that does not parse becomes ``UNKNOWN`` and
is rejected by every policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class License(str, Enum):
    """Closed vocabulary of licences we are willing to reason about."""

    CC0 = "CC0-1.0"
    PD = "PUBLIC-DOMAIN"          # PD mark / expired copyright / US-Gov works
    CC_BY = "CC-BY-4.0"
    CC_BY_SA = "CC-BY-SA-4.0"
    CC_BY_NC = "CC-BY-NC-4.0"
    CC_BY_ND = "CC-BY-ND-4.0"
    CC_BY_NC_SA = "CC-BY-NC-SA-4.0"
    CC_BY_NC_ND = "CC-BY-NC-ND-4.0"
    ARR = "ALL-RIGHTS-RESERVED"
    UNKNOWN = "UNKNOWN"

    # --- capability predicates --------------------------------------------
    @property
    def allows_commercial(self) -> bool:
        return self in _COMMERCIAL_OK

    @property
    def allows_derivatives(self) -> bool:
        """Whether we permit this medium in a training corpus at all.

        Training a model on an image is an unsettled derivative-work question.
        We take the conservative position: NoDerivatives media never enters a
        training corpus, regardless of policy.
        """
        return self not in (
            License.CC_BY_ND,
            License.CC_BY_NC_ND,
            License.ARR,
            License.UNKNOWN,
        )

    @property
    def is_copyleft(self) -> bool:
        return self in (License.CC_BY_SA, License.CC_BY_NC_SA)

    @property
    def requires_attribution(self) -> bool:
        return self not in (License.CC0, License.PD, License.UNKNOWN, License.ARR)

    @property
    def url(self) -> str:
        return _LICENSE_URLS.get(self, "")


_COMMERCIAL_OK = frozenset(
    {License.CC0, License.PD, License.CC_BY, License.CC_BY_SA, License.CC_BY_ND}
)

_LICENSE_URLS = {
    License.CC0: "https://creativecommons.org/publicdomain/zero/1.0/",
    License.PD: "https://creativecommons.org/publicdomain/mark/1.0/",
    License.CC_BY: "https://creativecommons.org/licenses/by/4.0/",
    License.CC_BY_SA: "https://creativecommons.org/licenses/by-sa/4.0/",
    License.CC_BY_NC: "https://creativecommons.org/licenses/by-nc/4.0/",
    License.CC_BY_ND: "https://creativecommons.org/licenses/by-nd/4.0/",
    License.CC_BY_NC_SA: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    License.CC_BY_NC_ND: "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    License.ARR: "",
    License.UNKNOWN: "",
}

# Exact-match table for identifiers actually emitted by our sources.
# iNaturalist open data uses the short forms; GBIF uses full URIs; Wikimedia
# uses template names.
_EXACT: dict[str, License] = {
    "cc0": License.CC0,
    "cc-zero": License.CC0,
    "cc0-1.0": License.CC0,
    "http://creativecommons.org/publicdomain/zero/1.0/": License.CC0,
    "pd": License.PD,
    "pdm": License.PD,
    "public domain": License.PD,
    "publicdomain": License.PD,
    "http://creativecommons.org/publicdomain/mark/1.0/": License.PD,
    "cc-by": License.CC_BY,
    "cc-by-sa": License.CC_BY_SA,
    "cc-by-nc": License.CC_BY_NC,
    "cc-by-nd": License.CC_BY_ND,
    "cc-by-nc-sa": License.CC_BY_NC_SA,
    "cc-by-nc-nd": License.CC_BY_NC_ND,
    "c": License.ARR,
    "all rights reserved": License.ARR,
    "arr": License.ARR,
    "copyrighted": License.ARR,
    "": License.ARR,          # iNaturalist: empty licence means all rights reserved
    "none": License.ARR,
    "null": License.ARR,
}

# Our own canonical value strings must round-trip. The provenance store and the
# candidate Parquet persist `License.value`, so anything that reads a row back
# and re-normalises it has to land on the same member. Generated from the enum
# rather than hand-listed so a new member cannot be forgotten here.
_EXACT.update({lic.value.lower(): lic for lic in License})

# Structured CC URI parser, e.g.
#   http://creativecommons.org/licenses/by-nc-sa/4.0/  -> CC_BY_NC_SA
_CC_URI_RE = re.compile(
    r"creativecommons\.org/licenses/([a-z\-]+)(?:/([\d.]+))?", re.IGNORECASE
)

# Short-form parser, e.g. "CC BY-SA 3.0". Deliberately *strict*: every token
# after the "cc" prefix must be a recognised clause or a version number.
# A permissive regex here would let "CC-BY-MAYBE" resolve to plain CC BY, which
# is a fail-*open* on exactly the input we cannot afford to guess about.
_CC_SHORT_RE = re.compile(r"^cc[\s_/-]*(.+)$", re.IGNORECASE)
_VERSION_RE = re.compile(r"^\d(?:\.\d+)?$")
_KNOWN_CLAUSES = {"by", "nc", "sa", "nd"}

_CLAUSE_TO_LICENSE = {
    ("by",): License.CC_BY,
    ("by", "sa"): License.CC_BY_SA,
    ("by", "nc"): License.CC_BY_NC,
    ("by", "nd"): License.CC_BY_ND,
    ("by", "nc", "sa"): License.CC_BY_NC_SA,
    ("by", "nc", "nd"): License.CC_BY_NC_ND,
}


def normalize(raw: str | None) -> License:
    """Map a source-supplied licence identifier onto the closed vocabulary.

    Returns :attr:`License.UNKNOWN` when the string cannot be confidently
    resolved.  Never guesses in the permissive direction.
    """
    if raw is None:
        return License.ARR
    # Idempotence guard. `License` subclasses `str`, so callers can and do pass
    # an already-normalised member here. Falling through would stringify it to
    # 'License.CC_BY' on Python 3.11+ and resolve to UNKNOWN.
    if isinstance(raw, License):
        return raw
    s = str(raw).strip().lower().rstrip(".")
    if s in _EXACT:
        return _EXACT[s]
    s_url = s.replace("https://", "http://")
    if s_url in _EXACT:
        return _EXACT[s_url]

    m = _CC_URI_RE.search(s)
    if m:
        return _clauses_to_license(re.split(r"[\s_-]+", m.group(1).lower()))

    m = _CC_SHORT_RE.match(s)
    if m:
        tokens = [t for t in re.split(r"[\s_/-]+", m.group(1).lower()) if t]
        # Trailing version number is informational; drop it, reject the rest.
        clauses = [t for t in tokens if not _VERSION_RE.match(t)]
        return _clauses_to_license(clauses)

    if "publicdomain" in s or "public domain" in s or "pd-" in s:
        return License.PD
    return License.UNKNOWN


def _clauses_to_license(clauses: list[str]) -> License:
    """Resolve CC clause tokens, refusing anything with an unknown clause."""
    clauses = [c for c in clauses if c]
    if not clauses or any(c not in _KNOWN_CLAUSES for c in clauses):
        return License.UNKNOWN
    # Order-independent: "by-sa" and "sa-by" denote the same licence.
    key = tuple(
        c for c in ("by", "nc", "sa", "nd") if c in clauses
    )
    return _CLAUSE_TO_LICENSE.get(key, License.UNKNOWN)


@dataclass(frozen=True)
class Policy:
    """A named set of licences admissible for one corpus."""

    name: str
    allowed: frozenset[License]
    description: str
    commercial_safe: bool

    def admits(self, lic: License) -> bool:
        return lic in self.allowed

    def filter(self, licenses: Iterable[License]) -> list[License]:
        return [x for x in licenses if self.admits(x)]

    def sql_in_list(self) -> str:
        """Render the allowed set as a SQL IN-list literal."""
        vals = ", ".join(f"'{x.value}'" for x in sorted(self.allowed, key=lambda v: v.value))
        return f"({vals})"


#: Default corpus used to train the models we intend to ship.
PRODUCTION = Policy(
    name="production",
    allowed=frozenset({License.CC0, License.PD, License.CC_BY}),
    description=(
        "Commercially redistributable, attribution-only: CC0 / Public Domain / "
        "CC BY. Default corpus for any model shipped inside a pack."
    ),
    commercial_safe=True,
)

#: CC BY-SA kept separate: see docs/DATASET_LICENSES.md for the analysis of
#: whether trained weights constitute an adapted work.  Not enabled by default.
PRODUCTION_SA = Policy(
    name="production_sa",
    allowed=frozenset({License.CC0, License.PD, License.CC_BY, License.CC_BY_SA}),
    description=(
        "Production set plus CC BY-SA. Enabling this may impose ShareAlike "
        "obligations on derived model weights. Opt-in only."
    ),
    commercial_safe=True,
)

#: Research-only corpus. Never used for a shipped model.
RESEARCH_NC = Policy(
    name="research_nc",
    allowed=frozenset(
        {
            License.CC0,
            License.PD,
            License.CC_BY,
            License.CC_BY_SA,
            License.CC_BY_NC,
            License.CC_BY_NC_SA,
        }
    ),
    description=(
        "Adds NonCommercial media. Research and ablation only; a model trained "
        "on this corpus must never be distributed in a release pack."
    ),
    commercial_safe=False,
)

POLICIES: dict[str, Policy] = {p.name: p for p in (PRODUCTION, PRODUCTION_SA, RESEARCH_NC)}


def get_policy(name: str) -> Policy:
    try:
        return POLICIES[name]
    except KeyError:
        raise SystemExit(
            f"Unknown licence policy {name!r}. Available: {sorted(POLICIES)}"
        )


def attribution_string(creator: str | None, lic: License, source: str) -> str:
    """Human-readable attribution line as required by the CC licences."""
    who = (creator or "").strip() or "unknown"
    if lic is License.CC0:
        return f"{who}, no rights reserved (CC0), via {source}"
    if lic is License.PD:
        return f"{who}, public domain, via {source}"
    if lic.requires_attribution:
        short = lic.value.rsplit("-", 1)[0]
        return f"(c) {who}, some rights reserved ({short}), via {source}"
    return f"(c) {who}, all rights reserved, via {source}"
