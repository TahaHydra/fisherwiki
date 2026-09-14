"""Scientific name parsing and normalisation.

Matching names across GBIF, iNaturalist and Wikidata fails in boring, specific
ways, and each one is handled explicitly here rather than by a fuzzy matcher:

* trailing authorship - ``Perca fluviatilis Linnaeus, 1758``
* parenthesised authorship - ``Salmo trutta (Linnaeus, 1758)``
* subgenus in the middle - ``Salmo (Salmo) trutta``
* hybrid markers - ``Salmo x trutta``, ``Salmo trutta × salar``
* open nomenclature - ``Sebastes sp.``, ``Sebastes cf. norvegicus``,
  ``Sebastes aff. marinus``
* infraspecific ranks - ``Salvelinus alpinus subsp. erythrinus``,
  ``... var. x``, ``... f. y``
* unicode - non-breaking spaces, multiplication sign, curly apostrophes,
  accented authorship
* casing - ``PERCA FLUVIATILIS``, ``perca Fluviatilis``

Rules applied, in order: strip diacritics and odd whitespace, drop authorship,
drop subgenus, normalise hybrid/uncertainty markers, collapse infraspecific
rank abbreviations, then Capitalise-genus/lowercase-epithet.

The output of :func:`canonical_form` is what we join on. It is *not* a display
name - display names come from the canonical accepted record.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Rank abbreviations that may appear before an infraspecific epithet.
INFRA_MARKERS = {
    "subsp": "subspecies",
    "ssp": "subspecies",
    "subspecies": "subspecies",
    "var": "variety",
    "variety": "variety",
    "f": "form",
    "fo": "form",
    "form": "form",
    "forma": "form",
    "cv": "cultivar",
    "morph": "morph",
}

#: Markers of provisional / uncertain identification (open nomenclature).
UNCERTAIN_MARKERS = {"cf", "cf.", "aff", "aff.", "sp", "sp.", "spp", "spp.", "nr", "nr."}

_AUTHOR_PAREN = re.compile(r"\([^)]*\d{4}[^)]*\)")          # (Linnaeus, 1758)
_SUBGENUS = re.compile(r"\(\s*[A-Z][a-z\-]+\s*\)")           # (Salmo)
_YEAR_TAIL = re.compile(r",?\s*\b\d{4}\b\s*$")
_MULTI_WS = re.compile(r"\s+")
_NON_NAME = re.compile(r"[^A-Za-z\s\.\-×x]")


def _strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_name(raw: str) -> str:
    """Whitespace/unicode cleanup that is safe to apply to any name string."""
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", str(raw))
    s = s.replace(" ", " ").replace("’", "'").replace("‘", "'")
    s = s.replace("×", " x ")          # multiplication sign -> hybrid marker
    s = _strip_diacritics(s)
    return _MULTI_WS.sub(" ", s).strip()


@dataclass(frozen=True)
class NameParts:
    """Structured view of a parsed scientific name."""

    genus: str = ""
    specific_epithet: str = ""
    infraspecific_epithet: str = ""
    infraspecific_rank: str = ""
    authorship: str = ""
    hybrid: bool = False
    uncertain: bool = False
    #: Rank implied by the shape of the name.
    rank: str = "unknown"

    @property
    def canonical(self) -> str:
        """Canonical join key: genus + epithets, no authorship, no markers."""
        bits = [self.genus]
        if self.specific_epithet:
            bits.append(self.specific_epithet)
        if self.infraspecific_epithet:
            bits.append(self.infraspecific_epithet)
        return " ".join(b for b in bits if b)

    @property
    def binomial(self) -> str:
        """Species-level name, discarding any infraspecific part."""
        if self.genus and self.specific_epithet:
            return f"{self.genus} {self.specific_epithet}"
        return self.genus


def parse_scientific_name(raw: str) -> NameParts:
    """Parse a scientific name into parts, tolerating real-world messiness."""
    s = normalize_name(raw)
    if not s:
        return NameParts()

    # Authorship: capture then remove. Handles both "(L., 1758)" and "L., 1758".
    authorship = ""
    m = _AUTHOR_PAREN.search(s)
    if m:
        authorship = m.group(0)
        s = s.replace(authorship, " ")
    else:
        m2 = _YEAR_TAIL.search(s)
        if m2:
            # Walk back over the author words preceding the year.
            head = s[: m2.start()].rstrip(" ,")
            toks = head.split(" ")
            keep: list[str] = []
            author_toks: list[str] = []
            for i, t in enumerate(toks):
                # An author token is capitalised and appears after >=2 name tokens.
                if i >= 2 and (t[:1].isupper() or t == "&" or t.endswith(".")):
                    author_toks = toks[i:]
                    break
                keep.append(t)
            if author_toks:
                authorship = " ".join(author_toks) + s[m2.start():]
                s = " ".join(keep)
            else:
                s = head

    s = _SUBGENUS.sub(" ", s)
    s = _MULTI_WS.sub(" ", s).strip()

    tokens = [t for t in s.split(" ") if t]
    hybrid = False
    uncertain = False
    cleaned: list[str] = []
    infra_rank = ""

    for tok in tokens:
        low = tok.lower().strip(".")
        if low in ("x", "×") and cleaned:
            hybrid = True
            continue
        if low in {m.strip(".") for m in UNCERTAIN_MARKERS}:
            uncertain = True
            continue
        if low in INFRA_MARKERS and cleaned:
            infra_rank = INFRA_MARKERS[low]
            continue
        # Aquarium/trade placeholder codes ("L001", "C-121", "sp. 4") are not
        # epithets. Stripping their digits would leave a bogus one-letter
        # epithet, so drop any token that contained a digit outright.
        if any(ch.isdigit() for ch in tok):
            uncertain = True
            continue
        tok = _NON_NAME.sub("", tok)
        # A real epithet is at least three letters; shorter leftovers are
        # punctuation debris rather than names.
        if tok and (len(tok) >= 3 or not cleaned):
            cleaned.append(tok)

    if not cleaned:
        return NameParts(authorship=authorship, hybrid=hybrid, uncertain=uncertain)

    genus = cleaned[0].capitalize()
    specific = cleaned[1].lower() if len(cleaned) > 1 else ""
    infra = cleaned[2].lower() if len(cleaned) > 2 else ""

    if infra:
        rank = infra_rank or "subspecies"
    elif specific:
        rank = "species"
    else:
        rank = "genus"

    return NameParts(
        genus=genus,
        specific_epithet=specific,
        infraspecific_epithet=infra,
        infraspecific_rank=infra_rank,
        authorship=authorship.strip(),
        hybrid=hybrid,
        uncertain=uncertain,
        rank=rank,
    )


def canonical_form(raw: str) -> str:
    """Canonical join key for a scientific name (``''`` if unparseable)."""
    return parse_scientific_name(raw).canonical


def is_binomial(raw: str) -> bool:
    p = parse_scientific_name(raw)
    return bool(p.genus and p.specific_epithet)
