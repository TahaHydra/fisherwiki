"""Confirm every local link/reference in the docs actually resolves.

    python scripts/check_doc_links.py

This exists because it already caught two real gaps in this project: a
`tools/taxonomy.py` three documents referenced but that did not exist, and a
`docs/SPECIES_DATA.md` two other documents linked to before it was written.
Both were the specific failure mode this whole project's brief warns against -
documentation describing something that was never actually built - and both
were found by exactly this check, run by hand at the time. This is that check,
kept, so it runs on every push instead of only when someone happens to think
of it again.

Checks two things across `README.md`, `task.md`, `CONTRIBUTING.md` and every
file under `docs/`:

1. Markdown links `[text](path)` to a local file resolve to a real path
   relative to the linking document.
2. Inline code spans that look like a repo-relative path to a source file
   (``tools/fwdata/x.py``, ``app/core/.../Y.kt``, ...) resolve too - this is
   what catches "referenced before it exists" for files nobody got around to
   hyperlinking properly.

Deliberately conservative about (2): it only fires on paths containing a `/`
and ending in a small set of source/config extensions, specifically to avoid
flagging a code identifier or a generic filename mentioned in prose.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DOC_FILES = (
    [REPO / "README.md", REPO / "task.md", REPO / "CONTRIBUTING.md"]
    + sorted((REPO / "docs").glob("*.md"))
)

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#][^)]*)\)")
CODE_PATH_RE = re.compile(
    r"`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\."
    r"(?:py|kt|kts|sql|yaml|yml|json|toml|md|gradle|sh|ps1))`"
)


def find_broken(doc: Path) -> list[tuple[str, str]]:
    """[(kind, raw_reference), ...] for references in `doc` that do not exist."""
    text = doc.read_text(encoding="utf-8")
    broken: list[tuple[str, str]] = []

    seen: set[str] = set()
    link_spans: list[tuple[int, int]] = []
    for m in LINK_RE.finditer(text):
        link_spans.append(m.span())  # the whole [text](href), not just href
        target = m.group(1).split("#")[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if target in seen:
            continue
        seen.add(target)
        if not (doc.parent / target).exists():
            broken.append(("link", target))

    for m in CODE_PATH_RE.finditer(text):
        # A code span used as a markdown link's *link text* - `` [`x/y.py`](../x/y.py) ``
        # - is not an independent reference; it was already checked above
        # against its own href, which may legitimately differ (a relative
        # path vs. the code span's own shorthand). Only a code span that
        # stands on its own, outside any [...](...), is a second reference.
        if any(start <= m.start() and m.end() <= end for start, end in link_spans):
            continue
        target = m.group(1)
        if target in seen or target.startswith(("http:", "https:")):
            continue
        seen.add(target)
        # Code-span paths in these docs are always repo-relative, not
        # relative to the linking document - that convention is what makes
        # `tools/fwdata/x.py` mean the same thing whichever doc it appears in.
        if not (REPO / target).exists():
            broken.append(("code", target))

    return broken


def main(argv: list[str] | None = None) -> int:
    problems: list[tuple[Path, str, str]] = []
    for doc in DOC_FILES:
        if not doc.exists():
            continue
        for kind, target in find_broken(doc):
            problems.append((doc, kind, target))

    if problems:
        print(f"{len(problems)} unresolved reference(s):\n")
        for doc, kind, target in problems:
            print(f"  {doc.relative_to(REPO)!s:40} [{kind:4}] {target}")
        print(
            "\nEither the file has not been written yet (write it, or stop "
            "referencing it), or the path/anchor is wrong."
        )
        return 1

    print(f"checked {len(DOC_FILES)} documents, all local references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
