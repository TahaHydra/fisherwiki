# Regulation rule packs

**Empty on purpose.** No jurisdiction data ships yet.

## Why

Fishing regulations are legally consequential and change frequently. Getting one
wrong can cost an angler a fine, or cost a fish its life. Two things must be
true before any rule ships here:

1. **The source must permit redistribution.** Many fisheries authorities publish
   regulations as Crown copyright, government works, or under bespoke terms that
   are neither clearly open nor clearly closed. That needs per-jurisdiction
   review, exactly as was done for image licensing in
   [`DATASET_LICENSES.md`](../../docs/DATASET_LICENSES.md).
2. **There must be a maintenance commitment.** A rule pack nobody updates is
   worse than no rule pack, because the app would be confidently telling
   somebody a limit that changed last season.

Until both hold, the app shows no regulations and says so. That is the correct
behaviour, not a gap.

## What is built

The schema, loader and validation are implemented and tested
([`tools/fwdata/regulations.py`](../../tools/fwdata/regulations.py),
[`tests/test_regulations.py`](../../tests/test_regulations.py)):

* every rule carries a jurisdiction, a source URL and a retrieval date;
* every rule carries `valid_from`, and `valid_until` where the source states one;
* numeric rules (size limits, bag limits) are rejected without a number, because
  a rule that cannot be applied is not a rule;
* `RuleSet.staleness()` reports the age in days and `is_stale()` flags anything
  over six months, so the UI can lead with "this copy may be out of date"
  rather than burying it;
* nothing is inferred — a jurisdiction with no data returns no rules, never a
  neighbouring jurisdiction's.

## Adding a jurisdiction

Copy `_template.yaml`, fill it in, and run:

```bash
.venv/Scripts/python -c "import sys; sys.path.insert(0,'tools'); \
  from fwdata.regulations import load; \
  rs = load('data/regulations/gb-eng.yaml'); \
  print(rs.jurisdiction, rs.version, len(rs.rules), 'rules', rs.staleness(), 'days old')"
```

The loader raises on anything malformed rather than skipping it.

A pull request adding a jurisdiction should state, in the description, what the
licence of the source is and who is committing to update it.
