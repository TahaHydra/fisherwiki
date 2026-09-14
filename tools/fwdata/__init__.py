"""FisherWiki dataset tooling.

This package implements the data acquisition, provenance, licensing and
corpus-construction layer described in ``docs/DATASETS.md``.

Design rules enforced here:

* Every image that enters a corpus carries a full provenance row.  There is no
  code path that writes an image into content-addressed storage without also
  writing provenance.
* Licence strings are normalised into a closed vocabulary before any policy
  decision is made.  Unknown or unparseable licences are *never* silently
  treated as permissive.
* Downloads are resumable, rate-limited and checksum-verified.  No source is
  scraped; only official bulk endpoints are used.
"""

__version__ = "0.1.0"
