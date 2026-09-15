# V2 broad acquisition

The broad acquisition pass is intentionally separate from FisherWiki's release-corpus licence policies. It records source rights metadata when available, but does not use that metadata as an acquisition-time image filter.

## Full run

```powershell
cd C:\Dev\fisherwiki
.\.venv-train\Scripts\python.exe -u tools\acquire_v2.py --all
```

The command currently acquires from:

- the complete local iNaturalist candidate census, diversity-capped per species;
- non-iNaturalist GBIF occurrence media;
- Wikimedia Commons scientific-name categories;
- FathomNet scientific-name concept/box queries when the public service is available.

The default iNaturalist cap is 1,000 photos per species. Re-running later with a larger `--inat-cap` is safe. Newly fetched iNaturalist candidates use the 1024 px `/large` derivative; already stored historical `/medium` images are not silently replaced.

## Stop/resume contract

`Ctrl+C`, a reboot, loss of network, or a provider outage does not invalidate completed work:

- discovery checkpoints are persisted under `D:\fisherwiki-data\work\acquisition`;
- candidates are idempotent in DuckDB;
- downloads use deterministic staging paths and `.part` range-resume files;
- completed bytes enter the content-addressed store before temporary staging bytes are deleted;
- only permanent HTTP/decode failures are skipped permanently;
- transient failures are retried by the next invocation;
- old failures whose only reason was the former licence-policy gate are reopened automatically;
- V2 split assignment remains immutable and is verified before the command exits.

The command refuses to consume the final 50 GB of the data volume. `E:\FisherWiki\staging\acquisition` is used as hot staging when available.

## What it deliberately does **not** do

Acquisition stops after ingest and immutable split assignment. It never starts fish detection, cropping, derivative generation, shard preparation, a backbone pilot, or model training.

After acquisition completes, the existing big preparation entrypoint remains:

```powershell
.\.venv-train\Scripts\python.exe -u tools\v2_pipeline.py --stages detect,prepare,verify-shards
```

Do not run that command until acquisition is finished and the resulting corpus counts have been reviewed.

## Status

From another terminal:

```powershell
.\.venv-train\Scripts\python.exe tools\status.py --watch
```

The status command reads durable JSON/log state; it does not depend on the terminal or agent that started acquisition.

## Bounded smoke test

For a short source-integration smoke test without launching a large download:

```powershell
.\.venv-train\Scripts\python.exe -m pytest tests\test_acquisition_v2.py -q
.\.venv-train\Scripts\python.exe -u tools\acquire_v2.py --sources gbif-media,wikimedia-commons,fathomnet --taxa-limit 2 --discover-only
```

The second command performs metadata discovery only for two taxa per source and downloads no images.
