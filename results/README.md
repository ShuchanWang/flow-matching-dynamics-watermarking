# Compact Experimental Evidence

Generated with `python tools/export_run_summaries.py`. These snapshots are
intended for reviewed Git sync, unlike the full local `table_runs/` directory.

Folders group evidence by purpose, not by whether the outcome was favorable.
`pilots/` includes unsuccessful carrier/gating studies. Their structured metrics
and configurations are retained; a log alone is insufficient evidence.

`manifest.json` records source paths, raw-source hashes and exported-file hashes.
Large per-trial scores/decoded messages and selected credential fields are omitted.
Training seeds and per-trial exact/bit outcomes remain. Owner-key tensor files,
checkpoints, features, images and logs are never copied by this tool.
The 10,000-trial fresh-key audit stores its outcomes as CSV rows rather than a
large nested JSON array; its JSON metadata points to that CSV.

Explicitly unfinished studies are skipped. `state: unspecified` means the source
has no completion marker; consult that study's records before interpreting it.
Full raw artifacts must be archived separately before any local cleanup.
Review snapshots before publishing; redaction here is intentionally limited.
