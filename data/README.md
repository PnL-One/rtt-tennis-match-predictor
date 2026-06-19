# Data Control Files

`tournaments_master.xlsx` is the local master list of RTT tournaments used by the match page parser.
It is intentionally separated from raw saved HTML pages:

- `tour_id` is the main deduplication key.
- new calendar downloads should be merged into this file before match pages are saved;
- `notebooks/01_save_and_parse_matches.ipynb` reads this file when it exists;
- the training dataset is still built into `assembled_predictor/predictor_model_dataset_from_parsers.xlsx`.

`data_manifest.json` is a generated status snapshot. Rebuild it with:

```bash
python scripts/data_status.py --write-manifest
```

Current manifest snapshot:

| Area | Current value |
| --- | ---: |
| Tournaments in master file | 501 |
| Saved match pages | 507 |
| Ranking rows | 232487 |
| Unique ranking RNI | 8686 |
| Final ML rows | 15764 |
| Completed matches | 7882 |
| Match date range | 2025-03-02 - 2026-06-19 |
