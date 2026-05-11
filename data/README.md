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
