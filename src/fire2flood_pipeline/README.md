# Fire2Flood data pipeline

Builds the basin–storm event table and Prithvi-EO-2.0 input chips from the USGS runoff-generated postfire debris-flow inventory (Graber et al., 2026, https://doi.org/10.5066/P1R9VXC4).

## Setup

```
pip install -r requirements.txt
```

Place `USGS_RG_PFDF_Inventory_v1.csv` in `data/raw/`. Steps 03–04 require a NASA Earthdata account.

## Steps

| Script | Output (in `data/processed/`) |
|---|---|
| `01_clean_inventory.py` | `inventory_clean.parquet`, `fire_summary.csv` |
| `00_download_assessments.py` | basin shapefiles in `data/raw/assessments/<AssessmentFireID>/` |
| `02_join_basins.py` | `basins.gpkg`, `join_report.csv` |
| `03_select_scenes.py` | `scene_manifest.parquet`, `basin_clear.parquet` |
| `04_cut_chips.py` | `chips/<fire>/<segment>_<pre\|post>.npz`, `chip_manifest.parquet`, `event_table.parquet` |
| `05_freeze_splits.py` | `splits.json`, `split_summary.csv` |

## Run

```
python src/fire2flood_pipeline/run_pipeline.py --fires thomas dolan grizzly_creek
python src/fire2flood_pipeline/run_pipeline.py
python src/fire2flood_pipeline/05_freeze_splits.py
```

`--from` and `--to` select a range of steps (`clean`, `download`, `join`, `scenes`, `chips`). `05_freeze_splits.py` refuses to overwrite an existing `splits.json` unless `--overwrite` is passed.

## Test

```
python src/fire2flood_pipeline/tests/test_offline.py
```

## Data sources

- Basin polygons: USGS Post-Wildfire Debris-Flow Hazard Assessments, 2013–2023 (https://doi.org/10.5066/P142R8E2) and 2024 (https://doi.org/10.5066/P13GTMAX).
- Imagery: NASA HLS v2.0, HLSL30 and HLSS30.
- Model: Prithvi-EO-2.0 (IBM/NASA).
