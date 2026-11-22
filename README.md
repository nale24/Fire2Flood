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
| `clean_inventory.py` | `inventory_clean.parquet`, `fire_summary.csv` |
| `download_assessments.py` | basin shapefiles in `data/raw/assessments/<AssessmentFireID>/` |
| `join_basins.py` | `basins.gpkg`, `join_report.csv` |
| `select_scenes.py` | `scene_manifest.parquet`, `basin_clear.parquet` |
| `cut_chips.py` | `chips/<fire>/<segment>_<pre\|post>.npz`, `chip_manifest.parquet`, `event_table.parquet` |
| `freeze_splits.py` | `splits.json`, `split_summary.csv` |

## Run

```
python src/fire2flood_pipeline/run_pipeline.py --fires thomas dolan grizzly_creek
python src/fire2flood_pipeline/run_pipeline.py
python src/fire2flood_pipeline/freeze_splits.py
```

`--from` and `--to` select a range of steps (`clean`, `download`, `join`, `scenes`, `chips`). `freeze_splits.py` refuses to overwrite an existing `splits.json` unless `--overwrite` is passed.

## Models

| Script | Arm | Output (in `reports/results/`) |
|---|---|---|
| `baseline_a1.py` | A1: logistic regression on PeakI15, HM23, dNBR/1000, KF | `A1_logistic_<cohort>_metrics.csv`, `_predictions.parquet` |
| `llm_baseline.py --mode fewshot` | GPT-2 few-shot prompting, yes/no log-odds, Platt-calibrated on training rows | `GPT2_fewshot_<cohort>_*` and the prompts used |
| `llm_baseline.py --mode embed` | GPT-2 mean-pooled embeddings + PCA + logistic regression | `GPT2_embed_<cohort>_*` |
| `llm_baseline.py --mode probe` | Memorization probe: score shift when fire name and date are added | `GPT2_memorization_probe_<cohort>.csv` |
| `baseline_a1.py --model gbm` | A1b: gradient boosting on the same four predictors | `A1b_gbm_<cohort>_*` |
| `prithvi_features.py` | Frozen Prithvi-EO-2.0-300M features, basin and context pooling | `data/processed/prithvi/<fire>.npz` |
| `model_a4.py` | A4: four predictors + PCA of Prithvi features (requires `reports/protocol.md`) | `A4_<head>_<pooling>_image_*` |
| `compare_arms.py` | Paired comparison of all arms vs A1 on identical rows | `comparison_<cohort>.csv` |

All arms share `evaluation.py`: folds come from `data/processed/splits.json` when it exists (otherwise provisional fire-grouped folds), and hyperparameters are tuned by inner fire-grouped CV on log-loss, and metrics are average precision, Brier score and threat score with fire-cluster bootstrap 95% intervals, with and without the Thomas fire. `--cohort` selects `tabular`, `image_candidate` (default) or `image`.

## Test

```
python src/fire2flood_pipeline/tests/test_offline.py
```

## Data sources

- Basin polygons: USGS Post-Wildfire Debris-Flow Hazard Assessments, 2013–2023 (https://doi.org/10.5066/P142R8E2) and 2024 (https://doi.org/10.5066/P13GTMAX).
- Imagery: NASA HLS v2.0, HLSL30 and HLSS30.
- Model: Prithvi-EO-2.0 (IBM/NASA).
