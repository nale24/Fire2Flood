# Fire2Flood evaluation protocol

Status: DRAFT. Rename to `protocol.md` and commit it before running `model_a4.py`. Once committed, do not edit the rule after seeing any image-feature result.

## Data and splits

- Splits frozen 2026-11-21 00:35 (`data/processed/splits.json`, seed 2026).
- Image cohort: fires ignited on or after 2013-06-01, storm at least 11 days after ignition, pre- and post-fire HLS chips at least 80 % clear inside the basin mask. 5,918 test rows with 489 debris flows after removing the two development groups (`bot2018`, `rvr2020`).
- Five outer folds grouped by assessment (merged when two assessments share a rain gauge and storm date). Fold sizes: 9, 5, 6, 13 and 6 groups. The dominant group (`thm2017`, Thomas) is in fold 4.
- Basin masks: real USGS polygon where it matches and passes the area check (0.5 to 2 times `UpstreamArea`), otherwise an equal-area circle around the observation point.

## Arms

| Arm | Inputs | Head |
|---|---|---|
| A1_logistic (reference) | PeakI15, HM23, dNBR/1000, KF | Logistic regression |
| A1b_gbm | Same four | Gradient boosting |
| GPT2_embed | Same four, as an anonymized sentence, GPT-2 embeddings, PCA 32 | Logistic regression |
| GPT2_fewshot | Same four, as an anonymized sentence, 8-shot prompt, Platt-calibrated | None (log-odds of " yes" vs " no") |
| **A4_logistic_basin (primary)** | Same four + frozen Prithvi-EO-2.0-300M features (pre, post, post minus pre), basin-weighted token pooling, PCA 16 fitted on training folds | Logistic regression |
| A4 variants | Context pooling (3 x 3 tokens), gradient-boosting head, polygon-only rows | As named |

Hyperparameters are tuned only by inner fire-grouped cross-validation on training folds, scored by log-loss (changed from average precision on 2026-11-21, before any image-feature result existed, because AP cannot choose between regularization strengths and picked over-shrunk probabilities).

## Metrics

Primary: pooled average precision over all test rows. Also reported: mean per-fold AP, Brier score, threat score at a threshold chosen on training data. All with fire-cluster bootstrap 95 % intervals (2,000 resamples of whole split groups), for all fires and without Thomas.

## Success rule (primary hypothesis: Prithvi features add information beyond the four predictors)

A4_logistic_basin counts as an improvement over A1_logistic only if all three hold on the same rows:

1. Pooled AP difference (A4 minus A1) is above 0, and its 95 % interval excludes 0.
2. Brier score difference (A4 minus A1) has an upper 95 % bound of at most +0.005.
3. The AP difference is still above 0 without Thomas.

If 1 fails, the conclusion is "no measurable gain from frozen Prithvi features under fire-held-out evaluation". If 1 holds but 3 fails, the gain is attributed to the Thomas fire and reported as not transferable.

Secondary arms (context pooling, gradient-boosting head, polygon-only, GPT-2) are descriptive. They cannot turn a failed primary comparison into a success.

## Failure analysis (reported regardless of outcome)

Performance by basin size (`tokens_touched`), rainfall intensity, burn severity, days since fire, sensor pair (L30/S30), mask source (polygon vs proxy) and per fire.
