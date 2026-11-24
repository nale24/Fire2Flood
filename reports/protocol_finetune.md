# Fine-tuning protocol (written 2026-11-23, before any fine-tuning result)

This is a secondary, exploratory analysis. The primary result (frozen Prithvi, `protocol.md`) stands unchanged.

## Question
Does adapting Prithvi-EO-2.0-300M to the debris-flow task, instead of using it frozen, add skill over the
four-predictor logistic baseline (A1) on fires held out from training?

## Fixed design (no changes after seeing test-fold results)
- Cohort, folds and split groups: identical to the primary analysis (`image` cohort, frozen `splits.json`).
- Model: shared Prithvi encoder applied to the pre-fire and post-fire chips; basin-weighted token pooling;
  head = linear term on the four standardized predictors + a small image network on [post, post - pre],
  with the image output initialized to zero so training starts from the logistic baseline.
- Primary configuration: last 2 of 24 transformer blocks unfrozen (`--unfreeze 2`), encoder learning rate 1e-5,
  head learning rate 1e-3, weight decay 0.05, batch 32, at most 15 epochs, early stopping after 3 epochs without
  improvement in validation log-loss, random 90-degree rotations and flips, 3 seeds averaged.
- Validation: 20% of the training fire groups in each fold, never the test fold.
- Control: the same pipeline with `--unfreeze 0` (head trained with augmentation, encoder frozen), to separate the
  effect of adapting the encoder from the effect of the new head and augmentation.

## Success rule (same as the primary rule)
The primary configuration "helps" only if, versus A1 on the same test rows:
1. the 95% fire-cluster bootstrap interval of the AP gain is above zero,
2. Brier score is no worse than A1 + 0.005, and
3. the AP gain stays above zero when the Thomas Fire is removed.

## Reported regardless of outcome
AP, Brier score and threat score with intervals; the control; the regional hold-out of the primary configuration;
training curves (validation log-loss per epoch); total GPU time.

## Addendum, 2026-11-23 (after the first run)
The first run did not implement the design above. The linear term on the four predictors started from random
weights instead of the fitted logistic baseline, so training did not start from A1. This was found because the
frozen control (`--unfreeze 0`), which should do at least as well as A1, scored far below it (AP 0.098 vs 0.348;
fine-tuned 0.136), and its predictions had a rank correlation of only 0.13 with A1's.
Fix: in every fold, the linear term now starts from a logistic regression fitted (same search as A1) on that fold's
training rows, excluding the validation fires. Nothing else changed: same folds, hyperparameters, seeds and success
rule. The first run's results are kept and reported as a failed implementation, not as evidence about Prithvi.
The attention probe (`model_probe.py`) had the same flaw and is rerun with the same fix.
