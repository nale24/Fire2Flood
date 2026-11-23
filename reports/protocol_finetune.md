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
