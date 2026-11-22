import argparse

import pandas as pd

import config as C
from evaluation import assign_folds, cross_validate, load_cohort, run_label, summarize
from models import searcher

ARM_NAMES = {"logistic": "A1_logistic", "gbm": "A1b_gbm"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="image_candidate", choices=["tabular", "image_candidate", "image"])
    ap.add_argument("--model", default="logistic", choices=list(ARM_NAMES))
    ap.add_argument("--holdout", default="fire", choices=["fire", "region"])
    args = ap.parse_args(argv)

    df = load_cohort(args.cohort)
    folds, status = assign_folds(df, args.holdout)
    chosen = []

    def record(k, search):
        info = dict(fold=k, **{key.replace("clf__", ""): v for key, v in search.best_params_.items()})
        if args.model == "logistic":
            info.update(zip(C.PREDICTOR_COLS, search.best_estimator_.named_steps["clf"].coef_[0]))
        chosen.append(info)

    preds = cross_validate(df, df[C.PREDICTOR_COLS].values, folds, searcher(args.model), record)
    summarize(preds, ARM_NAMES[args.model], status, run_label(args.cohort, args.holdout))
    print(f"\nChosen settings per fold:\n{pd.DataFrame(chosen).round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
