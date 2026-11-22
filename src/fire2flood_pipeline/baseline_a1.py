import argparse

import pandas as pd

import config as C
from evaluation import run_label, assign_folds, best_threshold, load_cohort, prediction_frame, summarize
from models import make_search

ARM_NAMES = {"logistic": "A1_logistic", "gbm": "A1b_gbm"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="image_candidate", choices=["tabular", "image_candidate", "image"])
    ap.add_argument("--model", default="logistic", choices=["logistic", "gbm"])
    ap.add_argument("--holdout", default="fire", choices=["fire", "region"])
    args = ap.parse_args(argv)

    df = load_cohort(args.cohort)
    folds, status = assign_folds(df, args.holdout)
    parts, chosen = [], []
    for k in sorted(f for f in folds.unique() if f >= 0):
        train, test = df[(folds != k) & (folds >= 0)], folds == k
        search = make_search(args.model, train["split_group"].values)
        search.fit(train[C.PREDICTOR_COLS].values, train["response"].values, groups=train["split_group"].values)
        t = best_threshold(train["response"].values, search.predict_proba(train[C.PREDICTOR_COLS].values)[:, 1])
        p = search.predict_proba(df.loc[test, C.PREDICTOR_COLS].values)[:, 1]
        parts.append(prediction_frame(df, test, p, t, k))
        info = dict(fold=k, **{key.replace("clf__", ""): v for key, v in search.best_params_.items()})
        clf = search.best_estimator_.named_steps["clf"]
        if args.model == "logistic":
            info.update(dict(zip(C.PREDICTOR_COLS, clf.coef_[0])))
        chosen.append(info)
    summarize(pd.concat(parts, ignore_index=True), ARM_NAMES[args.model], status, run_label(args.cohort, args.holdout))
    print("\nChosen settings per fold:")
    print(pd.DataFrame(chosen).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
