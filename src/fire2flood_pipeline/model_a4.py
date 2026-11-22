import argparse

import numpy as np
import pandas as pd

import config as C
from evaluation import run_label, assign_folds, best_threshold, load_cohort, prediction_frame, summarize
from models import make_search
from prithvi_features import load_features

PROTOCOL = C.ROOT / "reports" / "protocol.md"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="basin", choices=["basin", "context"])
    ap.add_argument("--head", default="logistic", choices=["logistic", "gbm"])
    ap.add_argument("--pca", type=int, default=16)
    ap.add_argument("--polygon-only", action="store_true")
    ap.add_argument("--holdout", default="fire", choices=["fire", "region"])
    args = ap.parse_args(argv)

    if not PROTOCOL.exists():
        raise SystemExit(f"{PROTOCOL} not found. Commit the success rule before looking at any image-feature result.")

    df = load_cohort("image")
    keys, feats = load_features(args.pooling)
    keys["feat_idx"] = np.arange(len(keys))
    df = df.merge(keys, on=["fire_key", "segment_id"], how="left")
    missing = df["feat_idx"].isna()
    if missing.any():
        print(f"Warning: {int(missing.sum())} image-cohort rows have no Prithvi features and are dropped")
        df = df[~missing].reset_index(drop=True)
    if args.polygon_only:
        import geopandas as gpd

        src = gpd.read_file(C.OUT_DIR / "basins.gpkg", ignore_geometry=True)[["segment_id", "mask_source"]]
        df = df.merge(src, on="segment_id", how="left")
        df = df[df["mask_source"] == "polygon"].reset_index(drop=True)
    X = np.hstack([df[C.PREDICTOR_COLS].values, feats[df["feat_idx"].astype(int).values]])
    n_tab = len(C.PREDICTOR_COLS)

    folds, status = assign_folds(df, args.holdout)
    parts = []
    for k in sorted(f for f in folds.unique() if f >= 0):
        tr, te = ((folds != k) & (folds >= 0)).values, (folds == k).values
        search = make_search(args.head, df.loc[tr, "split_group"].values, n_tab=n_tab, n_pca=args.pca)
        search.fit(X[tr], df.loc[tr, "response"].values, groups=df.loc[tr, "split_group"].values)
        t = best_threshold(df.loc[tr, "response"].values, search.predict_proba(X[tr])[:, 1])
        parts.append(prediction_frame(df, pd.Series(te, index=df.index), search.predict_proba(X[te])[:, 1], t, k))
        print(f"fold {k}: {search.best_params_}")
    arm = f"A4_{args.head}_{args.pooling}" + ("_polygon" if args.polygon_only else "")
    summarize(pd.concat(parts, ignore_index=True), arm, status, run_label("image", args.holdout))


if __name__ == "__main__":
    main()
