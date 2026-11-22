import argparse

import numpy as np

import config as C
from evaluation import assign_folds, cross_validate, load_cohort, run_label, summarize
from models import searcher
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

    keys, feats = load_features(args.pooling)
    keys = keys.assign(feat_idx=np.arange(len(keys)))
    df = load_cohort("image").merge(keys, on=["fire_key", "segment_id"], how="left")
    if df["feat_idx"].isna().any():
        print(f"Warning: {int(df['feat_idx'].isna().sum())} image-cohort rows have no Prithvi features and are dropped")
        df = df.dropna(subset=["feat_idx"]).reset_index(drop=True)
    if args.polygon_only:
        import geopandas as gpd

        src = gpd.read_file(C.OUT_DIR / "basins.gpkg", ignore_geometry=True)[["segment_id", "mask_source"]]
        df = df.merge(src, on="segment_id", how="left").query("mask_source == 'polygon'").reset_index(drop=True)
    X = np.hstack([df[C.PREDICTOR_COLS].values, feats[df["feat_idx"].astype(int).values]])

    folds, status = assign_folds(df, args.holdout)
    search = searcher(args.head, n_tab=len(C.PREDICTOR_COLS), n_pca=args.pca)
    preds = cross_validate(df, X, folds, search, lambda k, s: print(f"fold {k}: {s.best_params_}"))
    arm = f"A4_{args.head}_{args.pooling}" + ("_polygon" if args.polygon_only else "")
    summarize(preds, arm, status, run_label("image", args.holdout))


if __name__ == "__main__":
    main()
