import json
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

import config as C

DOMINANT_FIRE = "thomas"
COHORTS = ("tabular", "image_candidate", "image")
METRIC_COLS = ["subset", "rows", "debris_flows", "base_rate", "ap", "ap_lo", "ap_hi", "ap_fold_mean", "brier", "threat"]


def results_dir():
    d = C.ROOT / "reports" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cohort(cohort):
    if cohort not in COHORTS:
        raise SystemExit(f"cohort must be one of {COHORTS}")
    df = pd.read_parquet(C.OUT_DIR / f"{'event_table' if cohort == 'image' else 'inventory_clean'}.parquet")
    df = df[df[f"cohort_{cohort}"]].dropna(subset=C.PREDICTOR_COLS).copy()
    df["split_group"] = import_module("freeze_splits").split_groups(df)
    df["row_id"] = np.arange(len(df))
    return df.reset_index(drop=True)


def run_label(cohort, holdout):
    return cohort if holdout == "fire" else f"{cohort}_{holdout}"


def fire_folds(df):
    path = C.OUT_DIR / "splits.json"
    if path.exists():
        groups = json.loads(path.read_text())["outer_folds"].values()
        folds = df["split_group"].map({g: k for k, gl in enumerate(groups) for g in gl})
        if folds.isna().all():
            raise SystemExit("splits.json groups do not match this cohort; freeze splits on the same cohort")
        return folds.fillna(-1).astype(int), "frozen"
    folds = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=C.N_OUTER_FOLDS, shuffle=True, random_state=C.RANDOM_SEED)
    for k, (_, te) in enumerate(sgkf.split(df, df["response"], groups=df["split_group"])):
        folds[te] = k
    return pd.Series(folds, index=df.index), "provisional"


def assign_folds(df, holdout="fire"):
    folds, status = fire_folds(df)
    if holdout == "fire":
        return folds, status
    if holdout != "region":
        raise SystemExit(f"unknown holdout {holdout}")
    region = pd.Series(np.where(df[C.REGION_COL].isin(C.REGION_A), 0, 1), index=df.index)
    region[folds < 0] = -1
    return region, "region"


def fold_splits(folds):
    for k in sorted(f for f in folds.unique() if f >= 0):
        yield k, ((folds != k) & (folds >= 0)).values, (folds == k).values


def threat_score(y, yhat):
    y, yhat = np.asarray(y).astype(bool), np.asarray(yhat).astype(bool)
    tp = (y & yhat).sum()
    denom = tp + (~y & yhat).sum() + (y & ~yhat).sum()
    return tp / denom if denom else 0.0


def best_threshold(y, p):
    best_t, best_ts = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.995, 100))):
        ts = threat_score(y, p >= t)
        if ts > best_ts:
            best_t, best_ts = t, ts
    return best_t


def prediction_frame(df, test_mask, p, threshold, fold):
    d = df.loc[test_mask, ["row_id", "fire_key", "split_group", "response"]].rename(columns={"response": "y"})
    d["p"], d["threshold"], d["fold"] = p, threshold, fold
    return d


def cross_validate(df, X, folds, make_search, on_fold=None):
    parts = []
    for k, tr, te in fold_splits(folds):
        y, g = df.loc[tr, "response"].values, df.loc[tr, "split_group"].values
        search = make_search(g)
        search.fit(X[tr], y, groups=g)
        t = best_threshold(y, search.predict_proba(X[tr])[:, 1])
        parts.append(prediction_frame(df, pd.Series(te, index=df.index), search.predict_proba(X[te])[:, 1], t, k))
        if on_fold:
            on_fold(k, search)
    return pd.concat(parts, ignore_index=True)


def point_metrics(d):
    brier = brier_score_loss(d["y"], d["p"])
    if d["y"].nunique() < 2:
        return dict(ap=np.nan, brier=brier, threat=np.nan)
    return dict(ap=average_precision_score(d["y"], d["p"]), brier=brier,
                threat=threat_score(d["y"], d["p"] >= d["threshold"]))


def bootstrap(d, n_boot=1000, seed=C.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    by_group, groups = d.groupby("split_group").indices, d["split_group"].unique()
    stats = []
    for _ in range(n_boot):
        s = d.iloc[np.concatenate([by_group[g] for g in rng.choice(groups, size=len(groups), replace=True)])]
        if s["y"].nunique() == 2:
            stats.append(point_metrics(s))
    s = pd.DataFrame(stats)
    return {f"{k}_lo": s[k].quantile(0.025) for k in s} | {f"{k}_hi": s[k].quantile(0.975) for k in s}


def subsets(preds, region):
    no_thomas = preds["fire_key"] != DOMINANT_FIRE
    if not region:
        return {"all": preds["fold"] >= 0, "without_thomas": no_thomas}
    a = "_".join(C.REGION_A)
    return {f"test_{a}": preds["fold"] == 0, f"test_{a}_without_thomas": (preds["fold"] == 0) & no_thomas,
            f"test_not_{a}": preds["fold"] == 1}


def summarize(preds, arm, fold_status, cohort):
    rows = []
    for subset, mask in subsets(preds, fold_status == "region").items():
        d = preds[mask].reset_index(drop=True)
        if d.empty:
            continue
        per_fold = [average_precision_score(f["y"], f["p"]) for _, f in d.groupby("fold") if f["y"].nunique() == 2]
        rows.append(dict(arm=arm, cohort=cohort, folds=fold_status, subset=subset, rows=len(d),
                         debris_flows=int(d["y"].sum()), fires=d["fire_key"].nunique(), base_rate=d["y"].mean(),
                         **point_metrics(d), **bootstrap(d),
                         ap_fold_mean=float(np.mean(per_fold)) if per_fold else np.nan))
    out = pd.DataFrame(rows)
    preds.to_parquet(results_dir() / f"{arm}_{cohort}_predictions.parquet", index=False)
    out.to_csv(results_dir() / f"{arm}_{cohort}_metrics.csv", index=False)
    print(f"\n{arm} on {cohort} cohort ({fold_status} folds)\n{out[METRIC_COLS].round(3).to_string(index=False)}")
    if fold_status == "region":
        print(f"Regional hold-out: fold 0 trains outside {C.REGION_A} and tests inside it; fold 1 is the reverse.")
    if fold_status == "provisional":
        print("Folds are provisional (no splits.json yet). Re-run after freeze_splits.py for final numbers.")
    return out
