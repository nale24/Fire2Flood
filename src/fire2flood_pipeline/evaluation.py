import json
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

import config as C

DOMINANT_FIRE = "thomas"
COHORTS = ("tabular", "image_candidate", "image")
RESULTS_DIR_NAME = "results"


def results_dir():
    d = C.ROOT / "reports" / RESULTS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cohort(cohort):
    if cohort not in COHORTS:
        raise SystemExit(f"cohort must be one of {COHORTS}")
    if cohort == "image":
        df = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
        df = df[df["cohort_image"]]
    else:
        df = pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet")
        df = df[df[f"cohort_{cohort}"]]
    df = df.dropna(subset=C.PREDICTOR_COLS).copy()
    df["split_group"] = import_module("freeze_splits").split_groups(df)
    df["row_id"] = np.arange(len(df))
    return df.reset_index(drop=True)


def run_label(cohort, holdout):
    return cohort if holdout == "fire" else f"{cohort}_{holdout}"


def assign_folds(df, holdout="fire"):
    folds, status = fire_folds(df)
    if holdout == "fire":
        return folds, status
    if holdout != "region":
        raise SystemExit(f"unknown holdout {holdout}")
    in_a = df[C.REGION_COL].isin(C.REGION_A)
    region = pd.Series(np.where(in_a, 0, 1), index=df.index)
    region[folds < 0] = -1
    return region, "region"


def fire_folds(df):
    path = C.OUT_DIR / "splits.json"
    if path.exists():
        sp = json.loads(path.read_text())
        fold_of = {g: k for k, groups in enumerate(sp["outer_folds"].values()) for g in groups}
        folds = df["split_group"].map(fold_of)
        if folds.isna().all():
            raise SystemExit("splits.json groups do not match this cohort; freeze splits on the same cohort")
        return folds.fillna(-1).astype(int), "frozen"
    sgkf = StratifiedGroupKFold(n_splits=C.N_OUTER_FOLDS, shuffle=True, random_state=C.RANDOM_SEED)
    folds = np.full(len(df), -1)
    for k, (_, te) in enumerate(sgkf.split(df, df["response"], groups=df["split_group"])):
        folds[te] = k
    return pd.Series(folds, index=df.index), "provisional"


def best_threshold(y, p):
    best_t, best_ts = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.995, 100))):
        ts = threat_score(y, p >= t)
        if ts > best_ts:
            best_t, best_ts = t, ts
    return best_t


def threat_score(y, yhat):
    y, yhat = np.asarray(y).astype(bool), np.asarray(yhat).astype(bool)
    tp = (y & yhat).sum()
    denom = tp + (~y & yhat).sum() + (y & ~yhat).sum()
    return tp / denom if denom else 0.0


def point_metrics(d):
    if d["y"].nunique() < 2:
        return dict(ap=np.nan, brier=brier_score_loss(d["y"], d["p"]), threat=np.nan)
    return dict(
        ap=average_precision_score(d["y"], d["p"]),
        brier=brier_score_loss(d["y"], d["p"]),
        threat=threat_score(d["y"], d["p"] >= d["threshold"]),
    )


def bootstrap(d, n_boot=1000, seed=C.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    groups = d["split_group"].unique()
    by_group = {g: idx for g, idx in d.groupby("split_group").indices.items()}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        s = d.iloc[idx]
        if s["y"].nunique() == 2:
            stats.append(point_metrics(s))
    s = pd.DataFrame(stats)
    return {f"{k}_lo": s[k].quantile(0.025) for k in s} | {f"{k}_hi": s[k].quantile(0.975) for k in s}


def summarize(preds, arm, fold_status, cohort):
    rows = []
    if fold_status == "region":
        names = (f"test_{'_'.join(C.REGION_A)}", f"test_not_{'_'.join(C.REGION_A)}")
        subsets = ((names[0], preds[preds["fold"] == 0]),
                   (names[0] + "_without_thomas", preds[(preds["fold"] == 0) & (preds["fire_key"] != DOMINANT_FIRE)]),
                   (names[1], preds[preds["fold"] == 1]))
    else:
        subsets = (("all", preds), ("without_thomas", preds[preds["fire_key"] != DOMINANT_FIRE]))
    for subset, d in subsets:
        if d.empty:
            continue
        d = d.reset_index(drop=True)
        per_fold = [average_precision_score(f["y"], f["p"]) for _, f in d.groupby("fold") if f["y"].nunique() == 2]
        m = point_metrics(d) | bootstrap(d) | {"ap_fold_mean": float(np.mean(per_fold)) if per_fold else np.nan}
        rows.append(dict(arm=arm, cohort=cohort, folds=fold_status, subset=subset, rows=len(d),
                         debris_flows=int(d["y"].sum()), fires=d["fire_key"].nunique(),
                         base_rate=d["y"].mean(), **m))
    out = pd.DataFrame(rows)
    rd = results_dir()
    preds.to_parquet(rd / f"{arm}_{cohort}_predictions.parquet", index=False)
    out.to_csv(rd / f"{arm}_{cohort}_metrics.csv", index=False)
    cols = ["subset", "rows", "debris_flows", "base_rate", "ap", "ap_lo", "ap_hi", "ap_fold_mean", "brier", "threat"]
    print(f"\n{arm} on {cohort} cohort ({fold_status} folds)")
    print(out[cols].round(3).to_string(index=False))
    if fold_status == "region":
        print(f"Regional hold-out: fold 0 trains outside {C.REGION_A} and tests inside it; fold 1 is the reverse.")
    if fold_status == "provisional":
        print("Folds are provisional (no splits.json yet). Re-run after freeze_splits.py for final numbers.")
    return out


def prediction_frame(df, test_mask, p, threshold, fold):
    d = df.loc[test_mask, ["row_id", "fire_key", "split_group", "response"]].rename(columns={"response": "y"})
    d["p"], d["threshold"], d["fold"] = p, threshold, fold
    return d
