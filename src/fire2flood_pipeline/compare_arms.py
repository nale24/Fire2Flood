import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

import config as C
from evaluation import DOMINANT_FIRE, results_dir


def paired_bootstrap(a, b, n_boot=2000, seed=C.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = a.groupby("split_group").indices
    groups = list(idx)
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
        y = a["y"].values[pick]
        if y.min() == y.max():
            continue
        diffs.append((average_precision_score(y, b["p"].values[pick]) - average_precision_score(y, a["p"].values[pick]),
                      brier_score_loss(y, b["p"].values[pick]) - brier_score_loss(y, a["p"].values[pick])))
    d = np.array(diffs)
    return np.percentile(d[:, 0], [2.5, 97.5]), np.percentile(d[:, 1], [2.5, 97.5])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="image")
    ap.add_argument("--reference", default="A1_logistic")
    args = ap.parse_args(argv)

    files = sorted(results_dir().glob(f"*_{args.cohort}_predictions.parquet"))
    preds = {f.name.replace(f"_{args.cohort}_predictions.parquet", ""): pd.read_parquet(f) for f in files}
    if args.reference not in preds:
        raise SystemExit(f"Reference arm {args.reference} not found in {results_dir()}")
    ref_rows = set(preds[args.reference]["row_id"])
    partial = [a for a, p in preds.items() if not ref_rows <= set(p["row_id"])]
    for a in partial:
        print(f"Skipping {a}: it covers only part of the reference rows (compare it with --reference on its own rows)")
        preds.pop(a)
    common = ref_rows
    rows = []
    base = preds[args.reference]
    base = base[base["row_id"].isin(common)].sort_values("row_id").reset_index(drop=True)
    if args.cohort.endswith("_region"):
        region = "_".join(C.REGION_A)
        subsets = {f"test_{region}": base["fold"] == 0,
                   f"test_{region}_without_thomas": (base["fold"] == 0) & (base["fire_key"] != DOMINANT_FIRE),
                   f"test_not_{region}": base["fold"] == 1}
    else:
        subsets = {"all": base["fold"] >= 0, "without_thomas": base["fire_key"] != DOMINANT_FIRE}
    for subset, mask in subsets.items():
        ref = base[mask].reset_index(drop=True)
        if ref.empty or ref["y"].nunique() < 2:
            continue
        for arm, p in preds.items():
            p = p.set_index("row_id").loc[ref["row_id"]].reset_index()
            r = dict(arm=arm, subset=subset, rows=len(p), ap=average_precision_score(ref["y"], p["p"]),
                     brier=brier_score_loss(ref["y"], p["p"]))
            r["ap_fold_mean"] = np.mean([average_precision_score(g["y"], g["p"]) for _, g in
                                         p.assign(y=ref["y"].values).groupby("fold") if g["y"].nunique() == 2])
            if arm != args.reference:
                (lo, hi), (blo, bhi) = paired_bootstrap(ref, p)
                r.update(ap_diff=r["ap"] - average_precision_score(ref["y"], ref["p"]), ap_diff_lo=lo, ap_diff_hi=hi,
                         brier_diff=r["brier"] - brier_score_loss(ref["y"], ref["p"]), brier_diff_lo=blo,
                         brier_diff_hi=bhi)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(results_dir() / f"comparison_{args.cohort}.csv", index=False)
    print(f"{len(common):,} rows scored by every arm; differences are arm minus {args.reference}, "
          "with fire-cluster bootstrap 95% intervals\n")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
