import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps, brier_score_loss as brier

import config as C
from evaluation import results_dir, subsets


def paired_bootstrap(a, b, n_boot=2000, seed=C.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = a.groupby("split_group").indices
    groups, diffs = list(idx), []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
        y, pa, pb = a["y"].values[pick], a["p"].values[pick], b["p"].values[pick]
        if y.min() != y.max():
            diffs.append((aps(y, pb) - aps(y, pa), brier(y, pb) - brier(y, pa)))
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
    for a in [a for a, p in preds.items() if not ref_rows <= set(p["row_id"])]:
        print(f"Skipping {a}: it covers only part of the reference rows (compare it with --reference on its own rows)")
        preds.pop(a)
    base = preds[args.reference].sort_values("row_id").reset_index(drop=True)
    rows = []
    for subset, mask in subsets(base, args.cohort.endswith("_region")).items():
        ref = base[mask].reset_index(drop=True)
        if ref.empty or ref["y"].nunique() < 2:
            continue
        for arm, p in preds.items():
            p = p.set_index("row_id").loc[ref["row_id"]].reset_index()
            r = dict(arm=arm, subset=subset, rows=len(p), ap=aps(ref["y"], p["p"]), brier=brier(ref["y"], p["p"]))
            r["ap_fold_mean"] = np.mean([aps(g["y"], g["p"]) for _, g in p.assign(y=ref["y"].values).groupby("fold")
                                         if g["y"].nunique() == 2])
            if arm != args.reference:
                (lo, hi), (blo, bhi) = paired_bootstrap(ref, p)
                r.update(ap_diff=r["ap"] - aps(ref["y"], ref["p"]), ap_diff_lo=lo, ap_diff_hi=hi,
                         brier_diff=r["brier"] - brier(ref["y"], ref["p"]), brier_diff_lo=blo, brier_diff_hi=bhi)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(results_dir() / f"comparison_{args.cohort}.csv", index=False)
    print(f"{len(ref_rows):,} rows scored by every arm; differences are arm minus {args.reference}, "
          f"with fire-cluster bootstrap 95% intervals\n\n{out.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
