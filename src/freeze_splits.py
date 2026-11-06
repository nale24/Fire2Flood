import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C


def split_groups(df):
    parent = {a: a for a in df["assessment_id"].unique()}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    shared = df.dropna(subset=["gage"]).groupby(["gage", "storm_start"])["assessment_id"].unique()
    for ids in shared:
        for other in ids[1:]:
            parent[find(other)] = find(ids[0])
    return df["assessment_id"].map(find)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out_path = C.OUT_DIR / "splits.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists. Splits are frozen; pass --overwrite only before any results exist.")

    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
    img = ev[ev["cohort_image"]].copy()
    img["split_group"] = split_groups(img)

    groups = img.groupby("split_group")["response"].agg(n="size", pos="sum")
    groups["both"] = (groups.pos > 0) & (groups.pos < groups.n)
    dominant = groups.pos.idxmax()

    rng = np.random.default_rng(C.RANDOM_SEED)
    pool = groups[groups.both & (groups.index != dominant)].sort_values("n").head(8).index.tolist()
    dev = sorted(rng.choice(pool, size=min(C.N_DEV_GROUPS, len(pool)), replace=False).tolist())

    test = img[~img.split_group.isin(dev)]
    n_folds = min(C.N_OUTER_FOLDS, test.split_group.nunique())
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=C.RANDOM_SEED)
    folds = {
        f"fold_{k}": sorted(test.iloc[te].split_group.unique().tolist())
        for k, (_, te) in enumerate(sgkf.split(test, test.response, groups=test.split_group))
    }

    members = img.groupby("split_group")["fire_key"].unique().map(sorted).to_dict()
    out = dict(
        seed=C.RANDOM_SEED,
        frozen_on=pd.Timestamp.now().isoformat(timespec="minutes"),
        group_by="assessment_id, merged when two assessments share a rain gage and storm date",
        dev_groups=dev,
        outer_folds=folds,
        dominant_group=dominant,
        group_members=members,
    )
    out_path.write_text(json.dumps(out, indent=2))

    rows = []
    for name, gl in folds.items():
        t = test[test.split_group.isin(gl)]
        rows.append(dict(fold=name, groups=len(gl), rows=len(t), debris_flows=int(t.response.sum()), has_dominant=dominant in gl))
    summary = pd.DataFrame(rows)
    summary.to_csv(C.OUT_DIR / "split_summary.csv", index=False)
    print(f"Dev groups: {dev}\n{summary.to_string(index=False)}")
    if (summary.debris_flows == 0).any():
        print("Warning: a fold has no debris flows; lower N_OUTER_FOLDS")


if __name__ == "__main__":
    main()
