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

    for ids in df.dropna(subset=["gage"]).groupby(["gage", "storm_start"])["assessment_id"].unique():
        for other in ids[1:]:
            parent[find(other)] = find(ids[0])
    return df["assessment_id"].map(find)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    out_path = C.OUT_DIR / "splits.json"
    if out_path.exists() and not ap.parse_args(argv).overwrite:
        raise SystemExit(f"{out_path} already exists. Splits are frozen; "
                         "pass --overwrite only before any results exist.")

    img = pd.read_parquet(C.OUT_DIR / "event_table.parquet").query("cohort_image").copy()
    img["split_group"] = split_groups(img)
    groups = img.groupby("split_group")["response"].agg(n="size", pos="sum")
    dominant = groups.pos.idxmax()
    pool = groups[(groups.pos > 0) & (groups.pos < groups.n) & (groups.index != dominant)].sort_values("n").head(8)
    rng = np.random.default_rng(C.RANDOM_SEED)
    dev = sorted(rng.choice(pool.index.tolist(), size=min(C.N_DEV_GROUPS, len(pool)), replace=False).tolist())

    test = img[~img.split_group.isin(dev)]
    sgkf = StratifiedGroupKFold(n_splits=min(C.N_OUTER_FOLDS, test.split_group.nunique()), shuffle=True,
                                random_state=C.RANDOM_SEED)
    folds = {f"fold_{k}": sorted(test.iloc[te].split_group.unique().tolist())
             for k, (_, te) in enumerate(sgkf.split(test, test.response, groups=test.split_group))}
    out_path.write_text(json.dumps(dict(
        seed=C.RANDOM_SEED, frozen_on=pd.Timestamp.now().isoformat(timespec="minutes"),
        group_by="assessment_id, merged when two assessments share a rain gage and storm date",
        dev_groups=dev, outer_folds=folds, dominant_group=dominant,
        group_members=img.groupby("split_group")["fire_key"].unique().map(sorted).to_dict()), indent=2))

    summary = pd.DataFrame([dict(fold=name, groups=len(gl), rows=int(m.sum()), debris_flows=int(test.response[m].sum()),
                                 has_dominant=dominant in gl)
                            for name, gl in folds.items() for m in [test.split_group.isin(gl)]])
    summary.to_csv(C.OUT_DIR / "split_summary.csv", index=False)
    print(f"Dev groups: {dev}\n{summary.to_string(index=False)}")
    if (summary.debris_flows == 0).any():
        print("Warning: a fold has no debris flows; lower N_OUTER_FOLDS")


if __name__ == "__main__":
    main()
