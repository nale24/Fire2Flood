import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import config as C
from compare_arms import paired_bootstrap
from evaluation import results_dir, subsets

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POS, NEG = "#2a78d6", "#e34948"
NAMES = {"A1_logistic": "A1 logistic (4 predictors)", "A1b_gbm": "A1b gradient boosting",
         "A4_logistic_basin": "A4 + Prithvi, basin pooling", "A4_logistic_context": "A4 + Prithvi, context pooling",
         "A4_probe_attention": "A4 + Prithvi, attention probe", "FACEBOOK-OPT-1.3B_embed": "OPT-1.3B text embeddings",
         "GPT2_embed": "GPT-2 text embeddings", "GPT2_fewshot": "GPT-2 few-shot prompting"}
FOCUS = ["A1_logistic", "A4_logistic_basin", "FACEBOOK-OPT-1.3B_embed"]
FIG_DIR = C.ROOT / "reports" / "figures"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def finish(fig, name, note=None, note_y=-0.03):
    if note:
        fig.text(0.01, note_y, note, color=INK2, fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def ap(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan


def load_preds(cohort):
    files = {f.name.replace(f"_{cohort}_predictions.parquet", ""): f
             for f in sorted(results_dir().glob(f"*_{cohort}_predictions.parquet"))}
    out = {a: pd.read_parquet(f) for a, f in files.items() if a in NAMES}
    ref = set(out["A1_logistic"]["row_id"])
    return {a: p.set_index("row_id").loc[sorted(ref)].reset_index() for a, p in out.items() if ref <= set(p["row_id"])}


def fig_forest(preds, cohort):
    ref, rows = preds["A1_logistic"], []
    for (_, mask), label in zip(subsets(ref, False).items(), ("All fires", "Without Thomas")):
        r = ref[mask].reset_index(drop=True)
        for arm, p in [(a, p) for a, p in preds.items() if a != "A1_logistic"]:
            p = p[mask].reset_index(drop=True)
            (lo, hi), _ = paired_bootstrap(r, p)
            rows.append(dict(arm=arm, subset=label, diff=ap(r["y"], p["p"]) - ap(r["y"], r["p"]), lo=lo, hi=hi))
    d = pd.DataFrame(rows)
    order = d[d.subset == "All fires"].sort_values("diff")["arm"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))
    for ax, (subset, g) in zip(axes, d.groupby("subset", sort=False)):
        g = g.set_index("arm").loc[order]
        sig = (g["lo"] > 0) | (g["hi"] < 0)
        ax.axvline(0, color=INK2, lw=1)
        ax.hlines(y, g["lo"], g["hi"], color=SERIES[0], lw=2)
        ax.scatter(g["diff"], y, s=60, color=[SERIES[0] if s else SURFACE for s in sig], edgecolor=SERIES[0], lw=2,
                   zorder=3)
        ax.set_yticks(y, [NAMES[a] for a in order])
        ax.set_title(subset, loc="left")
    fig.supxlabel("Change in average precision vs A1 (95% fire-cluster interval)", color=INK2, fontsize=10)
    fig.suptitle("Across all fires, no foundation model clearly beats the 4-predictor logistic baseline",
                 x=0.01, ha="left", fontweight="bold")
    finish(fig, f"fig1_ap_change_vs_A1_{cohort}", "Filled dot = interval excludes 0. Same held-out rows and folds for "
           "every model.", -0.02)
    return d


def binned(preds, df, col, bins, labels, arms):
    cut = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for lab in labels:
        m = (cut == lab).values
        y = preds["A1_logistic"]["y"].values[m]
        rows += [dict(bin=lab, arm=arm, n=int(m.sum()), debris_flows=int(y.sum()),
                      base_rate=y.mean() if m.any() else np.nan, ap=ap(y, preds[arm]["p"].values[m])) for arm in arms]
    return pd.DataFrame(rows)


def fig_lines(table, title, xlabel, name, note):
    arms, labels = [a for a in FOCUS if a in table.arm.unique()], list(dict.fromkeys(table["bin"]))
    x, ends = np.arange(len(labels)), []
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, arm in enumerate(arms):
        g = table[table.arm == arm].set_index("bin").loc[labels]
        ax.plot(x, g["ap"], color=SERIES[i], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=NAMES[arm])
        ends.append([g["ap"].iloc[-1], NAMES[arm].split(" (")[0].split(",")[0]])
    top = max(table["ap"].max(), 0.01)
    ends.sort(key=lambda e: e[0])
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + top * 0.07)
    for yv, text in ends:
        ax.annotate(text, (x[-1], yv), xytext=(10, 0), textcoords="offset points", va="center", color=INK2, fontsize=9,
                    annotation_clip=False)
    base = table[table.arm == arms[0]].set_index("bin").loc[labels]
    ax.plot(x, base["base_rate"], color=INK2, lw=1.2, ls="--", label="Base rate (random model)")
    ax.set_xticks(x, [f"{l}\n(n={n}, {d} flows)" for l, n, d in zip(labels, base["n"], base["debris_flows"])])
    ax.set(xlim=(-0.3, len(labels) + 0.6), ylim=(0, max(top, ends[-1][0]) * 1.08), ylabel="Average precision",
           xlabel=xlabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.2), fontsize=9)
    finish(fig, name, note, -0.04)


def fig_per_fire(preds, cohort):
    a1, a4 = preds["A1_logistic"], preds["A4_logistic_basin"]
    d = pd.DataFrame([dict(fire=fire, n=len(y), debris_flows=int(y.sum()), A1=ap(y, a1.loc[idx, "p"]),
                           A4=ap(y, a4.loc[idx, "p"]))
                      for fire, idx in a1.groupby("fire_key").groups.items() for y in [a1.loc[idx, "y"].values]
                      if 5 <= y.sum() < len(y)])
    d = d.assign(diff=d["A4"] - d["A1"]).sort_values("diff")
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.barh(y, d["diff"], color=[POS if v > 0 else NEG for v in d["diff"]], height=0.7)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [f"{f.replace('_', ' ').title()} ({n} flows)" for f, n in zip(d["fire"], d["debris_flows"])])
    ax.set_xlabel("A4 minus A1 average precision within the fire")
    ax.set_title(f"Prithvi helps in {int((d['diff'] > 0).sum())} of {len(d)} fires and hurts in "
                 f"{int((d['diff'] < 0).sum())}", loc="left")
    finish(fig, f"fig4_per_fire_A4_vs_A1_{cohort}", "Fires with at least 5 debris flows and both outcomes; "
           "blue = A4 better, red = A1 better.")
    return d


def fig_funnel():
    raw = pd.read_csv(C.INVENTORY_CSV, low_memory=False)
    inv = pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet")
    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
    raw = raw.assign(response=pd.to_numeric(raw[C.COLS["response"]], errors="coerce"))
    steps = [("USGS inventory rows", raw), ("Known outcome (0/1)", inv), ("All 4 predictors", inv[inv.cohort_tabular]),
             ("Fire after mid-2013, storm >10 days later", inv[inv.cohort_image_candidate]),
             ("Clear pre- and post-fire HLS images", ev[ev.cohort_image])]
    t = pd.DataFrame([dict(step=name, rows=len(d), debris_flows=int((d["response"] == 1).sum()),
                           fires=d[C.COLS["fire_name"]].nunique()) for name, d in steps])
    if t.loc[0, "rows"] == t.loc[1, "rows"]:
        t = t.drop(index=1).reset_index(drop=True)
        t.loc[0, "step"] = "USGS inventory (all outcomes known)"
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(t) + 0.8))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["rows"], color=["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95"][-len(t):], height=0.7)
    for yi, r in zip(y, t.itertuples()):
        label = f"{r.rows:,} rows · {r.debris_flows} debris flows · {r.fires} fires"
        ax.text(r.rows + t["rows"].max() * 0.01, yi, label, va="center", fontsize=9, color=INK)
    ax.set_yticks(y, t["step"])
    ax.set(xlim=(0, t["rows"].max() * 1.6), xlabel="Basin-storm observations")
    ax.grid(axis="y", visible=False)
    ax.set_title("From inventory to the image cohort", loc="left")
    finish(fig, "fig0_cohort_funnel")
    return t


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="image")
    cohort = parser.parse_args(argv).cohort
    style()
    print(f"Writing figures to {FIG_DIR}")
    preds = load_preds(cohort)
    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet").query("cohort_image").dropna(subset=C.PREDICTOR_COLS)
    ev = ev.reset_index(drop=True).rename_axis("row_id")
    import geopandas as gpd

    mask = gpd.read_file(C.OUT_DIR / "basins.gpkg", ignore_geometry=True)[["segment_id", "mask_source"]]
    df = ev.loc[preds["A1_logistic"]["row_id"]].reset_index().merge(mask, on="segment_id", how="left")

    arms = [a for a in FOCUS if a in preds]
    tables = {"funnel": fig_funnel(), "forest": fig_forest(preds, cohort)}
    tables["size"] = binned(preds, df, "tokens_touched_post", [0, 4, 9, 24, 1e9],
                            ["up to 4", "5-9", "10-24", "25+"], arms)
    fig_lines(tables["size"], "Larger basins are easier for every model; Prithvi never pulls ahead",
              "Basin size in Prithvi tokens (1 token = 480 m square)", f"fig2_ap_by_basin_size_{cohort}",
              "Within each size group; base rate differs by group.")
    q = df["PeakI15"].quantile([1 / 3, 2 / 3]).values
    tables["rainfall"] = binned(preds, df, "PeakI15", [-np.inf, q[0], q[1], np.inf],
                                [f"< {q[0]:.0f} mm/h", f"{q[0]:.0f}-{q[1]:.0f} mm/h", f"> {q[1]:.0f} mm/h"], arms)
    fig_lines(tables["rainfall"], "Most skill comes from the heaviest storms",
              "Peak 15-minute rainfall intensity (terciles)", f"fig3_ap_by_rainfall_{cohort}",
              "Within each rainfall tercile; base rate rises with intensity.")
    tables["mask_source"] = binned(preds, df.assign(m=(df["mask_source"] == "polygon").astype(int)), "m",
                                   [-0.5, 0.5, 1.5], ["Proxy circle", "USGS polygon"], arms)
    tables["per_fire"] = fig_per_fire(preds, cohort)
    out = results_dir() / f"failure_analysis_{cohort}.csv"
    with open(out, "w", newline="") as fh:
        for name, t in tables.items():
            fh.write(f"# {name}\n")
            t.to_csv(fh, index=False)
            fh.write("\n")
    print(f"Tables written to {out}\n\nBy mask source:\n{tables['mask_source'].round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
