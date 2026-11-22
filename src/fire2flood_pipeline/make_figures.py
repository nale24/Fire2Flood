import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import config as C
from compare_arms import paired_bootstrap
from evaluation import DOMINANT_FIRE, results_dir

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POS, NEG = "#2a78d6", "#e34948"

NAMES = {
    "A1_logistic": "A1 logistic (4 predictors)",
    "A1b_gbm": "A1b gradient boosting",
    "A4_logistic_basin": "A4 + Prithvi, basin pooling",
    "A4_logistic_context": "A4 + Prithvi, context pooling",
    "A4_probe_attention": "A4 + Prithvi, attention probe",
    "FACEBOOK-OPT-1.3B_embed": "OPT-1.3B text embeddings",
    "GPT2_embed": "GPT-2 text embeddings",
    "GPT2_fewshot": "GPT-2 few-shot prompting",
}
FOCUS = ["A1_logistic", "A4_logistic_basin", "FACEBOOK-OPT-1.3B_embed"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    })


def fig_dir():
    d = C.ROOT / "reports" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name):
    fig.savefig(fig_dir() / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def load_preds(cohort):
    preds = {}
    for f in sorted(results_dir().glob(f"*_{cohort}_predictions.parquet")):
        arm = f.name.replace(f"_{cohort}_predictions.parquet", "")
        if arm in NAMES:
            preds[arm] = f
    out = {a: pd.read_parquet(f) for a, f in preds.items()}
    ref = set(out["A1_logistic"]["row_id"])
    return {a: p.set_index("row_id").loc[sorted(ref)].reset_index() for a, p in out.items() if ref <= set(p["row_id"])}


def ap(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan


def fig_forest(preds, cohort):
    ref = preds["A1_logistic"]
    rows = []
    for subset, mask in (("All fires", ref["fold"] >= 0), ("Without Thomas", ref["fire_key"] != DOMINANT_FIRE)):
        r = ref[mask].reset_index(drop=True)
        for arm, p in preds.items():
            if arm == "A1_logistic":
                continue
            p = p[mask].reset_index(drop=True)
            (lo, hi), _ = paired_bootstrap(r, p)
            rows.append(dict(arm=arm, subset=subset, diff=ap(r["y"], p["p"]) - ap(r["y"], r["p"]), lo=lo, hi=hi))
    d = pd.DataFrame(rows)
    order = d[d.subset == "All fires"].sort_values("diff")["arm"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(order) + 1.6), sharey=True)
    for ax, (subset, g) in zip(axes, d.groupby("subset", sort=False)):
        g = g.set_index("arm").loc[order]
        y = np.arange(len(order))
        sig = (g["lo"] > 0) | (g["hi"] < 0)
        ax.axvline(0, color=INK2, lw=1)
        ax.hlines(y, g["lo"], g["hi"], color=SERIES[0], lw=2)
        ax.scatter(g["diff"], y, s=60, color=[SERIES[0] if s else SURFACE for s in sig], edgecolor=SERIES[0], lw=2, zorder=3)
        ax.set_yticks(y, [NAMES[a] for a in order])
        ax.set_title(subset, loc="left")
    fig.supxlabel("Change in average precision vs A1 (95% fire-cluster interval)", color=INK2, fontsize=10)
    fig.suptitle("Across all fires, no foundation model clearly beats the 4-predictor logistic baseline",
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, -0.02, "Filled dot = interval excludes 0. Same held-out rows and folds for every model.", color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, f"fig1_ap_change_vs_A1_{cohort}")
    return d


def binned(preds, df, col, bins, labels, arms):
    rows = []
    cut = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    for lab in labels:
        m = (cut == lab).values
        y = preds["A1_logistic"]["y"].values[m]
        for arm in arms:
            rows.append(dict(bin=lab, arm=arm, n=int(m.sum()), debris_flows=int(y.sum()),
                             base_rate=y.mean() if m.any() else np.nan, ap=ap(y, preds[arm]["p"].values[m])))
    return pd.DataFrame(rows)


def fig_lines(table, title, xlabel, name, note):
    arms = [a for a in FOCUS if a in table.arm.unique()]
    labels = list(dict.fromkeys(table["bin"]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ends = []
    for i, arm in enumerate(arms):
        g = table[table.arm == arm].set_index("bin").loc[labels]
        ax.plot(x, g["ap"], color=SERIES[i], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=NAMES[arm])
        ends.append([g["ap"].iloc[-1], NAMES[arm].split(" (")[0].split(",")[0]])
    top = max(table["ap"].max(), 0.01)
    gap = top * 0.07
    ends.sort(key=lambda e: e[0])
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + gap)
    for yv, text in ends:
        ax.annotate(text, (x[-1], yv), xytext=(10, 0), textcoords="offset points", va="center", color=INK2, fontsize=9,
                    annotation_clip=False)
    base = table[table.arm == arms[0]].set_index("bin").loc[labels]
    ax.plot(x, base["base_rate"], color=INK2, lw=1.2, ls="--", label="Base rate (random model)")
    ax.set_xticks(x, [f"{l}\n(n={n}, {d} flows)" for l, n, d in zip(labels, base["n"], base["debris_flows"])])
    ax.set_xlim(-0.3, len(labels) - 0.3 + 0.9)
    ax.set_ylim(0, max(top, ends[-1][0]) * 1.08)
    ax.set_ylabel("Average precision")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.2), fontsize=9)
    fig.text(0.01, -0.04, note, color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, name)


def fig_per_fire(preds, cohort):
    a1, a4 = preds["A1_logistic"], preds["A4_logistic_basin"]
    rows = []
    for fire, idx in a1.groupby("fire_key").groups.items():
        y = a1.loc[idx, "y"].values
        if y.sum() >= 5 and y.sum() < len(y):
            rows.append(dict(fire=fire, n=len(y), debris_flows=int(y.sum()),
                             A1=ap(y, a1.loc[idx, "p"]), A4=ap(y, a4.loc[idx, "p"])))
    d = pd.DataFrame(rows)
    d["diff"] = d["A4"] - d["A1"]
    d = d.sort_values("diff")
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.barh(y, d["diff"], color=[POS if v > 0 else NEG for v in d["diff"]], height=0.7)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [f"{f.replace('_', ' ').title()} ({n} flows)" for f, n in zip(d["fire"], d["debris_flows"])])
    ax.set_xlabel("A4 minus A1 average precision within the fire")
    ax.set_title(f"Prithvi helps in {int((d['diff'] > 0).sum())} of {len(d)} fires and hurts in {int((d['diff'] < 0).sum())}",
                 loc="left")
    fig.text(0.01, -0.03, "Fires with at least 5 debris flows and both outcomes; blue = A4 better, red = A1 better.",
             color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, f"fig4_per_fire_A4_vs_A1_{cohort}")
    return d


def fig_funnel():
    raw = pd.read_csv(C.INVENTORY_CSV, low_memory=False)
    inv = pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet")
    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
    steps = [("USGS inventory rows", raw), ("Known outcome (0/1)", inv), ("All 4 predictors", inv[inv.cohort_tabular]),
             ("Fire after mid-2013, storm >10 days later", inv[inv.cohort_image_candidate]),
             ("Clear pre- and post-fire HLS images", ev[ev.cohort_image])]
    rows = []
    for name, d in steps:
        resp = pd.to_numeric(d[C.COLS["response"]], errors="coerce") if name == steps[0][0] else d["response"]
        fires = d[C.COLS["fire_name"]].nunique()
        rows.append(dict(step=name, rows=len(d), debris_flows=int((resp == 1).sum()), fires=fires))
    t = pd.DataFrame(rows)
    if t.loc[0, "rows"] == t.loc[1, "rows"]:
        t = t.drop(index=1).reset_index(drop=True)
        t.loc[0, "step"] = "USGS inventory (all outcomes known)"
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(t) + 0.8))
    y = np.arange(len(t))[::-1]
    shades = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95"][-len(t):]
    ax.barh(y, t["rows"], color=shades, height=0.7)
    for yi, r in zip(y, t.itertuples()):
        ax.text(r.rows + t["rows"].max() * 0.01, yi, f"{r.rows:,} rows · {r.debris_flows} debris flows · {r.fires} fires",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, t["step"])
    ax.set_xlim(0, t["rows"].max() * 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Basin-storm observations")
    ax.set_title("From inventory to the image cohort", loc="left")
    fig.tight_layout()
    save(fig, "fig0_cohort_funnel")
    return t


def main(argv=None):
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--cohort", default="image")
    args = ap_.parse_args(argv)
    style()
    print(f"Writing figures to {fig_dir()}")
    preds = load_preds(args.cohort)
    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
    ev = ev[ev["cohort_image"]].dropna(subset=C.PREDICTOR_COLS).reset_index(drop=True)
    ev["row_id"] = np.arange(len(ev))
    df = ev.set_index("row_id").loc[preds["A1_logistic"]["row_id"]].reset_index()
    import geopandas as gpd

    mask = gpd.read_file(C.OUT_DIR / "basins.gpkg", ignore_geometry=True)[["segment_id", "mask_source"]]
    df = df.merge(mask, on="segment_id", how="left")

    tables = {"funnel": fig_funnel(), "forest": fig_forest(preds, args.cohort)}
    arms = [a for a in FOCUS if a in preds]
    size = binned(preds, df, "tokens_touched_post", [0, 4, 9, 24, 1e9], ["up to 4", "5-9", "10-24", "25+"], arms)
    fig_lines(size, "Larger basins are easier for every model; Prithvi never pulls ahead", "Basin size in Prithvi tokens (1 token = 480 m square)",
              f"fig2_ap_by_basin_size_{args.cohort}", "Within each size group; base rate differs by group.")
    q = df["PeakI15"].quantile([1 / 3, 2 / 3]).values
    rain = binned(preds, df, "PeakI15", [-np.inf, q[0], q[1], np.inf],
                  [f"< {q[0]:.0f} mm/h", f"{q[0]:.0f}-{q[1]:.0f} mm/h", f"> {q[1]:.0f} mm/h"], arms)
    fig_lines(rain, "Most skill comes from the heaviest storms", "Peak 15-minute rainfall intensity (terciles)",
              f"fig3_ap_by_rainfall_{args.cohort}", "Within each rainfall tercile; base rate rises with intensity.")
    src = binned(preds, df.assign(m=(df["mask_source"] == "polygon").astype(int)), "m", [-0.5, 0.5, 1.5],
                 ["Proxy circle", "USGS polygon"], arms)
    tables.update(size=size, rainfall=rain, mask_source=src, per_fire=fig_per_fire(preds, args.cohort))
    with open(results_dir() / f"failure_analysis_{args.cohort}.csv", "w", newline="") as fh:
        for name, t in tables.items():
            fh.write(f"# {name}\n")
            t.to_csv(fh, index=False)
            fh.write("\n")
    print(f"Tables written to {results_dir() / f'failure_analysis_{args.cohort}.csv'}")
    print("\nBy mask source:\n", src.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
