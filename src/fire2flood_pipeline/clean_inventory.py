import config as C
from common import ensure_dirs, load_inventory


def main():
    ensure_dirs()
    df = load_inventory()
    out = C.OUT_DIR / "inventory_clean.parquet"
    df.to_parquet(out, index=False)
    df.groupby("fire_key").agg(
        fire_name=("fire_name", "first"), assessment_id=("assessment_id", "first"), ignition=("ignition", "first"),
        rows=("response", "size"), debris_flows=("response", "sum"), image_rows=("cohort_image_candidate", "sum"),
    ).to_csv(C.OUT_DIR / "fire_summary.csv")

    img, tab = df[df["cohort_image_candidate"]], df[df["cohort_tabular"]]
    per_fire = img.groupby("fire_key")["response"].agg(["min", "max"])
    both = ((per_fire["min"] == 0) & (per_fire["max"] == 1)).sum()
    print(f"Labelled rows: {len(df):,}\nTabular cohort: {len(tab):,} rows, {tab.fire_key.nunique()} fires")
    print(f"Image candidates: {len(img):,} rows, {img.fire_key.nunique()} fires, {img.assessment_id.nunique()} "
          f"assessments, {int(img.response.sum())} debris flows, {int(both)} fires with both outcomes\nSaved {out}")


if __name__ == "__main__":
    main()
