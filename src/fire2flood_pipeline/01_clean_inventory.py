import config as C
from common import ensure_dirs, load_inventory


def main():
    ensure_dirs()
    df = load_inventory()
    out = C.OUT_DIR / "inventory_clean.parquet"
    df.to_parquet(out, index=False)

    fires = df.groupby("fire_key").agg(
        fire_name=("fire_name", "first"),
        assessment_id=("assessment_id", "first"),
        ignition=("ignition", "first"),
        rows=("response", "size"),
        debris_flows=("response", "sum"),
        image_rows=("cohort_image_candidate", "sum"),
    )
    fires.to_csv(C.OUT_DIR / "fire_summary.csv")

    img = df[df["cohort_image_candidate"]]
    per_fire = img.groupby("fire_key")["response"].agg(["min", "max"])
    print(f"Labelled rows: {len(df):,}")
    print(f"Tabular cohort: {int(df.cohort_tabular.sum()):,} rows, {df.loc[df.cohort_tabular, 'fire_key'].nunique()} fires")
    print(
        f"Image candidates: {len(img):,} rows, {img.fire_key.nunique()} fires, "
        f"{img.assessment_id.nunique()} assessments, {int(img.response.sum())} debris flows, "
        f"{int(((per_fire['min'] == 0) & (per_fire['max'] == 1)).sum())} fires with both outcomes"
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
