import geopandas as gpd
import pandas as pd

import config as C
from common import ensure_dirs, find_basin_shapefiles, norm_id, selected

EQUAL_AREA = "EPSG:5070"


def segment_keys(gdf, assessment_id):
    keys = {}
    for col in C.BASIN_KEY_COLUMNS:
        if col not in gdf.columns:
            continue
        values = gdf[col].map(norm_id)
        keys[col] = values
        keys[f"{col}+prefix"] = values.map(lambda v: None if v is None else f"{assessment_id}_{v}")
    return keys


def load_basins(assessment_id, wanted_segments):
    best = None
    for path in find_basin_shapefiles(assessment_id)[:3]:
        gdf = gpd.read_file(path)
        if gdf.crs is None or not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            continue
        for name, key in segment_keys(gdf, assessment_id).items():
            hits = key.isin(wanted_segments).sum()
            if best is None or hits > best[0]:
                best = (hits, path, gdf, key, name)
    if best is None or best[0] == 0:
        return None, "no basin file with matching segment IDs"
    hits, path, gdf, key, name = best
    gdf = gdf.assign(segment_id=key.values).dropna(subset=["segment_id"])
    gdf = gdf[gdf["segment_id"].isin(wanted_segments)].to_crs("EPSG:4326")
    gdf = gdf.dissolve(by="segment_id", as_index=False)[["segment_id", "geometry"]]
    gdf["assessment_id"] = assessment_id
    return gdf, f"{path.name} [{name}]"


def main():
    ensure_dirs()
    inv = pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet")
    inv = selected(inv[inv["cohort_image_candidate"]])

    parts, report = [], []
    for aid, grp in inv.groupby("assessment_id"):
        segs = grp.drop_duplicates("segment_id")[["assessment_id", "segment_id", "basin_area_km2"]]
        basins, note = load_basins(aid, set(segs["segment_id"]))
        if basins is None:
            report.append(dict(assessment_id=aid, basins=len(segs), matched=0, source=note))
            print(f"{aid}: skipped ({note})")
            continue
        m = segs.merge(basins, on=["assessment_id", "segment_id"], how="inner")
        parts.append(m)
        report.append(dict(assessment_id=aid, basins=len(segs), matched=len(m), source=note))
        print(f"{aid}: matched {len(m)}/{len(segs)} basins from {note}")

    if not parts:
        raise SystemExit("No basins joined. Run 00_download_assessments.py first.")
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    gdf["poly_area_km2"] = gdf.to_crs(EQUAL_AREA).area / 1e6
    gdf["area_ratio"] = gdf["poly_area_km2"] / gdf["basin_area_km2"]
    gdf["area_ok"] = gdf["area_ratio"].between(*C.AREA_RATIO_RANGE)
    pts = gdf.to_crs(EQUAL_AREA).geometry.representative_point().to_crs("EPSG:4326")
    gdf["lon"], gdf["lat"] = pts.x.values, pts.y.values
    gdf.to_file(C.OUT_DIR / "basins.gpkg", driver="GPKG")

    rep = pd.DataFrame(report).merge(
        gdf.groupby("assessment_id")["area_ok"].mean().rename("share_area_ok"), on="assessment_id", how="left"
    )
    rep.to_csv(C.OUT_DIR / "join_report.csv", index=False)
    print(
        f"\nJoined {len(gdf):,} basins across {gdf.assessment_id.nunique()} assessments; "
        f"{int((~gdf.area_ok).sum())} fail the area check"
    )


if __name__ == "__main__":
    main()
