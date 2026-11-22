import geopandas as gpd
import numpy as np
import pandas as pd

import config as C
from common import ensure_dirs, find_basin_shapefiles, norm_id, selected

EQUAL_AREA = "EPSG:5070"
WGS84 = "EPSG:4326"


def segment_keys(gdf, assessment_id):
    keys = {}
    for col in [c for c in C.BASIN_KEY_COLUMNS if c in gdf.columns]:
        keys[col] = gdf[col].map(norm_id)
        keys[f"{col}+prefix"] = keys[col].map(lambda v: None if v is None else f"{assessment_id}_{v}")
    return keys


def load_basins(assessment_id, wanted):
    best = None
    for path in find_basin_shapefiles(assessment_id)[:3]:
        gdf = gpd.read_file(path)
        if gdf.crs is None or not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            continue
        for name, key in segment_keys(gdf, assessment_id).items():
            if best is None or key.isin(wanted).sum() > best[0]:
                best = (key.isin(wanted).sum(), path, gdf, key, name)
    if best is None or best[0] == 0:
        return None, "no basin file with matching segment IDs"
    _, path, gdf, key, name = best
    gdf = gdf.assign(segment_id=key.values).dropna(subset=["segment_id"])
    gdf = gdf[gdf["segment_id"].isin(wanted)].to_crs(WGS84).dissolve(by="segment_id", as_index=False)
    return gdf[["segment_id", "geometry"]].assign(assessment_id=assessment_id), f"{path.name} [{name}]"


def polygon_basins(segs):
    parts, report = [], []
    for aid, grp in segs.groupby("assessment_id"):
        basins, note = load_basins(aid, set(grp["segment_id"]))
        m = grp[["assessment_id", "segment_id", "basin_area_km2"]].merge(basins, on=["assessment_id", "segment_id"]) \
            if basins is not None else grp.iloc[:0]
        parts += [m] if basins is not None else []
        report.append(dict(assessment_id=aid, basins=len(grp), polygon_matched=len(m), source=note))
        print(f"{aid}: {len(m)}/{len(grp)} polygons from {note}")
    if not parts:
        empty = gpd.GeoDataFrame(columns=["assessment_id", "segment_id", "basin_area_km2", "geometry"],
                                 geometry="geometry", crs=WGS84)
        return empty, pd.DataFrame(report)
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=WGS84)
    gdf["poly_area_km2"] = gdf.to_crs(EQUAL_AREA).area / 1e6
    gdf["area_ratio"] = gdf["poly_area_km2"] / gdf["basin_area_km2"]
    gdf["area_ok"] = gdf["area_ratio"].between(*C.AREA_RATIO_RANGE)
    pts = gdf.to_crs(EQUAL_AREA).geometry.representative_point().to_crs(WGS84)
    gdf["lon"], gdf["lat"], gdf["mask_source"] = pts.x.values, pts.y.values, "polygon"
    return gdf, pd.DataFrame(report)


def proxy_basins(segs):
    pts = gpd.GeoSeries(gpd.points_from_xy(segs["obs_lon"], segs["obs_lat"]), crs=WGS84, index=segs.index)
    radius = np.maximum(C.PROXY_MIN_RADIUS_M, np.sqrt(segs["basin_area_km2"].fillna(0) * 1e6 / np.pi))
    circles = pts.to_crs(EQUAL_AREA).buffer(radius.values, resolution=16).to_crs(WGS84)
    out = gpd.GeoDataFrame(segs[["assessment_id", "segment_id", "basin_area_km2"]].copy(), geometry=circles.values,
                           crs=WGS84)
    out["lon"], out["lat"], out["mask_source"] = segs["obs_lon"].values, segs["obs_lat"].values, "proxy"
    return out


def main():
    ensure_dirs()
    inv = selected(pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet").query("cohort_image_candidate"))
    has_obs = {C.OBS_LAT_COL, C.OBS_LON_COL} <= set(inv.columns)
    cols = ["assessment_id", "segment_id", "basin_area_km2"] + ([C.OBS_LAT_COL, C.OBS_LON_COL] if has_obs else [])
    segs = inv[cols].drop_duplicates("segment_id").rename(columns={C.OBS_LAT_COL: "obs_lat", C.OBS_LON_COL: "obs_lon"})

    poly, report = polygon_basins(segs)
    good = poly[poly["area_ok"]] if len(poly) else poly
    rest = segs[~segs["segment_id"].isin(good["segment_id"])]
    rest = rest.dropna(subset=["obs_lat", "obs_lon"]) if has_obs else rest.iloc[:0]
    gdf = gpd.GeoDataFrame(pd.concat([good] + ([proxy_basins(rest)] if len(rest) else []), ignore_index=True),
                           geometry="geometry", crs=WGS84)
    if gdf.empty:
        raise SystemExit("No basins built. Run download_assessments.py first, or check the observation coordinates.")
    gdf.to_file(C.OUT_DIR / "basins.gpkg", driver="GPKG")
    counts = gdf.groupby(["assessment_id", "mask_source"]).size().unstack(fill_value=0).reset_index()
    report.merge(counts, on="assessment_id", how="left").fillna(0).to_csv(C.OUT_DIR / "join_report.csv", index=False)

    by = inv.merge(gdf[["segment_id", "mask_source"]], on="segment_id", how="left") \
        .groupby("mask_source", dropna=False)["response"].agg(rows="size", debris_flows="sum")
    failed = int((~poly["area_ok"]).sum()) if len(poly) else 0
    print(f"\nBasins: {len(gdf):,} ({int((gdf.mask_source == 'polygon').sum()):,} polygon, "
          f"{int((gdf.mask_source == 'proxy').sum()):,} proxy); "
          f"{failed} polygons failed the area check and use a proxy")
    print(by.to_string())


if __name__ == "__main__":
    main()
