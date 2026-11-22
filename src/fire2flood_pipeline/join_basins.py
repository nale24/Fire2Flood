import numpy as np
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


def polygon_basins(segs):
    parts, report = [], []
    for aid, grp in segs.groupby("assessment_id"):
        basins, note = load_basins(aid, set(grp["segment_id"]))
        matched = 0
        if basins is not None:
            m = grp[["assessment_id", "segment_id", "basin_area_km2"]].merge(
                basins, on=["assessment_id", "segment_id"], how="inner"
            )
            parts.append(m)
            matched = len(m)
        report.append(dict(assessment_id=aid, basins=len(grp), polygon_matched=matched, source=note))
        print(f"{aid}: {matched}/{len(grp)} polygons from {note}")
    if not parts:
        return gpd.GeoDataFrame(columns=["assessment_id", "segment_id", "basin_area_km2", "geometry"],
                                geometry="geometry", crs="EPSG:4326"), pd.DataFrame(report)
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    gdf["poly_area_km2"] = gdf.to_crs(EQUAL_AREA).area / 1e6
    gdf["area_ratio"] = gdf["poly_area_km2"] / gdf["basin_area_km2"]
    gdf["area_ok"] = gdf["area_ratio"].between(*C.AREA_RATIO_RANGE)
    pts = gdf.to_crs(EQUAL_AREA).geometry.representative_point().to_crs("EPSG:4326")
    gdf["lon"], gdf["lat"] = pts.x.values, pts.y.values
    gdf["mask_source"] = "polygon"
    return gdf, pd.DataFrame(report)


def proxy_basins(segs):
    pts = gpd.GeoSeries(gpd.points_from_xy(segs["obs_lon"], segs["obs_lat"]), crs="EPSG:4326", index=segs.index)
    radius = np.maximum(C.PROXY_MIN_RADIUS_M, np.sqrt(segs["basin_area_km2"].fillna(0) * 1e6 / np.pi))
    circles = pts.to_crs(EQUAL_AREA).buffer(radius.values, resolution=16).to_crs("EPSG:4326")
    out = gpd.GeoDataFrame(segs[["assessment_id", "segment_id", "basin_area_km2"]].copy(), geometry=circles.values,
                           crs="EPSG:4326")
    out["lon"], out["lat"] = segs["obs_lon"].values, segs["obs_lat"].values
    out["mask_source"] = "proxy"
    return out


def main():
    ensure_dirs()
    inv = pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet")
    inv = selected(inv[inv["cohort_image_candidate"]])
    has_obs = C.OBS_LAT_COL in inv.columns and C.OBS_LON_COL in inv.columns
    cols = ["assessment_id", "segment_id", "basin_area_km2"] + ([C.OBS_LAT_COL, C.OBS_LON_COL] if has_obs else [])
    segs = inv[cols].drop_duplicates("segment_id").rename(columns={C.OBS_LAT_COL: "obs_lat", C.OBS_LON_COL: "obs_lon"})

    poly, report = polygon_basins(segs)
    good = poly[poly["area_ok"]] if len(poly) else poly
    rest = segs[~segs["segment_id"].isin(good["segment_id"])]
    if has_obs:
        rest = rest.dropna(subset=["obs_lat", "obs_lon"])
        proxy = proxy_basins(rest) if len(rest) else None
    else:
        proxy = None
    gdf = gpd.GeoDataFrame(pd.concat([good] + ([proxy] if proxy is not None else []), ignore_index=True),
                           geometry="geometry", crs="EPSG:4326")
    if gdf.empty:
        raise SystemExit("No basins built. Run download_assessments.py first, or check the observation coordinates.")
    gdf.to_file(C.OUT_DIR / "basins.gpkg", driver="GPKG")

    counts = gdf.groupby(["assessment_id", "mask_source"]).size().unstack(fill_value=0)
    report = report.merge(counts.reset_index(), on="assessment_id", how="left").fillna(0)
    report.to_csv(C.OUT_DIR / "join_report.csv", index=False)

    rows = inv.merge(gdf[["segment_id", "mask_source"]], on="segment_id", how="left")
    by = rows.groupby("mask_source", dropna=False)["response"].agg(rows="size", debris_flows="sum")
    print(f"\nBasins: {len(gdf):,} ({int((gdf.mask_source == 'polygon').sum()):,} polygon, "
          f"{int((gdf.mask_source == 'proxy').sum()):,} proxy); "
          f"{int((~poly['area_ok']).sum()) if len(poly) else 0} polygons failed the area check and use a proxy")
    print(by.to_string())


if __name__ == "__main__":
    main()
