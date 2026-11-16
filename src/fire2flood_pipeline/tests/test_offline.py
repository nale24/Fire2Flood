import importlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import config as C

WORK = Path(tempfile.mkdtemp())
C.ROOT = WORK
C.DATA_DIR = WORK / "data"
C.RAW_DIR = C.DATA_DIR / "raw"
C.INVENTORY_CSV = C.RAW_DIR / "USGS_RG_PFDF_Inventory_v1.csv"
C.ASSESSMENT_DIR = C.RAW_DIR / "assessments"
C.HLS_DIR = C.DATA_DIR / "interim" / "hls"
C.OUT_DIR = C.DATA_DIR / "processed"
C.CHIP_DIR = C.OUT_DIR / "chips"
C.N_OUTER_FOLDS = 3
UTM = "EPSG:32611"
rng = np.random.default_rng(0)


def step(name):
    return importlib.import_module(name)


def make_inventory():
    rows, polys = [], {}
    for f in range(8):
        aid = f"f{f:02d}{2015 + f % 5}"
        ignition = pd.Timestamp("2012-07-01") if f == 0 else pd.Timestamp(f"{2015 + f % 5}-08-15")
        x0, y0 = 300000 + f * 20000, 3800000
        for s in range(12):
            g = box(x0 + s * 900, y0, x0 + s * 900 + 700, y0 + 1200)
            polys.setdefault(aid, []).append(
                dict(Segment_ID=float(s + 1), Fire_ID=aid, Fire_Segme=f"{aid}_{s + 1}", geometry=g)
            )
            for storm in (ignition + pd.Timedelta(days=5 if f == 7 else 60), ignition + pd.Timedelta(days=200)):
                r = rng.random()
                rows.append(
                    {
                        "FireName": f"Fire {f}", "AssessmentFireID": aid, "AssessmentSegmentID": f"{aid}_{s + 1}",
                        "FireIgnitionDate": int(ignition.strftime("%Y%m%d")), "StormStart": int(storm.strftime("%Y%m%d")),
                        "Response": 1 if r < 0.25 else 0, "UpstreamArea": g.area / 1e6, "GageName": f"G{f}",
                        "PeakI15": rng.uniform(5, 60), "HM23": rng.random(), "dNBR/1000": rng.random(),
                        "KF": "-9999" if s == 11 else rng.random(),
                    }
                )
    C.RAW_DIR.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(C.INVENTORY_CSV, index=False)
    return polys


def make_assessment_zips(polys):
    zips = {}
    for aid, recs in polys.items():
        tmp = WORK / "zipsrc" / aid
        tmp.mkdir(parents=True)
        gdf = gpd.GeoDataFrame(recs, crs=UTM)
        gdf.to_file(tmp / f"{aid}_Basin_DFPredictions_15min_24mmh.shp")
        gdf.to_file(tmp / f"{aid}_Basin_DFPredictions_15min_40mmh.shp")
        gdf.to_file(tmp / f"{aid}_Segments.shp")
        zpath = WORK / f"Fire_2015-01-01_{aid}.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            for p in tmp.iterdir():
                z.write(p, f"{aid}/{p.name}")
        zips[aid] = zpath
    return zips


def test_download_extraction(zips):
    dl = step("00_download_assessments")
    for aid, zpath in zips.items():
        files = dl.extract_basins(zpath, C.ASSESSMENT_DIR / aid)
        shps = [p.name for p in files if p.suffix == ".shp"]
        assert shps == [f"{aid}_Basin_DFPredictions_15min_24mmh.shp"], shps
        assert {p.suffix for p in files} >= {".shp", ".shx", ".dbf", ".prj"}
    aid = next(iter(zips))
    nested = WORK / "Shapefiles_outer.zip"
    with zipfile.ZipFile(nested, "w") as z:
        z.write(zips[aid], "inner/Shapefiles.zip")
    files = dl.extract_basins(nested, WORK / "nested_out")
    assert any(p.name.endswith("24mmh.shp") for p in files), "nested zip not handled"


def make_hls(fb, aid):
    x0 = 300000 + int(aid[1:3]) * 20000 - 3000
    tr = from_origin(x0, 3805000, 30, 30)
    prof = dict(driver="GTiff", width=600, height=400, count=1, crs=UTM, transform=tr)
    folder = C.HLS_DIR / "tiles"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for b in C.BANDS["S30"]:
        p = folder / f"HLS.S30.T11SXX.2018001T000000.v2.0.{aid}.{b}.tif"
        with rasterio.open(p, "w", dtype="int16", nodata=-9999, **prof) as d:
            d.write(rng.integers(200, 4000, (400, 600)).astype("int16"), 1)
        paths[b] = str(p)
    fm = np.zeros((400, 600), np.uint8)
    seg1 = fb[fb.segment_id == f"{aid}_1"].to_crs(UTM).geometry.iloc[0]
    c0, r0 = int((seg1.bounds[0] - x0) / 30), int((3805000 - seg1.bounds[3]) / 30)
    fm[r0 : r0 + 40, c0 : c0 + 23] = 0b10
    fpath = folder / f"HLS.S30.T11SXX.2018001T000000.v2.0.{aid}.Fmask.tif"
    with rasterio.open(fpath, "w", dtype="uint8", **prof) as d:
        d.write(fm, 1)
    return paths, str(fpath)


def main():
    polys = make_inventory()
    test_download_extraction(make_assessment_zips(polys))

    step("01_clean_inventory").main()
    inv = pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet")
    assert inv.ignition.notna().all() and inv.storm_start.notna().all(), "yyyymmdd dates not parsed"
    assert not inv[inv.assessment_id == "f002015"].cohort_image_candidate.any(), "pre-HLS fire in image cohort"
    assert not inv[(inv.fire_key == "fire_7") & (inv.days_since_fire < 11)].cohort_image_candidate.any()
    assert not inv[inv.segment_id.str.endswith("_12")].cohort_tabular.any(), "text -9999 not treated as missing"

    step("02_join_basins").main()
    basins = gpd.read_file(C.OUT_DIR / "basins.gpkg")
    assert basins.area_ok.all(), "area check failed"
    assert basins.crs.to_epsg() == 4326

    s3, s4 = step("03_select_scenes"), step("04_cut_chips")
    cand = inv[inv.cohort_image_candidate]
    fb_all = s3.fire_basins(cand, basins)
    records = []
    for fire, fb in fb_all.groupby("fire_key"):
        aid = fb.assessment_id.iloc[0]
        paths, fpath = make_hls(fb, aid)
        bc = s3.basin_clear_fractions({"T11SXX": fpath}, fb).set_index("segment_id")
        assert bc.loc[f"{aid}_1", "clear_frac"] < 0.2, "cloud not detected"
        assert (bc.drop(f"{aid}_1").clear_frac > 0.99).all(), "clear basins scored as cloudy"
        ignition = cand[cand.fire_key == fire].ignition.iloc[0]
        for w in ("pre", "post"):
            for _, b in fb.iterrows():
                arrays, st = s4.cut_one([paths[x] for x in C.BANDS["S30"]], fpath, b.geometry, b.lon, b.lat)
                assert arrays["bands"].shape == (6, 224, 224) and arrays["basin"].sum() > 0
                out = C.CHIP_DIR / fire / f"{b.segment_id}_{w}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, **arrays)
                date = ignition - pd.Timedelta(days=30) if w == "pre" else ignition + pd.Timedelta(days=20)
                records.append(dict(fire_key=fire, segment_id=b.segment_id, window=w, date=date,
                                    chip_path=str(out.relative_to(C.ROOT)), **st))

    x, _ = s4.load_chip_for_prithvi(records[0]["chip_path"])
    assert x.dtype == np.float32 and x.shape == (6, 224, 224)
    s4.build_event_table(pd.DataFrame(records))
    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
    assert not ev[ev.segment_id.str.endswith("_1") & (ev.segment_id.str.count("_") == 1)].cohort_image.any(), \
        "cloudy basin in image cohort"

    s5 = step("05_freeze_splits")
    s5.main([])
    sp = json.loads((C.OUT_DIR / "splits.json").read_text())
    groups = sum(sp["outer_folds"].values(), [])
    assert len(groups) == len(set(groups)) and not set(groups) & set(sp["dev_groups"]), "group in two folds"
    try:
        s5.main([])
        raise AssertionError("splits were overwritten without --overwrite")
    except SystemExit:
        pass

    print("\nALL OFFLINE CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        os.chdir(HERE)
        shutil.rmtree(WORK, ignore_errors=True)
