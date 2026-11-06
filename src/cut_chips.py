import json
import shutil
from importlib import import_module
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window

import config as C
from common import clear_mask, ensure_dirs, selected


def cut_one(band_paths, fmask_path, geom_4326, lon, lat):
    half = C.CHIP_SIZE // 2
    with rasterio.open(fmask_path) as fsrc:
        crs, transform = fsrc.crs, fsrc.transform
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
        row, col = fsrc.index(x, y)
        win = Window(col - half, row - half, C.CHIP_SIZE, C.CHIP_SIZE)
        fmask = fsrc.read(1, window=win, boundless=True, fill_value=C.FMASK_FILL).astype(np.uint8)
        wt = fsrc.window_transform(win)
    bands = np.full((6, C.CHIP_SIZE, C.CHIP_SIZE), C.HLS_NODATA, dtype=np.int16)
    for i, p in enumerate(band_paths):
        with rasterio.open(p) as src:
            bands[i] = src.read(1, window=win, boundless=True, fill_value=C.HLS_NODATA)

    geom = gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(crs).iloc[0]
    basin = (~geometry_mask([geom], out_shape=fmask.shape, transform=wt, all_touched=True)).astype(np.uint8)
    valid = clear_mask(fmask) & (bands != C.HLS_NODATA).all(axis=0)
    full_px = geom.area / abs(transform.a * transform.e)
    p = C.PATCH_SIZE
    tokens = basin.reshape(C.CHIP_SIZE // p, p, C.CHIP_SIZE // p, p).max(axis=(1, 3))
    stats = dict(
        chip_valid_frac=float(valid.mean()),
        basin_pixels=int(basin.sum()),
        basin_clear_frac=float(valid[basin == 1].mean()) if basin.any() else 0.0,
        basin_inside_chip=float(min(1.0, basin.sum() / max(full_px, 1))),
        tokens_touched=int(tokens.sum()),
    )
    return dict(bands=bands, fmask=fmask, basin=basin), stats


def load_chip_for_prithvi(path):
    path = Path(path)
    z = np.load(path if path.is_absolute() else C.ROOT / path)
    x = z["bands"].astype(np.float32)
    bad = (z["bands"] == C.HLS_NODATA) | ~clear_mask(z["fmask"])[None]
    mean = np.array(C.PRITHVI_MEAN, np.float32)[:, None, None]
    std = np.array(C.PRITHVI_STD, np.float32)[:, None, None]
    x = (x - mean) / std
    x[bad] = 0.0
    return x, z["basin"]


def band_urls(granule, sensor):
    return {b: next(u for u in granule["links"] if u.endswith(f".{b}.tif")) for b in C.BANDS[sensor] + ["Fmask"]}


def build_event_table(cm):
    inv = pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet")
    wide = cm.pivot_table(
        index=["fire_key", "segment_id"], columns="window",
        values=["chip_path", "basin_clear_frac", "date", "tokens_touched"], aggfunc="first",
    )
    wide.columns = [f"{v}_{w}" for v, w in wide.columns]
    ev = inv.merge(wide.reset_index(), on=["fire_key", "segment_id"], how="left")

    def clear(w):
        col = f"basin_clear_frac_{w}"
        return ev[col].fillna(0) >= C.MIN_CLEAR_FRACTION if col in ev else pd.Series(False, index=ev.index)

    ev["cohort_image"] = ev["cohort_image_candidate"] & clear("pre") & clear("post")
    if "date_post" in ev:
        leak = ev["cohort_image"] & (pd.to_datetime(ev["date_post"]) >= ev["storm_start"])
        if leak.any():
            raise RuntimeError(f"{int(leak.sum())} rows have a post-fire image on or after their storm")
    ev.to_parquet(C.OUT_DIR / "event_table.parquet", index=False)
    img = ev[ev.cohort_image]
    print(
        f"\nEvent table: {len(ev):,} rows; image cohort {len(img):,} rows, "
        f"{img.fire_key.nunique()} fires, {int(img.response.sum())} debris flows"
    )
    return ev


def main():
    import earthaccess

    fire_basins = import_module("03_select_scenes").fire_basins

    earthaccess.login(persist=True)
    ensure_dirs()
    inv = pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet")
    inv = selected(inv[inv["cohort_image_candidate"]])
    scenes = selected(pd.read_parquet(C.OUT_DIR / "scene_manifest.parquet"))
    clear = pd.read_parquet(C.OUT_DIR / "basin_clear.parquet")
    basins = fire_basins(inv, gpd.read_file(C.OUT_DIR / "basins.gpkg"))

    records = []
    for _, sc in scenes.iterrows():
        fire, wname, sensor = sc.fire_key, sc.window, sc.sensor
        local = {}
        for tile, g in json.loads(sc.granules).items():
            urls = band_urls(g, sensor)
            paths = earthaccess.download(list(urls.values()), local_path=str(C.HLS_DIR / fire / wname / tile), show_progress=False)
            local[tile] = {b: next(str(p) for p in paths if str(p).endswith(f".{b}.tif")) for b in urls}

        cw = clear[(clear.fire_key == fire) & (clear.window == wname)][["segment_id", "tile"]]
        fb = basins[basins.fire_key == fire].merge(cw, on="segment_id", how="left")
        out_dir = C.CHIP_DIR / fire
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, b in fb.iterrows():
            tile = b.tile if isinstance(b.tile, str) and b.tile in local else next(iter(local))
            paths = local[tile]
            arrays, stats = cut_one([paths[x] for x in C.BANDS[sensor]], paths["Fmask"], b.geometry, b.lon, b.lat)
            out = out_dir / f"{b.segment_id}_{wname}.npz"
            np.savez_compressed(out, **arrays)
            records.append(
                dict(fire_key=fire, segment_id=b.segment_id, window=wname, sensor=sensor, date=sc.date,
                     tile=tile, chip_path=str(out.relative_to(C.ROOT)), **stats)
            )
        print(f"{fire} {wname}: {len(fb)} chips")
        if not C.KEEP_HLS_DOWNLOADS:
            shutil.rmtree(C.HLS_DIR / fire / wname, ignore_errors=True)

    cm = pd.DataFrame(records)
    cm.to_parquet(C.OUT_DIR / "chip_manifest.parquet", index=False)
    build_event_table(cm)


if __name__ == "__main__":
    main()
