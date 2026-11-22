import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

import config as C
from common import clear_mask, ensure_dirs, parse_granule_ur, selected


def basin_clear_fractions(fmask_paths, basins):
    best = {s: (None, 0.0) for s in basins["segment_id"]}
    for tile, path in fmask_paths.items():
        with rasterio.open(path) as src:
            tb, pad = src.bounds, 2 * abs(src.res[0])
            for seg, geom in zip(basins["segment_id"], basins.to_crs(src.crs).geometry):
                minx, miny, maxx, maxy = geom.bounds
                if maxx < tb.left or minx > tb.right or maxy < tb.bottom or miny > tb.top:
                    continue
                win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, src.transform)
                win = win.round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
                if win.width < 1 or win.height < 1:
                    continue
                fm = src.read(1, window=win)
                inside = ~geometry_mask([geom], out_shape=fm.shape, transform=src.window_transform(win),
                                        all_touched=True)
                if inside.any():
                    frac = float(clear_mask(fm)[inside].mean()) * geom.intersection(box(*tb)).area / geom.area
                    if frac > best[seg][1]:
                        best[seg] = (tile, frac)
    return pd.DataFrame([dict(segment_id=s, tile=t, clear_frac=f) for s, (t, f) in best.items()])


def windows_for_fire(ignition, first_storm):
    days = pd.Timedelta(days=1)
    out = {"pre": (ignition - C.PRE_FIRE_LOOKBACK_DAYS * days, ignition - C.PRE_FIRE_GAP_DAYS * days)}
    post = (ignition + C.POST_FIRE_MIN_DAYS * days, first_storm - C.POST_FIRE_STORM_BUFFER_DAYS * days)
    if post[1] >= post[0]:
        out["post"] = post
    return out


def search_passes(bbox, start, end):
    import earthaccess

    passes = {}
    for short in ("HLSL30", "HLSS30"):
        for g in earthaccess.search_data(short_name=short, version="2.0", bounding_box=bbox, count=500,
                                         temporal=(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"),
                                         cloud_cover=(0, C.MAX_SCENE_CLOUD)):
            ur = g["umm"]["GranuleUR"]
            sensor, tile, date = parse_granule_ur(ur)
            p = passes.setdefault((sensor, date), {"sensor": sensor, "date": date, "granules": {}})
            p["granules"][tile] = {"ur": ur, "links": g.data_links(access="external")}
    return sorted(passes.values(), key=lambda p: p["date"], reverse=True)


def download(urls, folder):
    import earthaccess

    folder.mkdir(parents=True, exist_ok=True)
    return [str(p) for p in earthaccess.download(urls, local_path=str(folder), show_progress=False)]


def fire_basins(inv, basins):
    segs = inv[["fire_key", "assessment_id", "segment_id"]].drop_duplicates()
    return basins.merge(segs, on=["assessment_id", "segment_id"], how="inner")


def best_pass(fire, wname, fb, bbox, start, end):
    best = None
    for p in search_passes(bbox, start, end)[: C.CANDIDATES_PER_WINDOW]:
        urls = [next(u for u in g["links"] if u.endswith(".Fmask.tif")) for g in p["granules"].values()]
        fmasks = download(urls, C.HLS_DIR / fire / "fmask")
        bc = basin_clear_fractions({Path(f).name.split(".")[2]: f for f in fmasks}, fb)
        score = float((bc.clear_frac >= C.MIN_CLEAR_FRACTION).mean())
        print(f"  {fire} {wname} {p['sensor']} {p['date'].date()}: {score:.0%} of basins clear")
        if best is None or score > best[0]:
            best = (score, p, bc)
        if score == 1.0:
            break
    return best


def main():
    import earthaccess

    earthaccess.login(persist=True)
    ensure_dirs()
    inv = selected(pd.read_parquet(C.OUT_DIR / "inventory_clean.parquet").query("cohort_image_candidate"))
    scenes, clear_rows = [], []
    for fire, fb in fire_basins(inv, gpd.read_file(C.OUT_DIR / "basins.gpkg")).groupby("fire_key"):
        rows = inv[inv.fire_key == fire]
        first_storm = rows.storm_start.min()
        minx, miny, maxx, maxy = fb.total_bounds
        wins = windows_for_fire(rows.ignition.iloc[0], first_storm)
        if "post" not in wins:
            print(f"{fire}: first storm too soon after ignition, no post-fire window")
        for wname, (start, end) in wins.items():
            best = best_pass(fire, wname, fb, (minx - 0.01, miny - 0.01, maxx + 0.01, maxy + 0.01), start, end)
            if best is None:
                print(f"{fire} {wname}: no HLS granules found")
                continue
            score, p, bc = best
            scenes.append(dict(fire_key=fire, window=wname, sensor=p["sensor"], date=p["date"],
                               share_basins_clear=score, first_storm=first_storm, granules=json.dumps(p["granules"])))
            clear_rows.append(bc.assign(fire_key=fire, window=wname))
    sm = pd.DataFrame(scenes)
    sm.to_parquet(C.OUT_DIR / "scene_manifest.parquet", index=False)
    pd.concat(clear_rows).to_parquet(C.OUT_DIR / "basin_clear.parquet", index=False)
    print(f"\nShare of basins passing the clear-pixel rule:\n"
          f"{sm.pivot(index='fire_key', columns='window', values='share_basins_clear').round(2)}")


if __name__ == "__main__":
    main()
