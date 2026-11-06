import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def norm_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def ensure_dirs():
    for d in (C.OUT_DIR, C.CHIP_DIR, C.HLS_DIR, C.ASSESSMENT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def selected(df, col="fire_key"):
    return df[df[col].isin(C.ONLY_FIRES)] if C.ONLY_FIRES else df


def load_inventory():
    df = pd.read_csv(C.INVENTORY_CSV, na_values=[-9999, "-9999", -9999.0, "-9999.0"], low_memory=False)
    needed = list(dict.fromkeys(list(C.COLS.values()) + C.PREDICTOR_COLS))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found in {C.INVENTORY_CSV.name}: {missing}\nAvailable: {list(df.columns)}")
    k = C.COLS
    df["fire_name"] = df[k["fire_name"]]
    df["fire_key"] = df[k["fire_name"]].map(slug)
    df["assessment_id"] = df[k["assessment_id"]].map(norm_id)
    df["segment_id"] = df[k["segment_id"]].map(norm_id)
    df["ignition"] = parse_date(df[k["ignition"]])
    df["storm_start"] = parse_date(df[k["storm_start"]])
    df["response"] = pd.to_numeric(df[k["response"]], errors="coerce")
    df["basin_area_km2"] = pd.to_numeric(df[k["basin_area"]], errors="coerce") * C.BASIN_AREA_UNITS_TO_KM2
    df["gage"] = df[k["gage"]]
    df = df[df["response"].isin([0, 1])].copy()
    df["response"] = df["response"].astype(int)
    df["days_since_fire"] = (df["storm_start"] - df["ignition"]).dt.days

    df["cohort_tabular"] = df[C.PREDICTOR_COLS].notna().all(axis=1)
    df["cohort_image_candidate"] = (
        df["cohort_tabular"]
        & (df["ignition"] >= pd.Timestamp(C.MIN_IGNITION))
        & (df["days_since_fire"] >= C.POST_FIRE_MIN_DAYS + C.POST_FIRE_STORM_BUFFER_DAYS)
        & df["segment_id"].notna()
        & df["assessment_id"].notna()
    )
    return df


def parse_date(s):
    as_text = pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    out = pd.to_datetime(as_text, format=C.DATE_FORMAT, errors="coerce")
    return out.fillna(pd.to_datetime(s, errors="coerce"))


def clear_mask(fmask):
    fm = fmask.astype(np.uint16)
    bad = fm == C.FMASK_FILL
    for b in C.FMASK_BAD_BITS:
        bad |= ((fm >> b) & 1).astype(bool)
    return ~bad


def parse_granule_ur(ur):
    parts = ur.split(".")
    return parts[1], parts[2], pd.to_datetime(parts[3][:7], format="%Y%j")


def basin_file_rank(name):
    n = Path(name).stem.lower()
    for i, key in enumerate(C.BASIN_FILE_PREFERENCE):
        if key in n:
            return i
    return len(C.BASIN_FILE_PREFERENCE)


def assessment_folders(assessment_id):
    folders = [C.ASSESSMENT_DIR / assessment_id]
    folders += [p for p in C.RAW_DIR.glob(f"*_{assessment_id}") if p.is_dir()]
    return [f for f in folders if f.exists()]


def find_basin_shapefiles(assessment_id):
    files = [p for f in assessment_folders(assessment_id) for p in f.rglob("*.shp")]
    files = [p for p in files if basin_file_rank(p.name) < len(C.BASIN_FILE_PREFERENCE)] or files
    return sorted(set(files), key=lambda p: (basin_file_rank(p.name), len(p.name)))
