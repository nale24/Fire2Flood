from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INVENTORY_CSV = RAW_DIR / "USGS_RG_PFDF_Inventory_v1.csv"
ASSESSMENT_DIR = RAW_DIR / "assessments"
HLS_DIR = DATA_DIR / "interim" / "hls"
OUT_DIR = DATA_DIR / "processed"
CHIP_DIR = OUT_DIR / "chips"

COLS = {
    "fire_name": "FireName",
    "assessment_id": "AssessmentFireID",
    "segment_id": "AssessmentSegmentID",
    "ignition": "FireIgnitionDate",
    "storm_start": "StormStart",
    "response": "Response",
    "basin_area": "UpstreamArea",
    "gage": "GageName",
}
DATE_FORMAT = "%Y%m%d"
PREDICTOR_COLS = ["PeakI15", "HM23", "dNBR/1000", "KF"]
BASIN_AREA_UNITS_TO_KM2 = 1.0
AREA_RATIO_RANGE = (0.5, 2.0)

SCIENCEBASE_URL = "https://www.sciencebase.gov/catalog"
SCIENCEBASE_PARENTS = {
    "2013-2023": "68bb4218d4be025329aa1bd0",
    "2024": "68c325ebd4be0260db1940bf",
}
BASIN_FILE_PREFERENCE = ["basin_dfpredictions_15min_24mmh", "basin_dfpredictions", "basin"]
BASIN_KEY_COLUMNS = ["Fire_Segme", "Segment_ID", "SegmentID", "segment_id", "Segment"]

HLS_L30_START = "2013-04-11"
MIN_IGNITION = "2013-06-01"
PRE_FIRE_LOOKBACK_DAYS = 365
PRE_FIRE_GAP_DAYS = 7
POST_FIRE_MIN_DAYS = 10
POST_FIRE_STORM_BUFFER_DAYS = 1

MAX_SCENE_CLOUD = 60
CANDIDATES_PER_WINDOW = 8
MIN_CLEAR_FRACTION = 0.80
CHIP_SIZE = 224
PATCH_SIZE = 16
KEEP_HLS_DOWNLOADS = False

BANDS = {
    "L30": ["B02", "B03", "B04", "B05", "B06", "B07"],
    "S30": ["B02", "B03", "B04", "B8A", "B11", "B12"],
}
HLS_NODATA = -9999
FMASK_BAD_BITS = (1, 2, 3, 4)
FMASK_FILL = 255

PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

N_OUTER_FOLDS = 5
N_DEV_GROUPS = 2
RANDOM_SEED = 2026

ONLY_FIRES = []
