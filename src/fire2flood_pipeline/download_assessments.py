import argparse
import shutil
import zipfile
from pathlib import Path

import requests

import config as C
from common import basin_file_rank, ensure_dirs, find_basin_shapefiles, load_inventory, selected

SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def list_children(session, parent_id):
    url = f"{C.SCIENCEBASE_URL}/items"
    params = {"parentId": parent_id, "format": "json", "fields": "title,files,hasChildren", "max": 1000}
    items = []
    while url:
        r = session.get(url, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        items += data.get("items", [])
        url = (data.get("nextlink") or {}).get("url")
        params = None
    return items


def item_files(session, item):
    if "files" in item:
        return item["files"] or []
    r = session.get(f"{C.SCIENCEBASE_URL}/item/{item['id']}", params={"format": "json"}, timeout=120)
    r.raise_for_status()
    return r.json().get("files", []) or []


def build_index(session, wanted):
    index = {}
    stack = list(C.SCIENCEBASE_PARENTS.values())
    while stack:
        for item in list_children(session, stack.pop()):
            title = item.get("title", "")
            for f in item_files(session, item):
                name = f.get("name", "")
                for aid in wanted:
                    if aid in index:
                        continue
                    if name.lower().endswith(f"_{aid.lower()}.zip"):
                        index[aid] = f
                    elif f"({aid})" in title and name.lower() == "shapefiles.zip":
                        index[aid] = f
            if item.get("hasChildren"):
                stack.append(item["id"])
        if len(index) == len(wanted):
            break
    return index


def pick_members(names):
    shps = [n for n in names if n.lower().endswith(".shp")]
    if not shps:
        return []
    best = min(basin_file_rank(n) for n in shps)
    chosen = [n for n in shps if basin_file_rank(n) == best]
    if best < len(C.BASIN_FILE_PREFERENCE):
        chosen = sorted(chosen, key=len)[:1]
    stems = {n[:-4].lower() for n in chosen}
    return [n for n in names if n.lower().endswith(SIDECARS) and n[: n.rfind(".")].lower() in stems]


def extract_basins(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        members = pick_members(names)
        for inner in [n for n in names if n.lower().endswith(".zip")] if not members else []:
            nested = out_dir / Path(inner).name
            with z.open(inner) as src, open(nested, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += extract_basins(nested, out_dir)
            nested.unlink()
        for m in members:
            target = out_dir / Path(m).name
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def download(session, url, path):
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    ensure_dirs()
    inv = selected(load_inventory())
    pool = inv if args.all else inv[inv["cohort_image_candidate"]]
    wanted = sorted(pool["assessment_id"].dropna().unique())
    todo = [a for a in wanted if args.force or not find_basin_shapefiles(a)]
    print(f"{len(wanted)} assessments needed, {len(wanted) - len(todo)} already on disk")
    if not todo:
        return

    session = requests.Session()
    index = build_index(session, todo)
    for aid in todo:
        f = index.get(aid)
        if f is None:
            print(f"{aid}: not found on ScienceBase")
            continue
        out_dir = C.ASSESSMENT_DIR / aid
        zip_path = out_dir / f["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{aid}: downloading {f['name']} ({f.get('size', 0) / 1e6:.1f} MB)")
        download(session, f["url"], zip_path)
        files = extract_basins(zip_path, out_dir)
        zip_path.unlink()
        print(f"{aid}: extracted {[p.name for p in files if p.suffix == '.shp']}")

    missing = [a for a in wanted if not find_basin_shapefiles(a)]
    if missing:
        print(f"\nStill missing basin files for: {missing}")


if __name__ == "__main__":
    main()
