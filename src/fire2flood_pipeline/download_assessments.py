import argparse
import shutil
import zipfile
from pathlib import Path

import requests

import config as C
from common import basin_file_rank, ensure_dirs, find_basin_shapefiles, load_inventory, selected

SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def get_json(session, url, params=None):
    r = session.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def list_children(session, parent_id):
    url, items = f"{C.SCIENCEBASE_URL}/items", []
    params = {"parentId": parent_id, "format": "json", "fields": "title,files,hasChildren", "max": 1000}
    while url:
        data = get_json(session, url, params)
        items += data.get("items", [])
        url, params = (data.get("nextlink") or {}).get("url"), None
    return items


def item_files(session, item):
    if "files" in item:
        return item["files"] or []
    return get_json(session, f"{C.SCIENCEBASE_URL}/item/{item['id']}", {"format": "json"}).get("files", []) or []


def matches(aid, title, name):
    return name.lower().endswith(f"_{aid.lower()}.zip") or (f"({aid})" in title and name.lower() == "shapefiles.zip")


def build_index(session, wanted):
    index, stack = {}, list(C.SCIENCEBASE_PARENTS.values())
    while stack and len(index) < len(wanted):
        for item in list_children(session, stack.pop()):
            for f in item_files(session, item):
                for aid in wanted:
                    if aid not in index and matches(aid, item.get("title", ""), f.get("name", "")):
                        index[aid] = f
            if item.get("hasChildren"):
                stack.append(item["id"])
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


def copy_member(z, name, target):
    with z.open(name) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def extract_basins(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        members = pick_members(names)
        for inner in [] if members else [n for n in names if n.lower().endswith(".zip")]:
            nested = copy_member(z, inner, out_dir / Path(inner).name)
            extracted += extract_basins(nested, out_dir)
            nested.unlink()
        extracted += [copy_member(z, m, out_dir / Path(m).name) for m in members]
    return extracted


def download(session, url, path):
    with session.get(url, stream=True, timeout=600) as r, open(path, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    ensure_dirs()
    inv = selected(load_inventory())
    wanted = sorted((inv if args.all else inv[inv["cohort_image_candidate"]])["assessment_id"].dropna().unique())
    todo = [a for a in wanted if args.force or not find_basin_shapefiles(a)]
    print(f"{len(wanted)} assessments needed, {len(wanted) - len(todo)} already on disk")
    if not todo:
        return
    session = requests.Session()
    index = build_index(session, todo)
    for aid in todo:
        if (f := index.get(aid)) is None:
            print(f"{aid}: not found on ScienceBase")
            continue
        zip_path = C.ASSESSMENT_DIR / aid / f["name"]
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{aid}: downloading {f['name']} ({f.get('size', 0) / 1e6:.1f} MB)")
        download(session, f["url"], zip_path)
        files = extract_basins(zip_path, zip_path.parent)
        zip_path.unlink()
        print(f"{aid}: extracted {[p.name for p in files if p.suffix == '.shp']}")
    if missing := [a for a in wanted if not find_basin_shapefiles(a)]:
        print(f"\nStill missing basin files for: {missing}")


if __name__ == "__main__":
    main()
