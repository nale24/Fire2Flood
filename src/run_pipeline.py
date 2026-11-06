import argparse
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

STEPS = [
    ("clean", "01_clean_inventory"),
    ("download", "00_download_assessments"),
    ("join", "02_join_basins"),
    ("scenes", "03_select_scenes"),
    ("chips", "04_cut_chips"),
]


def main():
    names = [s for s, _ in STEPS]
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=names, default=names[0])
    ap.add_argument("--to", dest="end", choices=names, default=names[-1])
    ap.add_argument("--fires", nargs="*", default=None)
    args = ap.parse_args()
    if args.fires is not None:
        C.ONLY_FIRES = args.fires

    i, j = names.index(args.start), names.index(args.end)
    for name, module in STEPS[i : j + 1]:
        print(f"\n=== {name} ({module}) ===")
        mod = import_module(module)
        mod.main([]) if name == "download" else mod.main()


if __name__ == "__main__":
    main()
