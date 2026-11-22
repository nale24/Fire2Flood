import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import clean_inventory, cut_chips, download_assessments, join_basins, select_scenes

STEPS = {"clean": clean_inventory, "download": download_assessments, "join": join_basins, "scenes": select_scenes,
         "chips": cut_chips}


def main():
    names = list(STEPS)
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=names, default=names[0])
    ap.add_argument("--to", dest="end", choices=names, default=names[-1])
    ap.add_argument("--fires", nargs="*", default=None)
    args = ap.parse_args()
    if args.fires is not None:
        C.ONLY_FIRES = args.fires
    for name in names[names.index(args.start) : names.index(args.end) + 1]:
        print(f"\n=== {name} ({STEPS[name].__name__}) ===")
        STEPS[name].main([]) if name == "download" else STEPS[name].main()


if __name__ == "__main__":
    main()
