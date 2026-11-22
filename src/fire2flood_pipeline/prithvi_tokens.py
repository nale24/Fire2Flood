import numpy as np
import pandas as pd

import config as C
import prithvi_features as P

WINDOW = 7
FOLDER = "prithvi_tokens"


def window_bounds(basin):
    r, c = P.basin_center(basin)
    clamp = lambda v: min(max(0, v - WINDOW // 2), P.N_SIDE - WINDOW)
    return clamp(r), clamp(c)


def crop(tokens, basin):
    p, n = C.PATCH_SIZE, P.N_SIDE
    r0, c0 = window_bounds(basin)
    win = tokens[1:].reshape(n, n, -1)[r0 : r0 + WINDOW, c0 : c0 + WINDOW].reshape(WINDOW * WINDOW, -1)
    wmap = basin.reshape(n, p, n, p).mean(axis=(1, 3))[r0 : r0 + WINDOW, c0 : c0 + WINDOW].reshape(-1)
    return win.astype(np.float16), wmap.astype(np.float32)


def save_tokens(path, keys, results):
    np.savez(path, keys=np.array(keys), tokens=np.stack([t for t, _ in results]),
             wmap=np.stack([w for _, w in results]))


def load_tokens():
    files = sorted(P.out_dir(FOLDER).glob("*.npz"))
    if not files:
        raise SystemExit("No token grids found. Run prithvi_tokens.py first.")
    zs = [np.load(f) for f in files]
    idx = pd.DataFrame({"fire_key": [f.stem for f, z in zip(files, zs) for _ in z["keys"]],
                        "key": [str(k) for z in zs for k in z["keys"]]})
    idx[["segment_id", "window"]] = idx["key"].str.split("|", expand=True)
    idx["tok_idx"] = np.arange(len(idx))
    return idx.drop(columns="key"), np.concatenate([z["tokens"] for z in zs]), np.concatenate([z["wmap"] for z in zs])


def main(argv=None):
    P.main(argv, folder=FOLDER, per_chip=crop, save=save_tokens, label=f"; saving {WINDOW}x{WINDOW} token windows")


if __name__ == "__main__":
    main()
