import argparse

import numpy as np
import pandas as pd

import config as C
from common import selected
from cut_chips import load_chip_for_prithvi
from prithvi_features import encode, load_encoder

WINDOW = 7


def token_dir():
    d = C.OUT_DIR / "prithvi_tokens"
    d.mkdir(parents=True, exist_ok=True)
    return d


def window_bounds(basin):
    p, n = C.PATCH_SIZE, C.CHIP_SIZE // C.PATCH_SIZE
    if basin.any():
        rows, cols = np.nonzero(basin)
        r, c = int(rows.mean()) // p, int(cols.mean()) // p
    else:
        r = c = n // 2
    r0 = min(max(0, r - WINDOW // 2), n - WINDOW)
    c0 = min(max(0, c - WINDOW // 2), n - WINDOW)
    return r0, c0


def crop(tokens, basin):
    p, n = C.PATCH_SIZE, C.CHIP_SIZE // C.PATCH_SIZE
    grid = tokens[1:].reshape(n, n, -1)
    frac = basin.reshape(n, p, n, p).mean(axis=(1, 3))
    r0, c0 = window_bounds(basin)
    win = grid[r0 : r0 + WINDOW, c0 : c0 + WINDOW].reshape(WINDOW * WINDOW, -1)
    wmap = frac[r0 : r0 + WINDOW, c0 : c0 + WINDOW].reshape(-1)
    return win, wmap


def run(encoder, manifest, device, batch_size):
    done = 0
    for fire, grp in manifest.groupby("fire_key"):
        out = token_dir() / f"{fire}.npz"
        if out.exists():
            done += len(grp)
            continue
        keys, toks, wmaps = [], [], []
        rows = list(grp.itertuples())
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            loaded = [load_chip_for_prithvi(r.chip_path) for r in batch]
            enc = encode(encoder, np.stack([a for a, _ in loaded]).astype(np.float32), device)
            for t, (_, m), r in zip(enc, loaded, batch):
                win, wmap = crop(t, m)
                toks.append(win.astype(np.float16))
                wmaps.append(wmap.astype(np.float32))
                keys.append(f"{r.segment_id}|{r.window}")
            done += len(batch)
            print(f"\r  {fire}: {min(i + batch_size, len(rows))}/{len(rows)} chips (total {done}/{len(manifest)})",
                  end="", flush=True)
        np.savez(out, keys=np.array(keys), tokens=np.stack(toks), wmap=np.stack(wmaps))
        print()


def load_tokens():
    keys, toks, wmaps, fires = [], [], [], []
    for f in sorted(token_dir().glob("*.npz")):
        z = np.load(f)
        keys += [str(k) for k in z["keys"]]
        fires += [f.stem] * len(z["keys"])
        toks.append(z["tokens"])
        wmaps.append(z["wmap"])
    if not keys:
        raise SystemExit("No token grids found. Run prithvi_tokens.py first.")
    idx = pd.DataFrame({"fire_key": fires, "key": keys})
    idx[["segment_id", "window"]] = idx["key"].str.split("|", expand=True)
    idx["tok_idx"] = np.arange(len(idx))
    return idx.drop(columns="key"), np.concatenate(toks), np.concatenate(wmaps)


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
    needed = ev[ev["cohort_image"]][["fire_key", "segment_id"]].drop_duplicates()
    cm = selected(pd.read_parquet(C.OUT_DIR / "chip_manifest.parquet").merge(needed, on=["fire_key", "segment_id"]))
    print(f"{len(cm):,} chips from {cm.fire_key.nunique()} fires on {args.device}; saving {WINDOW}x{WINDOW} token windows")
    run(load_encoder(args.device), cm, args.device, args.batch_size)
    print(f"Token grids saved in {token_dir()}")


if __name__ == "__main__":
    main()
