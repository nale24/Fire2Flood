import argparse
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

import config as C
from common import selected
from cut_chips import load_chip_for_prithvi

HF_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
WEIGHTS_FILE = "Prithvi_EO_V2_300M.pt"
CONTEXT_TOKENS = 1
N_SIDE = C.CHIP_SIZE // C.PATCH_SIZE


def out_dir(name="prithvi"):
    d = C.OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_config(cfg):
    if list(cfg.get("mean", C.PRITHVI_MEAN)) != C.PRITHVI_MEAN or list(cfg.get("std", C.PRITHVI_STD)) != C.PRITHVI_STD:
        raise SystemExit("Prithvi config mean/std differ from config.py PRITHVI_MEAN/STD; update config.py first")
    if cfg.get("img_size", C.CHIP_SIZE) != C.CHIP_SIZE:
        raise SystemExit(f"Prithvi expects img_size {cfg.get('img_size')}, chips are {C.CHIP_SIZE}")


def load_encoder(device):
    import torch
    from huggingface_hub import hf_hub_download

    code, cfg_path, weights = (hf_hub_download(HF_REPO, f) for f in ("prithvi_mae.py", "config.json", WEIGHTS_FILE))
    spec = importlib.util.spec_from_file_location("prithvi_mae", code)
    mod = sys.modules["prithvi_mae"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = json.loads(open(cfg_path).read())["pretrained_cfg"]
    check_config(cfg)
    model = mod.PrithviMAE(**dict(cfg, num_frames=1))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if "pos_embed" not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if enc_missing := [k for k in missing if k.startswith("encoder.") and "pos_embed" not in k]:
        raise SystemExit(f"{len(enc_missing)} encoder weights missing from the checkpoint, e.g. {enc_missing[:3]}")
    print(f"Loaded {HF_REPO}: {len(missing)} missing (pos_embed/decoder ok), {len(unexpected)} unexpected keys")
    return model.encoder.to(device).eval()


def encode(encoder, x, device):
    import torch

    with torch.no_grad():
        return encoder.forward_features(torch.from_numpy(x).unsqueeze(2).to(device))[-1].float().cpu().numpy()


def basin_center(basin):
    if not basin.any():
        return N_SIDE // 2, N_SIDE // 2
    rows, cols = np.nonzero(basin)
    return int(rows.mean()) // C.PATCH_SIZE, int(cols.mean()) // C.PATCH_SIZE


def token_weights(basin):
    p = C.PATCH_SIZE
    frac = basin.reshape(N_SIDE, p, N_SIDE, p).mean(axis=(1, 3))
    if frac.sum() == 0:
        frac[N_SIDE // 2, N_SIDE // 2] = 1.0
    r, c = basin_center(basin)
    ctx = np.zeros((N_SIDE, N_SIDE))
    ctx[max(0, r - CONTEXT_TOKENS) : r + CONTEXT_TOKENS + 1, max(0, c - CONTEXT_TOKENS) : c + CONTEXT_TOKENS + 1] = 1.0
    return frac / frac.sum(), ctx / ctx.sum()


def pooled(tokens, basin):
    wb, wc = token_weights(basin)
    grid = tokens[1:].reshape(N_SIDE * N_SIDE, -1)
    return wb.reshape(-1) @ grid, wc.reshape(-1) @ grid


def save_pooled(path, keys, results):
    np.savez_compressed(path, keys=np.array(keys), basin=np.vstack([b for b, _ in results]).astype(np.float32),
                        context=np.vstack([c for _, c in results]).astype(np.float32))


def run(encoder, manifest, device, batch_size, folder, per_chip, save):
    done = 0
    for fire, grp in manifest.groupby("fire_key"):
        out = folder / f"{fire}.npz"
        if out.exists():
            done += len(grp)
            continue
        keys, results, rows = [], [], list(grp.itertuples())
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            loaded = [load_chip_for_prithvi(r.chip_path) for r in batch]
            enc = encode(encoder, np.stack([a for a, _ in loaded]).astype(np.float32), device)
            results += [per_chip(t, m) for t, (_, m) in zip(enc, loaded)]
            keys += [f"{r.segment_id}|{r.window}" for r in batch]
            done += len(batch)
            print(f"\r  {fire}: {min(i + batch_size, len(rows))}/{len(rows)} chips (total {done}/{len(manifest)})",
                  end="", flush=True)
        save(out, keys, results)
        print()


def load_features(pooling="basin"):
    rows = [(f.stem, *str(key).split("|"), vec) for f in sorted(out_dir().glob("*.npz"))
            for z in [np.load(f)] for key, vec in zip(z["keys"], z[pooling])]
    if not rows:
        raise SystemExit("No Prithvi features found. Run prithvi_features.py first.")
    df = pd.DataFrame(rows, columns=["fire_key", "segment_id", "window", "vec"])
    wide = df.pivot_table(index=["fire_key", "segment_id"], columns="window", values="vec", aggfunc="first").dropna()
    pre, post = np.stack(wide["pre"].values), np.stack(wide["post"].values)
    return wide.index.to_frame(index=False), np.hstack([pre, post, post - pre])


def main(argv=None, folder="prithvi", per_chip=pooled, save=save_pooled, label=""):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
    needed = ev[ev["cohort_image"]][["fire_key", "segment_id"]].drop_duplicates()
    cm = selected(pd.read_parquet(C.OUT_DIR / "chip_manifest.parquet").merge(needed, on=["fire_key", "segment_id"]))
    cm = cm.head(args.limit) if args.limit else cm
    print(f"{len(cm):,} chips from {cm.fire_key.nunique()} fires on {args.device}{label}")
    run(load_encoder(args.device), cm, args.device, args.batch_size, out_dir(folder), per_chip, save)
    print(f"Saved in {out_dir(folder)}")


if __name__ == "__main__":
    main()
