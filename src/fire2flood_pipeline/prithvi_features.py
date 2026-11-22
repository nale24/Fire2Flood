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


def feature_dir():
    d = C.OUT_DIR / "prithvi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_encoder(device):
    import torch
    from huggingface_hub import hf_hub_download

    code = hf_hub_download(HF_REPO, "prithvi_mae.py")
    cfg_path = hf_hub_download(HF_REPO, "config.json")
    weights = hf_hub_download(HF_REPO, WEIGHTS_FILE)
    spec = importlib.util.spec_from_file_location("prithvi_mae", code)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prithvi_mae"] = mod
    spec.loader.exec_module(mod)

    cfg = json.loads(open(cfg_path).read())["pretrained_cfg"]
    check_config(cfg)
    cfg = dict(cfg, num_frames=1)
    model = mod.PrithviMAE(**cfg)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if "pos_embed" not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    enc_missing = [k for k in missing if k.startswith("encoder.") and "pos_embed" not in k]
    if enc_missing:
        raise SystemExit(f"{len(enc_missing)} encoder weights missing from the checkpoint, e.g. {enc_missing[:3]}")
    print(f"Loaded {HF_REPO}: {len(missing)} missing (pos_embed/decoder ok), {len(unexpected)} unexpected keys")
    return model.encoder.to(device).eval()


def check_config(cfg):
    if list(cfg.get("mean", C.PRITHVI_MEAN)) != C.PRITHVI_MEAN or list(cfg.get("std", C.PRITHVI_STD)) != C.PRITHVI_STD:
        raise SystemExit("Prithvi config mean/std differ from config.py PRITHVI_MEAN/STD; update config.py first")
    if cfg.get("img_size", C.CHIP_SIZE) != C.CHIP_SIZE:
        raise SystemExit(f"Prithvi expects img_size {cfg.get('img_size')}, chips are {C.CHIP_SIZE}")


def token_weights(basin):
    p = C.PATCH_SIZE
    n = C.CHIP_SIZE // p
    frac = basin.reshape(n, p, n, p).mean(axis=(1, 3))
    if frac.sum() == 0:
        frac[n // 2, n // 2] = 1.0
    rows, cols = np.nonzero(basin) if basin.any() else (np.array([C.CHIP_SIZE // 2]), np.array([C.CHIP_SIZE // 2]))
    r, c = int(rows.mean()) // p, int(cols.mean()) // p
    ctx = np.zeros((n, n))
    ctx[max(0, r - CONTEXT_TOKENS) : r + CONTEXT_TOKENS + 1, max(0, c - CONTEXT_TOKENS) : c + CONTEXT_TOKENS + 1] = 1.0
    return frac / frac.sum(), ctx / ctx.sum()


def pool(tokens, basin_masks):
    n = C.CHIP_SIZE // C.PATCH_SIZE
    grid = tokens[:, 1:, :].reshape(tokens.shape[0], n * n, -1)
    out_basin, out_ctx = [], []
    for g, m in zip(grid, basin_masks):
        wb, wc = token_weights(m)
        out_basin.append(wb.reshape(-1) @ g)
        out_ctx.append(wc.reshape(-1) @ g)
    return np.stack(out_basin), np.stack(out_ctx)


def encode(encoder, x, device):
    import torch

    with torch.no_grad():
        t = torch.from_numpy(x).unsqueeze(2).to(device)
        return encoder.forward_features(t)[-1].float().cpu().numpy()


def run(encoder, manifest, device, batch_size):
    done = 0
    for fire, grp in manifest.groupby("fire_key"):
        out = feature_dir() / f"{fire}.npz"
        if out.exists():
            done += len(grp)
            continue
        keys, basin_f, ctx_f = [], [], []
        rows = list(grp.itertuples())
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            loaded = [load_chip_for_prithvi(r.chip_path) for r in batch]
            x = np.stack([a for a, _ in loaded]).astype(np.float32)
            b, c = pool(encode(encoder, x, device), [m for _, m in loaded])
            basin_f.append(b)
            ctx_f.append(c)
            keys += [f"{r.segment_id}|{r.window}" for r in batch]
            done += len(batch)
            print(f"\r  {fire}: {min(i + batch_size, len(rows))}/{len(rows)} chips (total {done}/{len(manifest)})",
                  end="", flush=True)
        np.savez_compressed(out, keys=np.array(keys), basin=np.vstack(basin_f).astype(np.float32),
                            context=np.vstack(ctx_f).astype(np.float32))
        print()


def load_features(pooling="basin"):
    rows = []
    for f in sorted(feature_dir().glob("*.npz")):
        z = np.load(f)
        for key, vec in zip(z["keys"], z[pooling]):
            seg, window = str(key).split("|")
            rows.append((f.stem, seg, window, vec))
    if not rows:
        raise SystemExit("No Prithvi features found. Run prithvi_features.py first.")
    df = pd.DataFrame(rows, columns=["fire_key", "segment_id", "window", "vec"])
    wide = df.pivot_table(index=["fire_key", "segment_id"], columns="window", values="vec", aggfunc="first")
    wide = wide.dropna()
    pre, post = np.stack(wide["pre"].values), np.stack(wide["post"].values)
    feats = np.hstack([pre, post, post - pre])
    return wide.index.to_frame(index=False), feats


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    ev = pd.read_parquet(C.OUT_DIR / "event_table.parquet")
    needed = ev[ev["cohort_image"]][["fire_key", "segment_id"]].drop_duplicates()
    cm = pd.read_parquet(C.OUT_DIR / "chip_manifest.parquet")
    cm = selected(cm.merge(needed, on=["fire_key", "segment_id"]))
    if args.limit:
        cm = cm.head(args.limit)
    print(f"{len(cm):,} chips from {cm.fire_key.nunique()} fires on {args.device}")
    encoder = load_encoder(args.device)
    run(encoder, cm, args.device, args.batch_size)
    print(f"Features saved in {feature_dir()}")


if __name__ == "__main__":
    main()
