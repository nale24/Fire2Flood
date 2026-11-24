import argparse
import re
import time

import numpy as np
import pandas as pd

import config as C
import prithvi_features as P
from cut_chips import load_chip_for_prithvi
from evaluation import assign_folds, best_threshold, fold_splits, load_cohort, prediction_frame, run_label, summarize
from model_probe import batches, split_validation

PROTOCOL = C.ROOT / "reports" / "protocol_finetune.md"
BLOCK = re.compile(r"blocks\.(\d+)\.")


def load_chips(df, device):
    import torch

    paths = pd.unique(pd.concat([df["chip_path_pre"], df["chip_path_post"]]))
    index = {p: i for i, p in enumerate(paths)}
    xs = np.zeros((len(paths), len(C.PRITHVI_MEAN), C.CHIP_SIZE, C.CHIP_SIZE), np.float16)
    ms = np.zeros((len(paths), C.CHIP_SIZE, C.CHIP_SIZE), np.float16)
    for i, p in enumerate(paths):
        x, m = load_chip_for_prithvi(p)
        xs[i], ms[i] = x, m
        if (i + 1) % 200 == 0 or i + 1 == len(paths):
            print(f"\r  loaded {i + 1}/{len(paths)} chips", end="", flush=True)
    print()
    pre = df["chip_path_pre"].map(index).values
    post = df["chip_path_post"].map(index).values
    return torch.from_numpy(xs).to(device), torch.from_numpy(ms).to(device), pre, post


def set_trainable(encoder, unfreeze):
    ids = sorted({int(m.group(1)) for n, _ in encoder.named_parameters() if (m := BLOCK.search(n))})
    keep = set(ids[-unfreeze:]) if unfreeze else set()
    for n, p in encoder.named_parameters():
        m = BLOCK.search(n)
        p.requires_grad = int(m.group(1)) in keep if m else (unfreeze > 0 and n.startswith("norm"))
    n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder: {len(ids)} blocks, last {unfreeze} unfrozen, {n_train / 1e6:.1f}M trainable parameters")


def build_model(encoder, n_tab, d=1024):
    import torch
    from torch import nn

    center = (P.N_SIDE // 2) * P.N_SIDE + P.N_SIDE // 2

    class FineTuned(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.img = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(0.3), nn.Linear(2 * d, 64), nn.GELU(),
                                     nn.Linear(64, 1))
            self.tab = nn.Linear(n_tab, 1)
            nn.init.zeros_(self.img[-1].weight)
            nn.init.zeros_(self.img[-1].bias)

        def embed(self, x, mask):
            tok = self.encoder.forward_features(x.unsqueeze(2))[-1][:, 1:].float()
            w = nn.functional.avg_pool2d(mask.unsqueeze(1).float(), C.PATCH_SIZE).flatten(1)
            empty = w.sum(1) == 0
            w[empty, center] = 1.0
            return ((w / w.sum(1, keepdim=True)).unsqueeze(-1) * tok).sum(1)

        def forward(self, pre, post, mask, tab):
            hp, hq = self.embed(pre, mask), self.embed(post, mask)
            return (self.tab(tab) + self.img(torch.cat([hq, hq - hp], dim=-1))).squeeze(-1)

    return FineTuned()


def augment(pre, post, mask, rng):
    import torch

    k, flip = int(rng.integers(4)), bool(rng.integers(2))

    def f(t):
        t = torch.rot90(t, k, dims=(-2, -1))
        return torch.flip(t, dims=(-1,)) if flip else t

    return f(pre), f(post), f(mask)


def batch_inputs(data, rows, rng=None):
    import torch

    xs, ms, pre_i, post_i, tab = data
    idx = lambda a: torch.as_tensor(a[rows], device=xs.device)
    pre, post, mask = xs[idx(pre_i)].float(), xs[idx(post_i)].float(), ms[idx(post_i)]
    if rng is not None:
        pre, post, mask = augment(pre, post, mask, rng)
    return pre, post, mask, torch.as_tensor(tab[rows], device=xs.device)


def run_rows(model, data, rows, batch_size, amp):
    import torch

    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for b in batches(len(rows), batch_size):
            out.append(model(*batch_inputs(data, rows[b])).float())
    return torch.cat(out)


def trainable_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def fit_one(model, data, y, tr, va, seed, args, amp):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    head = list(model.img.parameters()) + list(model.tab.parameters())
    groups = [{"params": enc, "lr": args.lr_encoder}] if enc else []
    opt = torch.optim.AdamW(groups + [{"params": head, "lr": args.lr_head}], weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    yt = torch.as_tensor(y.astype(np.float32), device=data[0].device)
    best, best_state, bad, epoch = np.inf, trainable_state(model), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in batches(len(tr), args.batch_size, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(*batch_inputs(data, tr[b], rng)).float()
            loss = loss_fn(logits, yt[tr[b]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc + head, 1.0)
            opt.step()
        val = loss_fn(run_rows(model, data, va, args.batch_size * 2, amp), yt[va]).item()
        print(f"    epoch {epoch}: validation log-loss {val:.4f}", flush=True)
        if val < best - 1e-4:
            best, bad, best_state = val, 0, trainable_state(model)
        elif (bad := bad + 1) >= args.patience:
            break
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n])
    return best, epoch


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="fire", choices=["fire", "region"])
    ap.add_argument("--unfreeze", type=int, default=2)
    for flag, typ, default in (("--seeds", int, 3), ("--epochs", int, 15), ("--patience", int, 3),
                               ("--batch-size", int, 32), ("--lr-encoder", float, 1e-5), ("--lr-head", float, 1e-3),
                               ("--weight-decay", float, 0.05), ("--val-share", float, 0.2)):
        ap.add_argument(flag, type=typ, default=default)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if not PROTOCOL.exists():
        raise SystemExit(f"{PROTOCOL} not found. Commit the fine-tuning rule before running.")
    if args.smoke:
        args.seeds, args.epochs = 1, 1

    t0 = time.time()
    amp = args.device.startswith("cuda")
    df = load_cohort("image").dropna(subset=["chip_path_pre", "chip_path_post"]).reset_index(drop=True)
    for col in ("chip_path_pre", "chip_path_post"):
        df[col] = df[col].str.replace("\\", "/", regex=False)
    folds, status = assign_folds(df, args.holdout)
    xs, ms, pre_i, post_i = load_chips(df, args.device)
    y, tab = df["response"].values, df[C.PREDICTOR_COLS].values.astype(np.float32)
    encoder = P.load_encoder(args.device)
    set_trainable(encoder, args.unfreeze)
    model = build_model(encoder, n_tab=tab.shape[1]).to(args.device)
    initial = {n: v.detach().clone() for n, v in model.state_dict().items()}

    rng, parts = np.random.default_rng(C.RANDOM_SEED), []
    for k, tr_mask, te_mask in fold_splits(folds):
        tr_all, te = np.flatnonzero(tr_mask), np.flatnonzero(te_mask)
        tr, va = split_validation(df, y, tr_all, rng, args.val_share)
        if args.smoke:
            tr, va, te = tr[:256], va[:128], te[:128]
            te_mask = np.isin(np.arange(len(df)), te)
        mu, sd = tab[tr].mean(0), tab[tr].std(0) + 1e-6
        data = (xs, ms, pre_i, post_i, ((tab - mu) / sd).astype(np.float32))
        p_te, p_tr = np.zeros(len(te)), np.zeros(len(tr_all))
        for s in range(args.seeds):
            model.load_state_dict(initial)
            seed = C.RANDOM_SEED + 100 * k + s
            vloss, epochs = fit_one(model, data, y, tr, va, seed, args, amp)
            p_te += torch.sigmoid(run_rows(model, data, te, args.batch_size * 2, amp)).cpu().numpy() / args.seeds
            p_tr += torch.sigmoid(run_rows(model, data, tr_all, args.batch_size * 2, amp)).cpu().numpy() / args.seeds
            print(f"fold {k} seed {s}: {epochs} epochs, best validation log-loss {vloss:.4f}, "
                  f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)
        parts.append(prediction_frame(df, te_mask, p_te, best_threshold(y[tr_all], p_tr), k))
        if args.smoke:
            print(f"Smoke test passed in {(time.time() - t0) / 60:.1f} min; nothing saved.")
            return

    arm = f"A4_finetune_last{args.unfreeze}"
    summarize(pd.concat(parts, ignore_index=True), arm, status, run_label("image", args.holdout))
    print(f"Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
