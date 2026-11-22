import argparse

import numpy as np
import pandas as pd

import config as C
from evaluation import assign_folds, best_threshold, load_cohort, prediction_frame, run_label, summarize
from prithvi_tokens import load_tokens


def build_model(n_tab, d_in=1024, d=64, n_tok=49):
    import torch
    from torch import nn

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
            self.pos = nn.Parameter(torch.zeros(n_tok, 2 * d))
            self.wemb = nn.Linear(1, 2 * d)
            self.score = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
            self.img = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, 32), nn.ReLU(), nn.Linear(32, 1))
            self.tab = nn.Linear(n_tab, 1)
            nn.init.zeros_(self.img[-1].weight)
            nn.init.zeros_(self.img[-1].bias)

        def forward(self, pre, post, wmap, tab):
            hp, hq = self.proj(pre), self.proj(post)
            x = torch.cat([hq, hq - hp], dim=-1) + self.pos + self.wemb(wmap.unsqueeze(-1))
            a = torch.softmax(self.score(x).squeeze(-1), dim=-1)
            pooled = (a.unsqueeze(-1) * x).sum(1)
            return (self.tab(tab) + self.img(pooled)).squeeze(-1), a

    return Probe()


def batches(n, size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield order[i : i + size]


def data_slice(data, rows):
    import torch

    toks, wmap, pre_i, post_i, tab = data
    pre = torch.from_numpy(toks[pre_i[rows]].astype(np.float32))
    post = torch.from_numpy(toks[post_i[rows]].astype(np.float32))
    w = torch.from_numpy(wmap[post_i[rows]])
    t = torch.from_numpy(tab[rows])
    return pre, post, w, t


def fit_one(data, y, train_rows, val_rows, seed, args):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(n_tab=data[4].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, bad = np.inf, None, 0
    yt = torch.from_numpy(y.astype(np.float32))
    for epoch in range(args.epochs):
        model.train()
        for b in batches(len(train_rows), args.batch_size, rng):
            rows = train_rows[b]
            logit, _ = model(*data_slice(data, rows))
            loss = loss_fn(logit, yt[rows])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for b in batches(len(val_rows), 512):
                rows = val_rows[b]
                logit, _ = model(*data_slice(data, rows))
                vl.append(loss_fn(logit, yt[rows]).item() * len(rows))
            val_loss = sum(vl) / len(val_rows)
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best, epoch + 1


def predict_rows(model, data, rows):
    import torch

    model.eval()
    out, att = [], []
    with torch.no_grad():
        for b in batches(len(rows), 512):
            logit, a = model(*data_slice(data, rows[b]))
            out.append(torch.sigmoid(logit).numpy())
            att.append(a.numpy())
    return np.concatenate(out), np.concatenate(att)


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="fire", choices=["fire", "region"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-share", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    df = load_cohort("image")
    idx, toks, wmap = load_tokens()
    pre = idx[idx.window == "pre"][["fire_key", "segment_id", "tok_idx"]].rename(columns={"tok_idx": "pre_i"})
    post = idx[idx.window == "post"][["fire_key", "segment_id", "tok_idx"]].rename(columns={"tok_idx": "post_i"})
    df = df.merge(pre, on=["fire_key", "segment_id"], how="left").merge(post, on=["fire_key", "segment_id"], how="left")
    missing = df[["pre_i", "post_i"]].isna().any(axis=1)
    if missing.any():
        print(f"Warning: {int(missing.sum())} rows have no token grids and are dropped")
        df = df[~missing].reset_index(drop=True)

    folds, status = assign_folds(df, args.holdout)
    y = df["response"].values
    rng = np.random.default_rng(C.RANDOM_SEED)
    parts = []
    for k in sorted(f for f in folds.unique() if f >= 0):
        tr_all = np.flatnonzero(((folds != k) & (folds >= 0)).values)
        te = np.flatnonzero((folds == k).values)
        groups = df.loc[tr_all, "split_group"].unique()
        val_groups = set(rng.choice(groups, size=max(1, int(len(groups) * args.val_share)), replace=False))
        is_val = df.loc[tr_all, "split_group"].isin(val_groups).values
        tr, va = tr_all[~is_val], tr_all[is_val]
        if y[va].min() == y[va].max():
            va, tr = tr_all[is_val | (rng.random(len(tr_all)) < 0.15)], tr_all[~is_val]

        tab = df[C.PREDICTOR_COLS].values.astype(np.float32)
        mu, sd = tab[tr].mean(0), tab[tr].std(0) + 1e-6
        data = (toks, wmap, df["pre_i"].astype(int).values, df["post_i"].astype(int).values, ((tab - mu) / sd).astype(np.float32))

        p_te, p_tr = np.zeros(len(te)), np.zeros(len(tr_all))
        for s in range(args.seeds):
            model, vloss, epochs = fit_one(data, y, tr, va, C.RANDOM_SEED + 100 * k + s, args)
            p_te += predict_rows(model, data, te)[0] / args.seeds
            p_tr += predict_rows(model, data, tr_all)[0] / args.seeds
            print(f"fold {k} seed {s}: stopped after {epochs} epochs, validation log-loss {vloss:.4f}")
        t = best_threshold(y[tr_all], p_tr)
        parts.append(prediction_frame(df, pd.Series(np.isin(np.arange(len(df)), te), index=df.index), p_te, t, k))

    print("\nExploratory arm (not part of the pre-registered success rule).")
    summarize(pd.concat(parts, ignore_index=True), "A4_probe_attention", status, run_label("image", args.holdout))


if __name__ == "__main__":
    main()
