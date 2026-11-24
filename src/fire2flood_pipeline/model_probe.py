import argparse

import numpy as np
import pandas as pd

import config as C
from evaluation import assign_folds, best_threshold, fold_splits, load_cohort, prediction_frame, run_label, summarize
from models import logistic_start
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
            return (self.tab(tab) + self.img((a.unsqueeze(-1) * x).sum(1))).squeeze(-1), a

    return Probe()


def batches(n, size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield order[i : i + size]


def data_slice(data, rows):
    import torch

    toks, wmap, pre_i, post_i, tab = data
    as_tensor = lambda a: torch.from_numpy(a)
    return (as_tensor(toks[pre_i[rows]].astype(np.float32)), as_tensor(toks[post_i[rows]].astype(np.float32)),
            as_tensor(wmap[post_i[rows]]), as_tensor(tab[rows]))


def evaluate(model, data, rows):
    import torch

    model.eval()
    with torch.no_grad():
        return [(rows[b], *model(*data_slice(data, rows[b]))) for b in batches(len(rows), 512)]


def set_linear(layer, start):
    import torch

    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(start[0]).view_as(layer.weight))
        layer.bias.fill_(float(start[1]))


def fit_one(data, y, train_rows, val_rows, seed, args, start):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(n_tab=data[4].shape[1])
    set_linear(model.tab, start)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn, yt = torch.nn.BCEWithLogitsLoss(), torch.from_numpy(y.astype(np.float32))
    best, best_state, bad = np.inf, None, 0
    for epoch in range(args.epochs):
        model.train()
        for b in batches(len(train_rows), args.batch_size, rng):
            loss = loss_fn(model(*data_slice(data, train_rows[b]))[0], yt[train_rows[b]])
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses = [loss_fn(lg, yt[r]).item() * len(r) for r, lg, _ in evaluate(model, data, val_rows)]
        val_loss = sum(losses) / len(val_rows)
        if val_loss < best - 1e-4:
            best, bad, best_state = val_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        elif (bad := bad + 1) >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best, epoch + 1


def predict_rows(model, data, rows):
    import torch

    return np.concatenate([torch.sigmoid(lg).numpy() for _, lg, _ in evaluate(model, data, rows)])


def split_validation(df, y, tr_all, rng, share):
    groups = df.loc[tr_all, "split_group"].unique()
    val_groups = set(rng.choice(groups, size=max(1, int(len(groups) * share)), replace=False))
    is_val = df.loc[tr_all, "split_group"].isin(val_groups).values
    tr, va = tr_all[~is_val], tr_all[is_val]
    if y[va].min() == y[va].max():
        va, tr = tr_all[is_val | (rng.random(len(tr_all)) < 0.15)], tr_all[~is_val]
    return tr, va


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="fire", choices=["fire", "region"])
    for flag, typ, default in (("--seeds", int, 3), ("--epochs", int, 30), ("--patience", int, 5),
                               ("--batch-size", int, 128), ("--lr", float, 1e-3), ("--weight-decay", float, 1e-2),
                               ("--val-share", float, 0.2), ("--threads", int, 0)):
        ap.add_argument(flag, type=typ, default=default)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    idx, toks, wmap = load_tokens()
    df = load_cohort("image")
    for w in ("pre", "post"):
        sub = idx[idx.window == w][["fire_key", "segment_id", "tok_idx"]].rename(columns={"tok_idx": f"{w}_i"})
        df = df.merge(sub, on=["fire_key", "segment_id"], how="left")
    if (missing := df[["pre_i", "post_i"]].isna().any(axis=1)).any():
        print(f"Warning: {int(missing.sum())} rows have no token grids and are dropped")
        df = df[~missing].reset_index(drop=True)

    folds, status = assign_folds(df, args.holdout)
    y, tab = df["response"].values, df[C.PREDICTOR_COLS].values.astype(np.float32)
    rng, parts = np.random.default_rng(C.RANDOM_SEED), []
    for k, tr_mask, te_mask in fold_splits(folds):
        tr_all, te = np.flatnonzero(tr_mask), np.flatnonzero(te_mask)
        tr, va = split_validation(df, y, tr_all, rng, args.val_share)
        mu, sd = tab[tr].mean(0), tab[tr].std(0) + 1e-6
        data = (toks, wmap, df["pre_i"].astype(int).values, df["post_i"].astype(int).values,
                ((tab - mu) / sd).astype(np.float32))
        start = logistic_start(data[4][tr], y[tr], df["split_group"].values[tr])
        p_te, p_tr = np.zeros(len(te)), np.zeros(len(tr_all))
        for s in range(args.seeds):
            model, vloss, epochs = fit_one(data, y, tr, va, C.RANDOM_SEED + 100 * k + s, args, start)
            p_te += predict_rows(model, data, te) / args.seeds
            p_tr += predict_rows(model, data, tr_all) / args.seeds
            print(f"fold {k} seed {s}: stopped after {epochs} epochs, validation log-loss {vloss:.4f}")
        parts.append(prediction_frame(df, te_mask, p_te, best_threshold(y[tr_all], p_tr), k))
    print("\nExploratory arm (not part of the pre-registered success rule).")
    summarize(pd.concat(parts, ignore_index=True), "A4_probe_attention", status, run_label("image", args.holdout))


if __name__ == "__main__":
    main()
