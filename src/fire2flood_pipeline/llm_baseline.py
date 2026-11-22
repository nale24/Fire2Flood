import argparse
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from evaluation import run_label, assign_folds, best_threshold, load_cohort, prediction_frame, results_dir, summarize

HEADER = (
    "Each record describes a rainstorm over a recently burned watershed in the western United States "
    "and whether a debris flow occurred.\n\n"
)
RECORD = (
    "Peak 15-minute rainfall intensity: {i15:.0f} mm/h. "
    "Steep, moderately-to-severely burned share of the watershed: {hm23:.0f}%. "
    "Average burn severity (dNBR): {dnbr:.0f}. "
    "Soil erodibility factor: {kf:.2f}."
)
ANSWER = "Debris flow:"
C_GRID = [0.001, 0.01, 0.1, 1.0]


def describe(row):
    return RECORD.format(i15=row["PeakI15"], hm23=100 * row["HM23"], dnbr=1000 * row["dNBR/1000"], kf=row["KF"])


def load_lm(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device).eval(), device


def answer_ids(tok):
    try:
        yes, no = tok.encode(" yes", add_special_tokens=False), tok.encode(" no", add_special_tokens=False)
    except TypeError:
        yes, no = tok.encode(" yes"), tok.encode(" no")
    if yes[0] == no[0]:
        raise SystemExit("The tokenizer gives ' yes' and ' no' the same first token; choose another model.")
    return yes[0], no[0]


def score_prompts(prompts, tok, model, device, batch_size):
    import torch

    yes_id, no_id = answer_ids(tok)
    max_len = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    scores = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            scores.append((logp[:, yes_id] - logp[:, no_id]).cpu().numpy())
            print(f"\r  scored {min(i + batch_size, len(prompts))}/{len(prompts)}", end="", flush=True)
    print()
    return np.concatenate(scores) if scores else np.array([])


def build_prefix(examples):
    lines = [f"{describe(r)}\n{ANSWER} {'yes' if r['response'] == 1 else 'no'}\n\n" for _, r in examples.iterrows()]
    return HEADER + "".join(lines)


def pick_examples(train, k_shot, rng):
    pos = train[train["response"] == 1]
    neg = train[train["response"] == 0]
    n_pos = min(k_shot // 2, len(pos))
    ex = pd.concat([pos.sample(n_pos, random_state=rng.integers(1 << 31)),
                    neg.sample(k_shot - n_pos, random_state=rng.integers(1 << 31))])
    return ex.sample(frac=1, random_state=rng.integers(1 << 31))


def run_fewshot(df, folds, args, tok, model, device):
    rng = np.random.default_rng(C.RANDOM_SEED)
    parts, prompts_log = [], []
    for k in sorted(f for f in folds.unique() if f >= 0):
        train, test_mask = df[(folds != k) & (folds >= 0)], folds == k
        examples = pick_examples(train, args.k_shot, rng)
        prefix = build_prefix(examples)
        cal = train.drop(examples.index).sample(min(args.n_cal, len(train) - len(examples)),
                                                random_state=rng.integers(1 << 31))
        print(f"fold {k}: {int(test_mask.sum())} test rows, {len(cal)} calibration rows")
        cal_s = score_prompts([prefix + describe(r) + f"\n{ANSWER}" for _, r in cal.iterrows()],
                              tok, model, device, args.batch_size)
        test_s = score_prompts([prefix + describe(r) + f"\n{ANSWER}" for _, r in df[test_mask].iterrows()],
                               tok, model, device, args.batch_size)
        platt = LogisticRegression().fit(cal_s.reshape(-1, 1), cal["response"].values)
        p_cal = platt.predict_proba(cal_s.reshape(-1, 1))[:, 1]
        p = platt.predict_proba(test_s.reshape(-1, 1))[:, 1]
        d = prediction_frame(df, test_mask, p, best_threshold(cal["response"].values, p_cal), k)
        d["raw_score"] = test_s
        parts.append(d)
        prompts_log.append(dict(fold=int(k), prefix=prefix, example_row_ids=[int(x) for x in examples["row_id"]]))
    (results_dir() / f"GPT2_fewshot_{run_label(args.cohort, args.holdout)}_prompts.json").write_text(json.dumps(prompts_log, indent=2))
    return pd.concat(parts, ignore_index=True)


def embed(texts, tok, model, device, batch_size):
    import torch

    tok.padding_side = "right"
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
            print(f"\r  embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    tok.padding_side = "left"
    return np.vstack(out)


def run_embed(df, folds, args, tok, model, device):
    X = embed([describe(r) for _, r in df.iterrows()], tok, model, device, args.batch_size * 4)
    parts = []
    for k in sorted(f for f in folds.unique() if f >= 0):
        tr, test_mask = ((folds != k) & (folds >= 0)).values, (folds == k).values
        n_comp = min(args.pca, X.shape[1], tr.sum() - 1)
        pipe = make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=C.RANDOM_SEED),
                             LogisticRegression(max_iter=3000))
        g = df.loc[tr, "split_group"].values
        search = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, scoring="neg_log_loss",
                              cv=GroupKFold(n_splits=min(5, len(np.unique(g)))), n_jobs=-1)
        search.fit(X[tr], df.loc[tr, "response"].values, groups=g)
        t = best_threshold(df.loc[tr, "response"].values, search.predict_proba(X[tr])[:, 1])
        parts.append(prediction_frame(df, pd.Series(test_mask, index=df.index), search.predict_proba(X[test_mask])[:, 1], t, k))
    return pd.concat(parts, ignore_index=True)


def run_probe(df, folds, args, tok, model, device):
    rng = np.random.default_rng(C.RANDOM_SEED + 1)
    k = sorted(f for f in folds.unique() if f >= 0)[0]
    train, test = df[(folds != k) & (folds >= 0)], df[folds == k]
    prefix = build_prefix(pick_examples(train, args.k_shot, rng))
    pos, neg = test[test.response == 1], test[test.response == 0]
    rows = pd.concat([pos.sample(min(10, len(pos)), random_state=1), neg.sample(min(10, len(neg)), random_state=1)])

    def named(r):
        when = pd.Timestamp(r["storm_start"]).strftime("%B %Y")
        return f"This storm hit the {r['fire_name']} Fire burn area in {when}. " + describe(r)

    plain = score_prompts([prefix + describe(r) + f"\n{ANSWER}" for _, r in rows.iterrows()], tok, model, device, args.batch_size)
    with_names = score_prompts([prefix + named(r) + f"\n{ANSWER}" for _, r in rows.iterrows()], tok, model, device, args.batch_size)
    out = rows[["row_id", "fire_name", "response"]].assign(score_anonymous=plain, score_named=with_names,
                                                           shift=with_names - plain)
    out.to_csv(results_dir() / f"GPT2_memorization_probe_{args.cohort}.csv", index=False)
    print("\nMemorization probe: mean change in yes-vs-no score when fire name and date are added")
    print(out.groupby("response")["shift"].agg(["mean", "std", "size"]).round(4).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fewshot", choices=["fewshot", "embed", "probe"])
    ap.add_argument("--cohort", default="image_candidate", choices=["tabular", "image_candidate", "image"])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--k-shot", type=int, default=8)
    ap.add_argument("--n-cal", type=int, default=300)
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--holdout", default="fire", choices=["fire", "region"])
    args = ap.parse_args(argv)

    df = load_cohort(args.cohort)
    if args.max_rows:
        frac = min(1.0, args.max_rows / len(df))
        df = df.groupby(["split_group", "response"]).sample(frac=frac, random_state=0).reset_index(drop=True)
        df["row_id"] = np.arange(len(df))
    folds, status = assign_folds(df, args.holdout)
    tok, model, device = load_lm(args.model)
    print(f"{args.model} on {device}; {len(df):,} rows")

    if args.mode == "probe":
        run_probe(df, folds, args, tok, model, device)
        return
    preds = run_fewshot(df, folds, args, tok, model, device) if args.mode == "fewshot" else run_embed(df, folds, args, tok, model, device)
    arm = f"{args.model.replace('/', '-').upper()}_{args.mode}"
    summarize(preds, arm, status, run_label(args.cohort, args.holdout))


if __name__ == "__main__":
    main()
