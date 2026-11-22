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
from evaluation import (assign_folds, best_threshold, cross_validate, fold_splits, load_cohort, prediction_frame,
                        results_dir, run_label, summarize)

HEADER = ("Each record describes a rainstorm over a recently burned watershed in the western United States "
          "and whether a debris flow occurred.\n\n")
RECORD = ("Peak 15-minute rainfall intensity: {i15:.0f} mm/h. Steep, moderately-to-severely burned share of the "
          "watershed: {hm23:.0f}%. Average burn severity (dNBR): {dnbr:.0f}. Soil erodibility factor: {kf:.2f}.")
ANSWER = "Debris flow:"
C_GRID = [0.001, 0.01, 0.1, 1.0]


def describe(row):
    return RECORD.format(i15=row["PeakI15"], hm23=100 * row["HM23"], dnbr=1000 * row["dNBR/1000"], kf=row["KF"])


def load_lm(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token, tok.padding_side, tok.truncation_side = tok.eos_token, "left", "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, AutoModelForCausalLM.from_pretrained(name).to(device).eval(), device


def answer_ids(tok):
    try:
        yes, no = tok.encode(" yes", add_special_tokens=False), tok.encode(" no", add_special_tokens=False)
    except TypeError:
        yes, no = tok.encode(" yes"), tok.encode(" no")
    if yes[0] == no[0]:
        raise SystemExit("The tokenizer gives ' yes' and ' no' the same first token; choose another model.")
    return yes[0], no[0]


def batched(model_fn, texts, tok, device, batch_size, label, **enc_kw):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, **enc_kw)
            out.append(model_fn({k: v.to(device) for k, v in enc.items()}))
            print(f"\r  {label} {min(i + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    return out


def score_prompts(prompts, tok, model, device, batch_size):
    import torch

    yes_id, no_id = answer_ids(tok)
    max_len = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)

    def score(enc):
        logp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
        return (logp[:, yes_id] - logp[:, no_id]).cpu().numpy()

    out = batched(score, prompts, tok, device, batch_size, "scored", max_length=max_len)
    return np.concatenate(out) if out else np.array([])


def embed(texts, tok, model, device, batch_size):
    def pool(enc):
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()

    tok.padding_side = "right"
    out = batched(pool, texts, tok, device, batch_size, "embedded", max_length=128)
    tok.padding_side = "left"
    return np.vstack(out)


def build_prefix(examples):
    return HEADER + "".join(f"{describe(r)}\n{ANSWER} {'yes' if r['response'] == 1 else 'no'}\n\n"
                            for _, r in examples.iterrows())


def prompts(prefix, rows, fmt=describe):
    return [prefix + fmt(r) + f"\n{ANSWER}" for _, r in rows.iterrows()]


def pick_examples(train, k_shot, rng):
    pos, neg = train[train["response"] == 1], train[train["response"] == 0]
    n_pos = min(k_shot // 2, len(pos))
    ex = pd.concat([pos.sample(n_pos, random_state=rng.integers(1 << 31)),
                    neg.sample(k_shot - n_pos, random_state=rng.integers(1 << 31))])
    return ex.sample(frac=1, random_state=rng.integers(1 << 31))


def run_fewshot(df, folds, args, lm):
    rng = np.random.default_rng(C.RANDOM_SEED)
    parts, log = [], []
    for k, tr, te in fold_splits(folds):
        train = df[tr]
        examples = pick_examples(train, args.k_shot, rng)
        prefix = build_prefix(examples)
        cal = train.drop(examples.index).sample(min(args.n_cal, len(train) - len(examples)),
                                                random_state=rng.integers(1 << 31))
        print(f"fold {k}: {int(te.sum())} test rows, {len(cal)} calibration rows")
        cal_s = score_prompts(prompts(prefix, cal), *lm, args.batch_size)
        test_s = score_prompts(prompts(prefix, df[te]), *lm, args.batch_size)
        platt = LogisticRegression().fit(cal_s.reshape(-1, 1), cal["response"].values)
        t = best_threshold(cal["response"].values, platt.predict_proba(cal_s.reshape(-1, 1))[:, 1])
        parts.append(prediction_frame(df, te, platt.predict_proba(test_s.reshape(-1, 1))[:, 1], t, k)
                     .assign(raw_score=test_s))
        log.append(dict(fold=int(k), prefix=prefix, example_row_ids=[int(x) for x in examples["row_id"]]))
    out = results_dir() / f"GPT2_fewshot_{run_label(args.cohort, args.holdout)}_prompts.json"
    out.write_text(json.dumps(log, indent=2))
    return pd.concat(parts, ignore_index=True)


def run_embed(df, folds, args, lm):
    X = embed([describe(r) for _, r in df.iterrows()], *lm, args.batch_size * 4)

    def make_search(g):
        pca = PCA(n_components=min(args.pca, X.shape[1], len(g) - 1), random_state=C.RANDOM_SEED)
        pipe = make_pipeline(StandardScaler(), pca, LogisticRegression(max_iter=3000))
        return GridSearchCV(pipe, {"logisticregression__C": C_GRID}, scoring="neg_log_loss",
                            cv=GroupKFold(n_splits=min(5, len(np.unique(g)))), n_jobs=-1)

    return cross_validate(df, X, folds, make_search)


def run_probe(df, folds, args, lm):
    rng = np.random.default_rng(C.RANDOM_SEED + 1)
    _, tr, te = next(fold_splits(folds))
    prefix = build_prefix(pick_examples(df[tr], args.k_shot, rng))
    test = df[te]
    groups = (test[test.response == 1], test[test.response == 0])
    rows = pd.concat([g.sample(min(10, len(g)), random_state=1) for g in groups])

    def named(r):
        when = f"{pd.Timestamp(r['storm_start']):%B %Y}"
        return f"This storm hit the {r['fire_name']} Fire burn area in {when}. " + describe(r)

    plain = score_prompts(prompts(prefix, rows), *lm, args.batch_size)
    with_names = score_prompts(prompts(prefix, rows, named), *lm, args.batch_size)
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
    for flag, default in (("--k-shot", 8), ("--n-cal", 300), ("--pca", 32), ("--batch-size", 16), ("--max-rows", 0)):
        ap.add_argument(flag, type=int, default=default)
    ap.add_argument("--holdout", default="fire", choices=["fire", "region"])
    args = ap.parse_args(argv)

    df = load_cohort(args.cohort)
    if args.max_rows:
        frac = min(1.0, args.max_rows / len(df))
        df = df.groupby(["split_group", "response"]).sample(frac=frac, random_state=0).reset_index(drop=True)
        df["row_id"] = np.arange(len(df))
    folds, status = assign_folds(df, args.holdout)
    lm = load_lm(args.model)
    print(f"{args.model} on {lm[2]}; {len(df):,} rows")
    if args.mode == "probe":
        return run_probe(df, folds, args, lm)
    preds = (run_fewshot if args.mode == "fewshot" else run_embed)(df, folds, args, lm)
    arm = f"{args.model.replace('/', '-').upper()}_{args.mode}"
    summarize(preds, arm, status, run_label(args.cohort, args.holdout))


if __name__ == "__main__":
    main()
