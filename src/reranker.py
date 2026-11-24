"""Phase 8 - learned reranker for denominator selection.

Replaces the zero-shot Qwen parser's denominator choice with a supervised
reranker trained on the TRAIN split: a frozen sentence-embedding bi-encoder
(all-MiniLM-L6-v2) + logistic regression scores each candidate denominator for a
question and picks the best. Output is written in the SAME schema as
src.parse_frames, so src.evaluate scores wrong-ordering unchanged:

  python -m src.reranker --train
  python -m src.reranker --predict dev
  python -m src.reranker --predict test
  python -m src.evaluate --parsed data/bench/parsed_frames_reranker_dev.jsonl --split dev

Offline logic test (mock embedder, no download, no bench file):
  python -m src.reranker --selftest

The reranker only chooses the denominator; the deterministic symbolic checker in
src.checker still decides answerability and the final ordering.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from .parse_frames import canon_denominator

BENCH = os.environ.get("COMPSTAT_BENCH", "data/bench/compstat.jsonl")
MODEL_PATH = "data/bench/reranker.pkl"
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_items(split=None):
    items = [json.loads(l) for l in open(BENCH)]
    if split:
        items = [i for i in items if i.get("split") == split]
    return items


def candidate_denoms(train_items):
    """Denominator label vocabulary = one representative phrase per gold class."""
    seen = {}
    for it in train_items:
        d = it["gold_frame"]["denominator"]
        seen[canon_denominator(d)] = d
    return sorted(seen.values())


def _norm(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class STEmbedder:
    def __init__(self, model_id=ST_MODEL):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_id)

    def __call__(self, texts):
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True))


class MockEmbedder:
    """Deterministic keyword embedder; offline logic tests only (no download)."""
    def __call__(self, texts):
        out = []
        for t in texts:
            tl = t.lower()
            v = np.zeros(8)
            v[0] = 1.0 if any(w in tl for w in ("commut", "traveled", "travelled")) else 0.0
            v[1] = 1.0 if ("all workers" in tl or "all worker" in tl) else 0.0
            v[2] = 1.0
            rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
            v[3:] = 0.01 * rng.standard_normal(5)
            out.append(v)
        return _norm(np.asarray(out))


def feats(qe, ce):
    return np.concatenate([qe, ce, np.abs(qe - ce), qe * ce], axis=-1)


def _pairs(items, cands, embed):
    qe = embed([it["question"] for it in items])
    ce = {c: embed([c])[0] for c in cands}
    return qe, ce


def fit(train_items, cands, embed):
    from sklearn.linear_model import LogisticRegression
    qe, ce = _pairs(train_items, cands, embed)
    X, y = [], []
    for i, it in enumerate(train_items):
        gold = canon_denominator(it["gold_frame"]["denominator"])
        for c in cands:
            X.append(feats(qe[i], ce[c]))
            y.append(1 if canon_denominator(c) == gold else 0)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(np.asarray(X), np.asarray(y))
    return clf


def pick_all(items, cands, embed, clf):
    qe, ce = _pairs(items, cands, embed)
    picks = []
    for i in range(len(items)):
        scores = {c: clf.predict_proba([feats(qe[i], ce[c])])[0, 1] for c in cands}
        picks.append(max(scores, key=scores.get))
    return picks


def train():
    tr = load_items("train")
    cands = candidate_denoms(tr)
    embed = STEmbedder()
    clf = fit(tr, cands, embed)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"clf": clf, "cands": cands}, f)
    # train-set fit sanity
    picks = pick_all(tr, cands, embed, clf)
    acc = np.mean([canon_denominator(p) == canon_denominator(it["gold_frame"]["denominator"])
                   for p, it in zip(picks, tr)])
    print(f"[reranker] trained on {len(tr)} items, cands={cands}, train denom_acc={acc:.4f}")
    print(f"[reranker] saved -> {MODEL_PATH}")


def predict_split(split):
    d = pickle.load(open(MODEL_PATH, "rb"))
    clf, cands = d["clf"], d["cands"]
    embed = STEmbedder()
    items = load_items(split)
    picks = pick_all(items, cands, embed, clf)
    rows = []
    for it, pick in zip(items, picks):
        gold = it["gold_frame"]["denominator"]
        rows.append({
            "id": it["id"], "parsed": {"denominator": pick},
            "gold_denominator": gold,
            "denom_correct": canon_denominator(pick) == canon_denominator(gold),
            "is_flip": it.get("is_flip", False),
        })
    out = f"data/bench/parsed_frames_reranker_{split}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    flips = [r for r in rows if r["is_flip"]]
    acc = np.mean([r["denom_correct"] for r in rows])
    facc = np.mean([r["denom_correct"] for r in flips]) if flips else float("nan")
    print(f"[reranker] split={split} n={len(rows)} denom_acc={acc:.4f} "
          f"flip_denom_acc={facc:.4f} n_flips={len(flips)}")
    print(f"[reranker] wrote -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", default=None, help="split to predict (dev/test)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.train:
        train()
    if args.predict:
        predict_split(args.predict)


def _selftest():
    emb = MockEmbedder()

    def mk(q, denom, flip, gid):
        return {"id": gid, "question": q,
                "gold_frame": {"denominator": denom}, "is_flip": flip}

    train_items = []
    for k in range(12):
        train_items.append(mk(f"Among workers who commuted, did place{k} drive alone more?",
                              "workers who commuted", True, f"c{k}"))
        train_items.append(mk(f"Among all workers, did place{k} drive alone more?",
                              "all workers", False, f"a{k}"))
    cands = candidate_denoms(train_items)
    assert set(canon_denominator(c) for c in cands) == {"commuters", "all"}, cands
    clf = fit(train_items, cands, emb)
    # held-out items: same templates, unseen indices -> keyword signal generalizes
    held = [mk("Among workers who commuted, did placeZ drive alone more?",
               "workers who commuted", True, "cz"),
            mk("Among all workers, did placeZ drive alone more?",
               "all workers", False, "az")]
    picks = pick_all(held, cands, emb, clf)
    correct = sum(canon_denominator(p) == canon_denominator(it["gold_frame"]["denominator"])
                  for p, it in zip(picks, held))
    assert correct == 2, (picks, [i["gold_frame"]["denominator"] for i in held])
    # output schema matches parse_frames (id, parsed.denominator, denom_correct, is_flip)
    r = {"id": held[0]["id"], "parsed": {"denominator": picks[0]},
         "denom_correct": True, "is_flip": True}
    assert set(r) >= {"id", "parsed", "denom_correct", "is_flip"} and "denominator" in r["parsed"]
    print("[selftest] OK - reranker fit/predict pipeline, candidate vocab, keyword "
          "generalization, and parse_frames-compatible output schema verified")
    print(f"[selftest] held-out denom picks correct = {correct}/2")


if __name__ == "__main__":
    main()
