#!/usr/bin/env python3
"""Phase 12 - robustness: deterministic surface-form paraphrases of dev+test questions.

Keeps gold_frame / gold_ordering / candidates / counts unchanged (those come from the
data, not the wording); only rewrites `question`, so we can re-run the parser, the
symbolic method and the learned reranker on paraphrased surface forms and check they
are not brittle to template wording. Options (A or B) are preserved verbatim, so the
gold ordering (by place name) stays valid.

Output: data/bench/compstat_para.jsonl with split tags dev_para / test_para.
"""
import json, re, argparse

BENCH = "data/bench/compstat.jsonl"
OUT = "data/bench/compstat_para.jsonl"
SRC_SPLITS = {"dev", "test"}

_OPEN = re.compile(r"^(In which \w+) (.*): (.+?)\?$")


def _variants(q):
    """Up to 2 genuine surface-form paraphrases of q (same meaning, same options)."""
    out = []
    m = _OPEN.match(q)
    # p1: move the options to the front ("Of A or B, in which place ...?")
    if m:
        g1, g2, g3 = m.group(1), m.group(2), m.group(3)
        p1 = f"Of {g3}, {g1[0].lower() + g1[1:]} {g2}?"
        if p1 != q:
            out.append(p1)
    # p2: reword the comparison phrase + swap the trailing colon for an em dash
    p2 = q
    p2 = p2.replace("a larger share", "a greater proportion")
    p2 = p2.replace("were a larger", "were a greater")
    if m:
        g3 = m.group(3)
        p2 = p2.replace(f": {g3}?", f" — {g3}?")
    if p2 != q and p2 not in out:
        out.append(p2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        q = ("In which place did a larger share among all workers drove alone "
             "to get to work: Arizona or Colorado?")
        vs = _variants(q)
        assert len(vs) == 2, vs
        assert all(q != v for v in vs), vs
        assert vs[0].startswith("Of Arizona or Colorado,") and "Arizona or Colorado" in vs[0]
        assert "greater proportion" in vs[1] and " — Arizona or Colorado?" in vs[1]
        print("[paraphrase] selftest OK ->", vs)
        return

    items = [json.loads(l) for l in open(args.bench)]
    n_in = n_out = 0
    with open(args.out, "w") as f:
        for it in items:
            sp = it.get("split")
            if sp not in SRC_SPLITS:
                continue
            n_in += 1
            for k, pq in enumerate(_variants(it["question"]), 1):
                j = dict(it)
                j["question"] = pq
                gf = dict(it["gold_frame"])
                gf["question"] = pq
                j["gold_frame"] = gf
                j["id"] = f'{it["id"]}~p{k}'
                j["split"] = f"{sp}_para"
                j["orig_id"] = it["id"]
                f.write(json.dumps(j) + "\n")
                n_out += 1
    print(f"[paraphrase] {n_in} source items (dev+test) -> {n_out} paraphrased -> {args.out}")


if __name__ == "__main__":
    main()
