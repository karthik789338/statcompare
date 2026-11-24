"""Baseline B1/B2/B4 — LLM-as-judge for comparison validity.

The FIRM-relevant baseline: instead of our parse->symbolic-checker pipeline, hand
the comparison directly to an instruct LLM acting as a judge and ask which place
ranks higher (or whether the two numbers are comparable at all). We score it with
the SAME wrong-ordering metric as src.evaluate, so the columns line up.

Modes (prompt variants):
  direct : question + the two reported (all-workers) shares only.          [B1]
  cot    : same inputs, but the model may reason step by step first.       [B2]
  meta   : also shows each place's population UNIVERSE and the raw counts
           (numerator + re-based denominator) -- the fair competitor to our
           symbolic re-basing, testing whether an LLM can do the comparability
           reasoning + arithmetic in-context.                              [B4]

Backends:
  --model Qwen/Qwen2.5-7B-Instruct   (transformers, runs on the L4)
  --mock {oracle,naive}              (offline, canned; for logic tests)

Run:
  python -m src.llm_judge --model Qwen/Qwen2.5-7B-Instruct --split dev --mode direct
  python -m src.llm_judge --selftest        # offline, no torch/GPU needed

Peer-reviewed anchors: LLM-as-judge (Zheng et al., NeurIPS 2023); G-Eval
(Liu et al., EMNLP 2023). Selective-prediction reporting (answered rate /
risk-coverage): Kamath et al. (ACL 2020); Varshney et al. (Findings of ACL 2022).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from .parse_frames import extract_json

SYSTEM = (
    "You are a careful statistics referee. You are given a comparison question "
    "and reported figures for two places. Decide whether the two figures can be "
    "VALIDLY compared to answer the question, and if so which place ranks higher. "
    "Two shares are only comparable when they use the same denominator / "
    "population base. If they do not, the comparison is NOT valid as reported. "
    "Output ONLY a JSON object, no prose."
)

OUT_SPEC = (
    'Return JSON with exactly these keys:\n'
    '{{"comparable": true|false,   // can these figures be validly compared for the question?\n'
    '  "higher": str|null}}        // exact place name that ranks higher, or null if not comparable'
)


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(v)


def _counts_for(item: dict, geo: str) -> dict:
    """All counts whose key is '<geo>|<var>' -> {var: value}."""
    out = {}
    for k, v in item.get("counts", {}).items():
        g, _, var = k.partition("|")
        if g == geo:
            out[var] = v
    return out


def build_judge_prompt(item: dict, mode: str) -> list[dict]:
    q = item["question"]
    cands = item["candidates"]
    lines = []
    for c in cands:
        line = f'- {c["geo_name"]}: {c["label"]} = {_pct(c["value"])}'
        if mode == "meta":
            line += f'  (reported on universe: "{c["universe"]}")'
            cnts = _counts_for(item, c.get("geography", ""))
            if cnts:
                inner = "; ".join(f"{k}={v}" for k, v in sorted(cnts.items()))
                line += f"\n    raw counts for this place: {inner}"
        lines.append(line)
    body = [f"Question: {q}", "", "Reported figures:", *lines, ""]
    if mode == "meta":
        body.append(
            "The reported shares may use a different denominator than the question "
            "asks for. Use the raw counts to re-base onto the question's population "
            "if needed before comparing."
        )
    elif mode == "cot":
        body.append("Think step by step about whether the two figures are comparable, "
                    "then give your answer.")
    body.append(OUT_SPEC.format())
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(body)}]


def judge_to_ordering(raw: str, places: list[str]):
    """Parse model output -> (answered: bool, higher: str|None)."""
    obj = extract_json(raw) or {}
    comparable = obj.get("comparable")
    higher = obj.get("higher")
    if comparable is False:
        return False, None
    if isinstance(higher, str):
        # tolerant match against the two place names
        for p in places:
            if p.lower() == higher.strip().lower() or p.lower() in higher.strip().lower():
                return True, p
    return False, None


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class MockJudge:
    """Canned judge for logic tests: 'oracle' answers gold, 'naive' answers naive."""
    def __init__(self, kind: str = "oracle"):
        self.kind = kind

    def chat(self, item: dict, mode: str) -> str:
        order = item["gold_ordering"] if self.kind == "oracle" else item["naive_ordering"]
        if order is None:
            return json.dumps({"comparable": False, "higher": None})
        return json.dumps({"comparable": True, "higher": order[0]})


class HFJudge:
    """Qwen2.5-7B-Instruct (or any chat model) via transformers; loads on GPU."""
    def __init__(self, model_id: str, dtype: str = "bfloat16",
                 max_new_tokens: int = 384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="auto")
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def chat(self, item: dict, mode: str) -> str:
        msgs = build_judge_prompt(item, mode)
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# run + score  (mirrors src.evaluate wrong-ordering convention)
# --------------------------------------------------------------------------- #
def run(items: list[dict], model, mode: str) -> list[dict]:
    rows = []
    for it in items:
        raw = model.chat(it, mode)
        answered, higher = judge_to_ordering(raw, it["gold_frame"]["places"])
        gold = it["gold_ordering"]
        rows.append({
            "id": it["id"], "is_flip": it["is_flip"],
            "judge_answered": answered,
            "judge_correct": bool(answered and gold is not None and higher == gold[0]),
            "raw": raw,
        })
    return rows


def _wrong_rate(rows, subset):
    answered = [r for r in rows if subset(r) and r["judge_answered"]]
    if not answered:
        return None, 0
    wrong = sum(1 for r in answered if not r["judge_correct"])
    return round(wrong / len(answered), 4), len(answered)


def _answered_rate(rows, subset):
    s = [r for r in rows if subset(r)]
    return round(sum(1 for r in s if r["judge_answered"]) / len(s), 4) if s else None


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name, subset in [("all", lambda r: True), ("flips", lambda r: r["is_flip"])]:
        w, nans = _wrong_rate(rows, subset)
        out[name] = {
            "n": len([r for r in rows if subset(r)]),
            "judge_wrong_ordering_rate": w,
            "judge_answered_rate": _answered_rate(rows, subset),
            "n_answered": nans,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench/compstat.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mock", choices=["oracle", "naive"], default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", choices=["direct", "cot", "meta"], default="direct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    items = [json.loads(l) for l in open(args.bench)]
    if args.split:
        items = [i for i in items if i.get("split") == args.split]
    if args.limit:
        items = items[:args.limit]
    model = MockJudge(args.mock) if args.mock else HFJudge(args.model)
    rows = run(items, model, args.mode)
    summary = summarize(rows)
    tag = (args.mock or args.model.split("/")[-1]) + f"_{args.mode}"
    out = os.path.join("data", "bench", f"judge_{tag}_{args.split}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    label = args.mock or args.model
    print(f"[llm_judge] {label} mode={args.mode} split={args.split} n={len(rows)}")
    for scope in ("all", "flips"):
        s = summary[scope]
        print(f"  [{scope}] n={s['n']}  judge_wrong={s['judge_wrong_ordering_rate']}  "
              f"judge_answered={s['judge_answered_rate']}  (answered n={s['n_answered']})")
    print(f"[llm_judge] wrote -> {out}")


# --------------------------------------------------------------------------- #
def _selftest():
    # reuse the same two-item fixture shape as src.evaluate
    base = dict(group="B08301", variable="B08301_003E", label="Drove alone",
                concept="MoT", universe="Workers 16 years and over",
                geo_level="state", year=2023, dataset="acs/acs1", unit="percent")
    def cand(geo, name, val):
        return {**base, "geography": geo, "geo_name": name, "value": val, "moe": None}
    flip = {
        "id": "it0", "question": "Among workers who commuted, did Statex or Statey "
        "have a higher share who drove alone?", "is_flip": True,
        "gold_verdict": "reconstruct",
        "gold_frame": {"measure": "drove alone", "denominator": "workers who commuted",
                       "period": 2023, "places": ["Statex", "Statey"], "direction": "greater"},
        "candidates": [cand("X", "Statex", 0.60), cand("Y", "Statey", 0.62)],
        "counts": {"X|B08301_003E": 60, "X|B08301__workers_who_commuted": 75,
                   "Y|B08301_003E": 62, "Y|B08301__workers_who_commuted": 97},
        "gold_ordering": ["Statex", "Statey"], "naive_ordering": ["Statey", "Statex"],
    }
    direct = {
        "id": "it1", "question": "Among all workers, did Statex or Statey have a "
        "higher share who drove alone?", "is_flip": False, "gold_verdict": "direct",
        "gold_frame": {"measure": "drove alone", "denominator": "all workers",
                       "period": 2023, "places": ["Statex", "Statey"], "direction": "greater"},
        "candidates": [cand("X", "Statex", 0.60), cand("Y", "Statey", 0.62)],
        "counts": {}, "gold_ordering": ["Statey", "Statex"],
        "naive_ordering": ["Statey", "Statex"],
    }
    items = [flip, direct]
    # prompt wiring
    assert "universe:" in build_judge_prompt(flip, "meta")[1]["content"]
    assert "raw counts" in build_judge_prompt(flip, "meta")[1]["content"]
    assert "raw counts" not in build_judge_prompt(flip, "direct")[1]["content"]
    # parsing
    a, h = judge_to_ordering('{"comparable": true, "higher": "Statex"}', ["Statex", "Statey"])
    assert a and h == "Statex"
    a, h = judge_to_ordering('{"comparable": false, "higher": null}', ["Statex", "Statey"])
    assert (not a) and h is None
    # scoring: oracle judge is always right; naive judge is wrong on the flip
    so = summarize(run(items, MockJudge("oracle"), "direct"))
    sn = summarize(run(items, MockJudge("naive"), "direct"))
    assert so["all"]["judge_wrong_ordering_rate"] == 0.0, so
    assert sn["flips"]["judge_wrong_ordering_rate"] == 1.0, sn
    assert so["all"]["judge_answered_rate"] == 1.0, so
    print("[selftest] OK — judge prompt modes, JSON parsing, and wrong-ordering "
          "scoring verified")
    print(f"[selftest] oracle-judge all_wrong={so['all']['judge_wrong_ordering_rate']}; "
          f"naive-judge flip_wrong={sn['flips']['judge_wrong_ordering_rate']}")


if __name__ == "__main__":
    main()
