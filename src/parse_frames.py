"""Phase 4 — LLM frame parser.

Given a CompStat-Bench question plus the *table definitions* of the candidate
cells (variable label, population universe, concept), an instruct LLM extracts
the comparison frame: measure, requested population/denominator, unit, period,
places, direction. The denominator field is the one that matters — getting
"workers who commuted" vs "all workers" right is what lets the downstream
checker pick the valid comparison.

Backends:
  --model Qwen/Qwen2.5-7B-Instruct   (transformers, runs on the L4)
  --mock                             (offline, canned output; for logic tests)

Run:
  python -m src.parse_frames --model Qwen/Qwen2.5-7B-Instruct --split dev
  python -m src.parse_frames --selftest        # offline, no torch/GPU needed
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

SYSTEM = (
    "You convert a statistical comparison question into a strict JSON frame. "
    "You are given the question and the definitions of the data cells that "
    "could answer it, including each cell's population UNIVERSE. Identify the "
    "population/denominator the question actually asks about, which may differ "
    "from a cell's universe. Output ONLY JSON, no prose."
)

USER_TMPL = """Question: {question}

Candidate data cells (definitions):
{defs}

Return JSON with exactly these keys:
{{"measure": str,           // what is compared, e.g. "share who drove alone"
  "population": str,        // the population the question restricts to
  "denominator": str,       // the denominator the share should use
  "unit": "percent"|"count"|"dollars",
  "period": int|null,       // year, if stated
  "places": [str, ...],     // places being compared
  "direction": "greater"|"less"}}"""


def build_prompt(item: dict) -> list[dict]:
    seen, defs = set(), []
    for c in item["candidates"]:
        key = (c["variable"], c["universe"])
        if key in seen:
            continue
        seen.add(key)
        defs.append(f'- {c["variable"]}: "{c["label"]}" '
                    f'(concept: {c["concept"]}; universe: {c["universe"]})')
    user = USER_TMPL.format(question=item["question"], defs="\n".join(defs))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced {...} object out of a model response."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def canon_denominator(s: str) -> str:
    """Map a denominator phrase to a canonical class for scoring."""
    s = (s or "").lower()
    if any(w in s for w in ("commut", "traveled", "travelled", "excl", "not at home")):
        return "commuters"
    if "home" in s:   # "excluding those who worked from home" etc.
        return "commuters"
    if "all" in s or "total" in s or "16 years" in s or "worker" in s:
        return "all"
    return "other"


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class MockModel:
    """Returns a correct-ish frame derived from the gold frame; logic-test only."""
    def __init__(self, oracle: bool = True):
        self.oracle = oracle

    def generate(self, item: dict) -> str:
        g = item["gold_frame"]
        denom = g["denominator"] if self.oracle else "all workers"
        return json.dumps({
            "measure": g["measure"], "population": denom, "denominator": denom,
            "unit": "percent", "period": g.get("period"),
            "places": g["places"], "direction": "greater"})


class HFModel:
    """Qwen2.5-7B-Instruct (or any chat model) via transformers; loads on GPU."""
    def __init__(self, model_id: str, dtype: str = "bfloat16",
                 max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map={"":0})
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def generate(self, item: dict) -> str:
        msgs = build_prompt(item)
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# run + score
# --------------------------------------------------------------------------- #
def run(items: list[dict], model) -> list[dict]:
    out = []
    for it in items:
        raw = model.generate(it)
        frame = extract_json(raw) or {}
        out.append({
            "id": it["id"],
            "parsed": frame,
            "raw": raw,
            "gold_denominator": it["gold_frame"]["denominator"],
            "denom_correct": canon_denominator(frame.get("denominator", ""))
                             == canon_denominator(it["gold_frame"]["denominator"]),
            "is_flip": it.get("is_flip", False),
        })
    return out


def score(parsed: list[dict]) -> dict:
    n = len(parsed)
    acc = sum(p["denom_correct"] for p in parsed) / max(1, n)
    flips = [p for p in parsed if p["is_flip"]]
    flip_acc = (sum(p["denom_correct"] for p in flips) / len(flips)) if flips else None
    return {"n": n, "denominator_accuracy": round(acc, 4),
            "flip_denominator_accuracy": round(flip_acc, 4) if flip_acc is not None else None,
            "n_flips": len(flips)}


def _load_items(path: str, split: Optional[str], limit: Optional[int]) -> list[dict]:
    items = [json.loads(l) for l in open(path)]
    if split:
        items = [i for i in items if i.get("split") == split]
    if limit:
        items = items[:limit]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench/compstat.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    items = _load_items(args.bench, args.split, args.limit)
    model = MockModel() if args.mock else HFModel(args.model)
    parsed = run(items, model)
    out = os.path.join("data", "bench", f"parsed_frames_{args.split}.jsonl")
    with open(out, "w") as f:
        for p in parsed:
            f.write(json.dumps(p) + "\n")
    print(f"[parse_frames] {args.model if not args.mock else 'mock'} "
          f"on split={args.split}: {score(parsed)}")
    print(f"[parse_frames] wrote -> {out}")


def _selftest():
    # extract_json robustness
    assert extract_json('noise {"a": 1, "b": [2,3]} tail')["b"] == [2, 3]
    assert extract_json("no json here") is None
    # canonicalization
    assert canon_denominator("workers who traveled to work") == "commuters"
    assert canon_denominator("all workers") == "all"
    # end-to-end with the mock oracle on a tiny synthetic item
    item = {
        "id": "t0",
        "question": "In which place did a larger share among workers who traveled "
                    "to work drove alone: Statex or Statey?",
        "gold_frame": {"measure": "drove alone", "population": "workers who commuted",
                       "denominator": "workers who commuted", "unit": "percent",
                       "period": 2023, "places": ["Statex", "Statey"],
                       "direction": "greater"},
        "candidates": [{"variable": "B08301_003E", "label": "Drove alone",
                        "concept": "Means of Transportation to Work",
                        "universe": "Workers 16 years and over"}],
        "is_flip": True,
    }
    good = run([item], MockModel(oracle=True))
    bad = run([item], MockModel(oracle=False))  # always says "all workers"
    assert good[0]["denom_correct"] is True
    assert bad[0]["denom_correct"] is False
    assert score(good)["denominator_accuracy"] == 1.0
    assert score(bad)["flip_denominator_accuracy"] == 0.0
    # prompt actually surfaces the universe to the model
    assert "universe:" in build_prompt(item)[1]["content"]
    print("[selftest] OK — json extraction, canonicalization, prompt, and "
          "oracle/non-oracle scoring verified")


if __name__ == "__main__":
    main()
