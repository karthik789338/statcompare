"""Baseline B3 — Program-of-Thought (PoT) for comparison validity.

Chen et al. (TMLR 2023): instead of reasoning in natural language, the model
writes PYTHON that computes the answer; we execute it in a restricted namespace
and read back `result`. This tests whether FORCING explicit computation makes the
model pick the right denominator (re-base onto the question's population) -- i.e.
whether the flip failure is a computation problem (PoT fixes it) or a semantics
problem (PoT does not). Same wrong-ordering metric as src.evaluate / src.llm_judge,
so the columns line up.

The model is given the SAME information as llm_judge --mode meta: each place's
reported share, its universe, and the raw counts. Generated code runs with a
5s alarm and a whitelisted builtins namespace (no imports, no file/network).

Backends:
  --model Qwen/Qwen2.5-7B-Instruct   (transformers, runs on the L4)
  --mock {oracle,naive}              (offline, canned code; for logic tests)

Run:
  python -m src.pot_baseline --model Qwen/Qwen2.5-7B-Instruct --split dev
  python -m src.pot_baseline --selftest        # offline, no torch/GPU needed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal

from .llm_judge import _counts_for, summarize

SYSTEM = (
    "You convert a statistical comparison into PYTHON that computes the answer. "
    "You are given the question, each place's reported share and population "
    "universe, and the raw counts. A share must use the denominator the QUESTION "
    "asks for; if a place's reported universe differs, RE-BASE from the raw counts "
    "before comparing. Write a short Python snippet (no imports) that ends by "
    "assigning a variable named `result` to a dict "
    "{'comparable': bool, 'higher': <exact place name> or None}. "
    "Output ONLY one ```python code block, nothing else."
)


def build_pot_prompt(item: dict) -> list[dict]:
    cand_lines, counts_repr = [], {}
    for c in item["candidates"]:
        name = c["geo_name"]
        counts_repr[name] = _counts_for(item, c.get("geography", ""))
        cand_lines.append(
            f'- {name}: reported "{c["label"]}" = {c["value"]} '
            f'(on universe "{c["universe"]}")')
    user = (
        f'Question: {item["question"]}\n\n'
        f'Reported figures:\n' + "\n".join(cand_lines) + "\n\n"
        f'Raw counts per place, keyed by census variable:\n'
        f'counts = {json.dumps(counts_repr)}\n\n'
        f'Notes: B08301_003E is the numerator (drove alone). A key containing '
        f'"commuted" is the commuters denominator; B08301_001E (if present) is all '
        f'workers. Use `counts` to compute each place\'s share on the population the '
        f'question asks about, then set `result`.')
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _CODE_RE.search(text)
    if m:
        return m.group(1)
    # no fenced block: take from the first line that looks like code
    return text


class _Timeout:
    def __init__(self, secs): self.secs = secs
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(self.secs)
    def __exit__(self, *a): signal.alarm(0)
    def _raise(self, *a): raise TimeoutError()


_SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
         for k in ("abs", "round", "min", "max", "len", "float", "int", "sum",
                   "sorted", "dict", "list", "tuple", "range", "bool", "str",
                   "enumerate", "zip", "map", "filter")}


def run_code(code: str, places: list[str]):
    """Exec generated code in a restricted namespace -> (answered, higher)."""
    ns = {"__builtins__": _SAFE}
    try:
        with _Timeout(5):
            exec(compile(code, "<pot>", "exec"), ns)  # noqa: S102 (sandboxed)
        res = ns.get("result")
    except Exception:
        return False, None
    if not isinstance(res, dict):
        return False, None
    if res.get("comparable") is False:
        return False, None
    hi = res.get("higher")
    if isinstance(hi, str):
        for p in places:
            if p.lower() == hi.strip().lower() or p.lower() in hi.strip().lower():
                return True, p
    return False, None


# --------------------------------------------------------------------------- #
class MockPoT:
    """Canned code generator for logic tests."""
    def __init__(self, kind="oracle"):
        self.kind = kind

    def generate(self, item: dict) -> str:
        order = item["gold_ordering"] if self.kind == "oracle" else item["naive_ordering"]
        hi = "None" if order is None else repr(order[0])
        comp = "False" if order is None else "True"
        return f"```python\nresult = {{'comparable': {comp}, 'higher': {hi}}}\n```"


class HFPoT:
    """Qwen2.5-7B-Instruct (or any chat model) via transformers; loads on GPU."""
    def __init__(self, model_id: str, dtype: str = "bfloat16",
                 max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="auto")
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def generate(self, item: dict) -> str:
        msgs = build_pot_prompt(item)
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


def run(items: list[dict], model) -> list[dict]:
    rows = []
    for it in items:
        raw = model.generate(it)
        code = extract_code(raw)
        answered, higher = run_code(code, it["gold_frame"]["places"])
        gold = it["gold_ordering"]
        rows.append({
            "id": it["id"], "is_flip": it["is_flip"],
            "judge_answered": answered,
            "judge_correct": bool(answered and gold is not None and higher == gold[0]),
            "raw": raw,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench/compstat.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mock", choices=["oracle", "naive"], default=None)
    ap.add_argument("--split", default="dev")
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
    model = MockPoT(args.mock) if args.mock else HFPoT(args.model)
    rows = run(items, model)
    summary = summarize(rows)
    tag = (args.mock or args.model.split("/")[-1]) + "_pot"
    out = os.path.join("data", "bench", f"judge_{tag}_{args.split}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    label = args.mock or args.model
    print(f"[pot] {label} mode=pot split={args.split} n={len(rows)}")
    for scope in ("all", "flips"):
        s = summary[scope]
        print(f"  [{scope}] n={s['n']}  pot_wrong={s['judge_wrong_ordering_rate']}  "
              f"pot_answered={s['judge_answered_rate']}  (answered n={s['n_answered']})")
    print(f"[pot] wrote -> {out}")


def _selftest():
    base = dict(group="B08301", variable="B08301_003E", label="Drove alone",
                concept="MoT", universe="Workers 16 years and over",
                geo_level="state", year=2023, dataset="acs/acs1", unit="percent")
    def cand(geo, name, val):
        return {**base, "geography": geo, "geo_name": name, "value": val, "moe": None}
    flip = {
        "id": "it0", "question": "Among workers who commuted, Statex or Statey "
        "higher share drove alone?", "is_flip": True, "gold_verdict": "reconstruct",
        "gold_frame": {"measure": "drove alone", "denominator": "workers who commuted",
                       "period": 2023, "places": ["Statex", "Statey"], "direction": "greater"},
        "candidates": [cand("X", "Statex", 0.60), cand("Y", "Statey", 0.62)],
        "counts": {"X|B08301_003E": 60, "X|B08301__workers_who_commuted": 75,
                   "Y|B08301_003E": 62, "Y|B08301__workers_who_commuted": 97},
        "gold_ordering": ["Statex", "Statey"], "naive_ordering": ["Statey", "Statex"],
    }
    direct = {
        "id": "it1", "question": "Among all workers, Statex or Statey higher share "
        "drove alone?", "is_flip": False, "gold_verdict": "direct",
        "gold_frame": {"measure": "drove alone", "denominator": "all workers",
                       "period": 2023, "places": ["Statex", "Statey"], "direction": "greater"},
        "candidates": [cand("X", "Statex", 0.60), cand("Y", "Statey", 0.62)],
        "counts": {}, "gold_ordering": ["Statey", "Statex"],
        "naive_ordering": ["Statey", "Statex"],
    }
    items = [flip, direct]
    # code extraction + safe exec
    a, h = run_code("result = {'comparable': True, 'higher': 'Statex'}", ["Statex", "Statey"])
    assert a and h == "Statex"
    # a real re-basing snippet should also work and be sandboxed (no imports available)
    bad = run_code("import os\nresult = 1", ["Statex", "Statey"])
    assert bad == (False, None)   # import blocked -> no result
    assert "```python" in MockPoT().generate(flip)
    assert "counts =" in build_pot_prompt(flip)[1]["content"]
    so = summarize(run(items, MockPoT("oracle")))
    sn = summarize(run(items, MockPoT("naive")))
    assert so["all"]["judge_wrong_ordering_rate"] == 0.0, so
    assert sn["flips"]["judge_wrong_ordering_rate"] == 1.0, sn
    print("[selftest] OK — PoT prompt, code extraction, sandboxed exec, and "
          "wrong-ordering scoring verified")
    print(f"[selftest] oracle-pot all_wrong={so['all']['judge_wrong_ordering_rate']}; "
          f"naive-pot flip_wrong={sn['flips']['judge_wrong_ordering_rate']}")


if __name__ == "__main__":
    main()
