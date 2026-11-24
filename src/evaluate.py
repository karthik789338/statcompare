"""Phase 5/6 — end-to-end evaluation: wrong-ordering rate vs baselines.

The headline integrity metric is the WRONG-ORDERING RATE: how often a system
tells the user the wrong place ranks higher. We compare:

  - naive   : a denominator-blind reranker that compares the retrieved
              (all-workers) shares as-is. It always answers. Encoded in the
              benchmark as ``naive_ordering``.
  - method  : our pipeline = LLM-parsed denominator -> symbolic checker. On a
              commuters question it must re-base onto the requested denominator
              (the RECONSTRUCT path); if the parser mis-reads the denominator it
              falls back to the all-workers share and errs on flip pairs.
  - oracle  : gold frame -> checker. A sanity upper bound (should be ~0 wrong).

Wrong-ordering rate is computed over ANSWERED items (a system that abstains is
incomplete, not wrong); we report the answered rate separately.

Run:
    python -m src.evaluate --bench data/bench/compstat.jsonl \
        --parsed data/bench/parsed_frames_dev.jsonl --split dev
    python -m src.evaluate --selftest        # offline, fabricates data
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from .frames import Estimate, ComparisonFrame
from .checker import check


def _counts_to_tuples(counts: dict) -> dict:
    return {(k.split("|")[0], k.split("|")[1]): v for k, v in counts.items()}


def _ordering_for(item: dict, denominator: str):
    """Run the checker on an item using the given requested denominator."""
    a = Estimate.from_dict(item["candidates"][0])
    b = Estimate.from_dict(item["candidates"][1])
    gf = item["gold_frame"]
    frame = ComparisonFrame(
        question=item.get("question", ""), measure=gf["measure"],
        population=denominator, denominator=denominator, unit="percent",
        period=gf.get("period"), geo_level=a.geo_level,
        places=gf["places"], direction=gf.get("direction", "greater"))
    return check(frame, a, b, counts=_counts_to_tuples(item.get("counts", {})))


def evaluate(items: list[dict], parsed_by_id: dict) -> list[dict]:
    rows = []
    for it in items:
        gold = it["gold_ordering"]
        rec = {"id": it["id"], "is_flip": it["is_flip"],
               "gold_verdict": it["gold_verdict"]}
        # naive baseline: always answers, compares all-workers shares
        rec["naive_answered"] = it["naive_ordering"] is not None
        rec["naive_correct"] = (it["naive_ordering"] == gold)
        # our method: parsed denominator -> checker
        p = parsed_by_id.get(it["id"])
        denom = (p or {}).get("parsed", {}).get("denominator") if p else None
        if denom:
            res = _ordering_for(it, denom)
            rec["method_answered"] = res.ordering is not None
            rec["method_correct"] = (res.ordering == gold)
            rec["method_verdict"] = res.verdict.value
        else:
            rec["method_answered"] = False
            rec["method_correct"] = False
            rec["method_verdict"] = None
        # oracle: gold denominator -> checker (sanity upper bound)
        res_o = _ordering_for(it, it["gold_frame"]["denominator"])
        rec["oracle_answered"] = res_o.ordering is not None
        rec["oracle_correct"] = (res_o.ordering == gold)
        rows.append(rec)
    return rows


def _wrong_rate(rows, system: str, subset=lambda r: True):
    """Wrong-ordering rate among ANSWERED items of this subset."""
    answered = [r for r in rows if subset(r) and r[f"{system}_answered"]]
    if not answered:
        return None, 0
    wrong = sum(1 for r in answered if not r[f"{system}_correct"])
    return round(wrong / len(answered), 4), len(answered)


def _answered_rate(rows, system: str, subset=lambda r: True):
    s = [r for r in rows if subset(r)]
    if not s:
        return None
    return round(sum(1 for r in s if r[f"{system}_answered"]) / len(s), 4)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name, subset in [("all", lambda r: True), ("flips", lambda r: r["is_flip"])]:
        n = len([r for r in rows if subset(r)])
        nw, nans = _wrong_rate(rows, "naive", subset)
        mw, mans = _wrong_rate(rows, "method", subset)
        ow, _ = _wrong_rate(rows, "oracle", subset)
        out[name] = {
            "n": n,
            "naive_wrong_ordering_rate": nw,
            "method_wrong_ordering_rate": mw,
            "oracle_wrong_ordering_rate": ow,
            "method_answered_rate": _answered_rate(rows, "method", subset),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench/compstat.jsonl")
    ap.add_argument("--parsed", default=None,
                    help="parsed_frames_<split>.jsonl from parse_frames")
    ap.add_argument("--split", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    items = [json.loads(l) for l in open(args.bench)]
    if args.split:
        items = [i for i in items if i.get("split") == args.split]
    parsed_by_id = {}
    if args.parsed:
        parsed_by_id = {p["id"]: p for p in (json.loads(l) for l in open(args.parsed))}
        items = [i for i in items if i["id"] in parsed_by_id]  # eval only parsed items
    rows = evaluate(items, parsed_by_id)
    summary = summarize(rows)
    out = os.path.join("data", "bench", f"eval_{args.split or 'all'}.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"[evaluate] split={args.split} n={len(rows)}")
    for scope in ("all", "flips"):
        s = summary[scope]
        print(f"  [{scope}] n={s['n']}  naive_wrong={s['naive_wrong_ordering_rate']}  "
              f"method_wrong={s['method_wrong_ordering_rate']}  "
              f"oracle_wrong={s['oracle_wrong_ordering_rate']}  "
              f"method_answered={s['method_answered_rate']}")
    print(f"[evaluate] wrote -> {out}")


# --------------------------------------------------------------------------- #
def _selftest():
    # one flip item (commuters question) + one direct item, with parsed frames
    base_cand = dict(group="B08301", variable="B08301_003E", label="Drove alone",
                     concept="MoT", universe="Workers 16 years and over",
                     geo_level="state", year=2023, dataset="acs/acs1", unit="percent")
    def cand(geo, name, val, base=None):
        return {**base_cand, "geography": geo, "geo_name": name, "value": val,
                "base_variable": base, "moe": None}
    # flip item: all-workers shares X=0.60,Y=0.62 (naive => Y first);
    # commuter shares X=0.80,Y=0.639 (gold => X first)
    flip = {
        "id": "it0", "question": "q", "is_flip": True, "gold_verdict": "reconstruct",
        "gold_frame": {"measure": "drove alone", "denominator": "workers who commuted",
                       "period": 2023, "places": ["Statex", "Statey"], "direction": "greater"},
        "candidates": [cand("X", "Statex", 0.60, "B08301__workers_who_commuted"),
                       cand("Y", "Statey", 0.62, "B08301__workers_who_commuted")],
        "counts": {"X|B08301_003E": 60, "X|B08301__workers_who_commuted": 75,
                   "Y|B08301_003E": 62, "Y|B08301__workers_who_commuted": 97},
        "gold_ordering": ["Statex", "Statey"], "naive_ordering": ["Statey", "Statex"],
    }
    direct = {
        "id": "it1", "question": "q", "is_flip": False, "gold_verdict": "direct",
        "gold_frame": {"measure": "drove alone", "denominator": "all workers",
                       "period": 2023, "places": ["Statex", "Statey"], "direction": "greater"},
        "candidates": [cand("X", "Statex", 0.60), cand("Y", "Statey", 0.62)],
        "counts": {},
        "gold_ordering": ["Statey", "Statex"], "naive_ordering": ["Statey", "Statex"],
    }
    items = [flip, direct]
    # parser that gets BOTH right
    good = {"it0": {"id": "it0", "parsed": {"denominator": "workers who commuted"}},
            "it1": {"id": "it1", "parsed": {"denominator": "all workers"}}}
    # parser that is denominator-blind (always "all workers") -> wrong on the flip
    blind = {"it0": {"id": "it0", "parsed": {"denominator": "all workers"}},
             "it1": {"id": "it1", "parsed": {"denominator": "all workers"}}}

    sg = summarize(evaluate(items, good))
    sb = summarize(evaluate(items, blind))
    # naive is wrong on the flip (1 of 1 flips), right on direct
    assert sg["flips"]["naive_wrong_ordering_rate"] == 1.0, sg
    # good parser: 0 wrong on flips; blind parser: 1.0 wrong on flips
    assert sg["flips"]["method_wrong_ordering_rate"] == 0.0, sg
    assert sb["flips"]["method_wrong_ordering_rate"] == 1.0, sb
    # oracle always right
    assert sg["all"]["oracle_wrong_ordering_rate"] == 0.0, sg
    # method answers everything here
    assert sg["all"]["method_answered_rate"] == 1.0, sg
    print("[selftest] OK — naive/method/oracle wrong-ordering rates computed correctly")
    print(f"[selftest] good-parser flips: naive_wrong="
          f"{sg['flips']['naive_wrong_ordering_rate']} method_wrong="
          f"{sg['flips']['method_wrong_ordering_rate']}; "
          f"blind-parser method_wrong={sb['flips']['method_wrong_ordering_rate']}")


if __name__ == "__main__":
    main()
