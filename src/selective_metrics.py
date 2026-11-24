"""Phase 11 — selective-prediction + ranking metrics (no GPU).

Reads the saved per-item eval/judge/pot outputs and computes, per split:
  (1) a risk-coverage operating-point table per method: coverage and
      selective-risk (wrong-ordering among answered), on all and on flips;
  (2) abstention targeting for the method: among the items the method
      ABSTAINED on, the error if force-answered with the naive (all-
      denominator) ordering, vs the method's error on answered items;
  (3) Kendall-tau for k-way ranking: pairwise orderings aggregated into
      per-place win-counts (gold vs method) per (family, denominator).

A full AURC curve needs a continuous confidence score; the deterministic
checker emits a hard abstain/answer decision, so we report risk-coverage
operating points + the Pareto frontier rather than a swept AURC.
"""
from __future__ import annotations
import os, json, glob, itertools

BENCH = {
    "dev": "data/bench/compstat.jsonl",
    "test": "data/bench/compstat.jsonl",
    "test_ood": "data/bench/compstat.jsonl",
    "worldbank": "data/bench/compstat_worldbank.jsonl",
    "eurostat": "data/bench/compstat_eurostat.jsonl",
}
SPLITS = ["dev", "test", "test_ood", "worldbank", "eurostat"]
MODES = ["direct", "cot", "meta", "pot"]


def _cov_risk(flags, subset=None):
    """flags: list of (answered, correct). -> (coverage, selective_risk, n)."""
    rows = flags if subset is None else [f for f, s in zip(flags, subset) if s]
    n = len(rows)
    ans = [c for a, c in rows if a]
    cov = len(ans) / n if n else None
    risk = (sum(1 for c in ans if not c) / len(ans)) if ans else None
    return cov, risk, n


def _load_eval(split):
    p = f"data/bench/eval_{split}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["rows"]


def _load_judge(split, mode):
    g = glob.glob(f"data/bench/judge_*_{mode}_{split}.jsonl")
    if not g:
        return None
    return [json.loads(l) for l in open(g[0])]


def _kendall(a, b):
    """Kendall-tau-b between two equal-length score vectors (no scipy)."""
    n = len(a)
    if n < 2:
        return None
    c = d = 0
    for i, j in itertools.combinations(range(n), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0:
            c += 1
        elif s < 0:
            d += 1
    tot = c + d
    return (c - d) / tot if tot else None


def operating_points(split):
    rows = _load_eval(split)
    if rows is None:
        return None
    flips = [r["is_flip"] for r in rows]
    out = {"n": len(rows), "n_flips": sum(flips), "methods": {}}
    # method / naive / oracle from the eval rows
    for name, (af, cf) in {
        "method": ("method_answered", "method_correct"),
        "naive": ("naive_answered", "naive_correct"),
        "oracle": ("oracle_answered", "oracle_correct"),
    }.items():
        flags = [(r[af], r[cf]) for r in rows]
        cov_a, risk_a, _ = _cov_risk(flags)
        cov_f, risk_f, _ = _cov_risk(flags, flips)
        out["methods"][name] = {"all": {"coverage": cov_a, "risk": risk_a},
                                 "flips": {"coverage": cov_f, "risk": risk_f}}
    # judges + pot
    for mode in MODES:
        jr = _load_judge(split, mode)
        if jr is None:
            continue
        flips_j = [r["is_flip"] for r in jr]
        flags = [(r["judge_answered"], r["judge_correct"]) for r in jr]
        cov_a, risk_a, _ = _cov_risk(flags)
        cov_f, risk_f, _ = _cov_risk(flags, flips_j)
        label = "pot" if mode == "pot" else f"judge_{mode}"
        out["methods"][label] = {"all": {"coverage": cov_a, "risk": risk_a},
                                  "flips": {"coverage": cov_f, "risk": risk_f}}
    return out


def abstention_targeting(split):
    rows = _load_eval(split)
    if rows is None:
        return None
    answered = [r for r in rows if r["method_answered"]]
    abstained = [r for r in rows if not r["method_answered"]]
    err_ans = (sum(1 for r in answered if not r["method_correct"]) / len(answered)
               if answered else None)
    # force-answer the abstained items with the naive (all-denominator) ordering
    forced_err = (sum(1 for r in abstained if not r["naive_correct"]) / len(abstained)
                  if abstained else None)
    return {"n_answered": len(answered), "method_error_on_answered": err_ans,
            "n_abstained": len(abstained),
            "forced_naive_error_on_abstained": forced_err}


def kendall_ranking(split):
    rows = _load_eval(split)
    if rows is None or not os.path.exists(BENCH[split]):
        return None
    bench = {}
    for l in open(BENCH[split]):
        it = json.loads(l)
        bench[it["id"]] = it
    # group by (family, denominator)
    groups = {}
    for r in rows:
        it = bench.get(r["id"])
        if not it:
            continue
        key = (it["family"], it["gold_frame"]["denominator"])
        groups.setdefault(key, []).append((r, it))
    taus, used, total = [], 0, 0
    for key, recs in groups.items():
        gold_win, meth_win, places = {}, {}, set()
        for r, it in recs:
            order = it["gold_ordering"]
            if not order or len(order) != 2:
                continue
            hi, lo = order[0], order[1]
            places.add(hi); places.add(lo)
            gold_win[hi] = gold_win.get(hi, 0) + 1
            gold_win.setdefault(lo, 0)
            total += 1
            if r["method_answered"]:
                used += 1
                mhi = hi if r["method_correct"] else lo
                mlo = lo if r["method_correct"] else hi
                meth_win[mhi] = meth_win.get(mhi, 0) + 1
                meth_win.setdefault(mlo, 0)
        pl = sorted(places)
        if len(pl) < 3:
            continue
        g = [gold_win.get(p, 0) for p in pl]
        m = [meth_win.get(p, 0) for p in pl]
        t = _kendall(g, m)
        if t is not None:
            taus.append(t)
    return {"groups_scored": len(taus),
            "mean_kendall_tau": (sum(taus) / len(taus) if taus else None),
            "pair_coverage": (used / total if total else None)}


def main():
    report = {}
    for split in SPLITS:
        op = operating_points(split)
        if op is None:
            continue
        report[split] = {"operating_points": op,
                         "abstention_targeting": abstention_targeting(split),
                         "kendall_ranking": kendall_ranking(split)}
    out = "data/bench/selective_metrics.json"
    json.dump(report, open(out, "w"), indent=2)
    # human-readable summary
    for split, d in report.items():
        print(f"\n==== {split} (n={d['operating_points']['n']}, "
              f"flips={d['operating_points']['n_flips']}) ====")
        print(f"{'method':<14}{'cov(all)':>9}{'risk(all)':>10}"
              f"{'cov(flip)':>10}{'risk(flip)':>11}")
        for name, mv in d["operating_points"]["methods"].items():
            a, f = mv["all"], mv["flips"]
            def fmt(x):
                return "  n/a" if x is None else f"{100*x:5.1f}"
            print(f"{name:<14}{fmt(a['coverage']):>9}{fmt(a['risk']):>10}"
                  f"{fmt(f['coverage']):>10}{fmt(f['risk']):>11}")
        at = d["abstention_targeting"]
        print(f"  abstention: method err on answered="
              f"{at['method_error_on_answered']} (n={at['n_answered']}); "
              f"forced-naive err on abstained="
              f"{at['forced_naive_error_on_abstained']} (n={at['n_abstained']})")
        kr = d["kendall_ranking"]
        if kr:
            print(f"  kendall-tau (method vs gold place-ranking): "
                  f"mean={kr['mean_kendall_tau']} over {kr['groups_scored']} "
                  f"groups; pair coverage={kr['pair_coverage']}")
    print(f"\n[selective_metrics] wrote -> {out}")


if __name__ == "__main__":
    main()
