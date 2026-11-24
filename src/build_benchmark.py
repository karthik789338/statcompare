"""Phase 2 — build CompStat-Bench.

From the Phase-1 estimates table, generate matched comparison questions where
only the population/denominator phrase changes ("among all workers" vs "among
workers who traveled to work"), attach gold labels with the symbolic checker,
flag the discriminative FLIP pairs (ordering changes with the denominator), and
split by indicator family so success requires more than memorising table names.

Each emitted item is a ready ranking instance:
  { id, question, gold_frame, candidates[], counts{}, gold_verdict,
    gold_ordering, naive_ordering, is_flip, family, split }

Run:
    python -m src.build_benchmark --config config.yaml --recipes recipes.yaml
    python -m src.build_benchmark --selftest     # offline, fabricates tiny data
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from dataclasses import asdict
from typing import Optional

import yaml

from .frames import Estimate, ComparisonFrame
from .checker import check, Verdict


# --------------------------------------------------------------------------- #
# data access helpers
# --------------------------------------------------------------------------- #
def _load_estimates(parquet_path: str):
    """Return (val, meta, geoname, geolevel, year, dataset).

    val[(geo, var)] = value ; meta[var] = {universe,label,concept,group}
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    val, meta, geoname, geolevel = {}, {}, {}, {}
    year = int(df["year"].iloc[0]); dataset = str(df["dataset"].iloc[0])
    for r in df.itertuples(index=False):
        val[(r.geography, r.variable)] = r.value
        meta.setdefault(r.variable, {
            "universe": r.universe, "label": r.label,
            "concept": r.concept, "group": r.group})
        geoname[r.geography] = r.geo_name
        geolevel[r.geography] = r.geo_level
    return val, meta, geoname, geolevel, year, dataset


def _denom_value(val: dict, geo: str, formula: list[str]) -> Optional[float]:
    total = 0.0
    for term in formula:
        sign, var = (term[0], term[1:]) if term[0] in "+-" else ("+", term)
        v = val.get((geo, var))
        if v is None:
            return None
        total += v if sign == "+" else -v
    return total


# --------------------------------------------------------------------------- #
# item construction
# --------------------------------------------------------------------------- #
def _make_item(item_id, group, num, denom, other_denom, geoA, geoB,
               val, meta, geoname, geolevel, year, dataset, n_distractors, rng,
               template=None):
    """Build one ranking item for a (numerator, denominator, placeA, placeB)."""
    numvar = num["var"]
    uni = meta[numvar]["universe"]
    base_key = f"{group}__{denom['name'].replace(' ', '_')}"

    # ground-truth shares under THIS denominator and the OTHER one (for is_flip)
    def share(geo, formula):
        n = val.get((geo, numvar)); d = _denom_value(val, geo, formula)
        return (n / d) if (n is not None and d not in (None, 0)) else None

    sA, sB = share(geoA, denom["formula"]), share(geoB, denom["formula"])
    oA, oB = share(geoA, other_denom["formula"]), share(geoB, other_denom["formula"])
    if None in (sA, sB, oA, oB):
        return None

    gold_order = [geoname[geoA], geoname[geoB]] if sA >= sB else [geoname[geoB], geoname[geoA]]
    naive_order = [geoname[geoA], geoname[geoB]] if oA >= oB else [geoname[geoB], geoname[geoA]]
    # "naive" = compare the share on the OTHER (default, all-workers) denominator,
    # which is what a system that ignores the requested denominator would do.
    is_flip = (gold_order != naive_order)

    # does the requested denominator match the cell's own universe? -> verdict
    direct = (denom["name"].lower().startswith("all"))
    requested_denom = denom["name"]

    def cand(geo, s_default):
        # the retrieved estimate is the share on the cell's own (all-workers)
        # universe; for the commuters question this is the "wrong" denominator.
        return Estimate(
            group=group, variable=numvar, label=meta[numvar]["label"],
            concept=meta[numvar]["concept"], universe=uni,
            geography=geo, geo_name=geoname[geo], geo_level=geolevel[geo],
            year=year, dataset=dataset, unit="percent",
            value=s_default, base_variable=(None if direct else base_key))

    default_formula = next(d for d in (denom, other_denom)
                           if d["name"].lower().startswith("all"))["formula"]
    a = cand(geoA, share(geoA, default_formula))
    b = cand(geoB, share(geoB, default_formula))

    # counts needed for the RECONSTRUCT path
    counts = {}
    if not direct:
        for geo in (geoA, geoB):
            counts[f"{geo}|{numvar}"] = val.get((geo, numvar))
            counts[f"{geo}|{base_key}"] = _denom_value(val, geo, denom["formula"])

    frame = ComparisonFrame(
        question="", measure=num["name"], population=requested_denom,
        denominator=requested_denom, unit="percent", period=year,
        geo_level=geolevel[geoA], places=[geoname[geoA], geoname[geoB]],
        direction="greater")
    tmpl = template or ("In which place did a larger share {phrase} {measure} "
                        "to get to work: {A} or {B}?")
    question = tmpl.format(phrase=denom["phrase"], measure=num["name"],
                           A=geoname[geoA], B=geoname[geoB])
    frame.question = question

    # distractors: other numerator cells in the same table, same places
    distractors = []
    pool = [v for v in meta if v != numvar and meta[v]["group"] == group]
    rng.shuffle(pool)
    for dv in pool[:n_distractors]:
        for geo in (geoA, geoB):
            distractors.append(Estimate(
                group=group, variable=dv, label=meta[dv]["label"],
                concept=meta[dv]["concept"], universe=meta[dv]["universe"],
                geography=geo, geo_name=geoname[geo], geo_level=geolevel[geo],
                year=year, dataset=dataset, unit="percent", value=None))

    # gold verdict from the symbolic checker (re-keying counts to tuple form)
    tuple_counts = {(k.split("|")[0], k.split("|")[1]): v for k, v in counts.items()}
    res = check(frame, a, b, counts=tuple_counts)

    return {
        "id": item_id,
        "question": question,
        "gold_frame": asdict(frame),
        "candidates": [a.to_dict(), b.to_dict()] + [d.to_dict() for d in distractors],
        "counts": counts,
        "gold_verdict": res.verdict.value,
        "gold_ordering": res.ordering,
        "naive_ordering": naive_order,
        "is_flip": is_flip,
        "family": f"{group}:{numvar}",
        "shares": {"requested": {geoname[geoA]: sA, geoname[geoB]: sB},
                   "naive": {geoname[geoA]: oA, geoname[geoB]: oB}},
    }


def build(cfg: dict, recipes: dict, parquet_path: str):
    val, meta, geoname, geolevel, year, dataset = _load_estimates(parquet_path)
    rng = random.Random(recipes["sampling"].get("random_seed", 13))
    cap = recipes["sampling"]["max_pairs_per_cell"]
    keep_flips = recipes["sampling"].get("keep_all_flips", True)
    gap = recipes["sampling"].get("min_share_gap", 0.0)

    geos = sorted({g for (g, _) in val})
    items = []
    n = 0
    for group, fam in recipes["families"].items():
        denoms = fam["denominators"]
        if len(denoms) < 2:
            continue
        d_all = next(d for d in denoms if d["name"].lower().startswith("all"))
        d_sub = next(d for d in denoms if not d["name"].lower().startswith("all"))
        for num in fam["numerators"]:
            if geos and not any((g, num["var"]) in val for g in geos):
                continue
            pairs = list(itertools.combinations(geos, 2))
            rng.shuffle(pairs)
            kept, flips_first = [], []
            for (gA, gB) in pairs:
                for denom, other in ((d_all, d_sub), (d_sub, d_all)):
                    it = _make_item(f"it{n:06d}", group, num, denom, other, gA, gB,
                                    val, meta, geoname, geolevel, year, dataset,
                                    recipes.get("n_distractors", 3), rng,
                                    template=fam.get("question_template"))
                    if it is None:
                        continue
                    sa, sb = it["shares"]["requested"].values()
                    if abs(sa - sb) < gap:
                        continue
                    n += 1
                    (flips_first if it["is_flip"] else kept).append(it)
            # keep all flips + sample of non-flips up to cap
            take = (flips_first if keep_flips else []) + kept
            rng.shuffle(kept)
            take = (flips_first if keep_flips else kept[:cap]) + kept[:max(0, cap - len(flips_first))]
            items.extend(flips_first + kept[:max(0, cap - len(flips_first))])

    # split: held-out families -> test_ood; held-out states -> dev/test (so those
    # states never appear in train); everything else -> train.
    held = set(recipes.get("held_out_families", []))
    sp = recipes.get("split", {})
    dev_states = set(sp.get("dev_states", []))
    test_states = set(sp.get("test_states", []))
    forced = cfg.get("split_all_as")
    for it in items:
        grp = it["family"].split(":")[0]
        places = set(it["gold_frame"]["places"])
        if forced:
            it["split"] = forced
        elif grp in held:
            it["split"] = "test_ood"
        elif places & test_states:
            it["split"] = "test"
        elif places & dev_states:
            it["split"] = "dev"
        else:
            it["split"] = "train"
    return items


def _write(items, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    # summary
    from collections import Counter
    sp = Counter(i["split"] for i in items)
    fl = sum(i["is_flip"] for i in items)
    vd = Counter(i["gold_verdict"] for i in items)
    print(f"[build_benchmark] wrote {len(items):,} items -> {out_path}")
    print(f"  splits: {dict(sp)}")
    print(f"  flip pairs (ordering changes with denominator): {fl:,} "
          f"({100*fl/max(1,len(items)):.1f}%)")
    print(f"  gold verdicts: {dict(vd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--recipes", default="recipes.yaml")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    cfg = yaml.safe_load(open(args.config))
    recipes = yaml.safe_load(open(args.recipes))
    parquet = os.path.join(cfg["out_dir"], "raw",
                           cfg.get("raw_file", f"estimates_{cfg['year']}.parquet"))
    items = build(cfg, recipes, parquet)
    _write(items, os.path.join(cfg["out_dir"], "bench",
                               cfg.get("bench_file", "compstat.jsonl")))


# --------------------------------------------------------------------------- #
# offline self-test: fabricate a 3-state estimates parquet with a known flip
# --------------------------------------------------------------------------- #
def _selftest():
    import pandas as pd, tempfile
    # State X: high WFH; State Y: low WFH. Drove-alone counts chosen so the
    # ordering flips between the all-workers and commuters denominators.
    rows = []
    def add(geo, name, tot, drove, wfh):
        for var, v in [("B08301_001E", tot), ("B08301_003E", drove),
                       ("B08301_021E", wfh), ("B08301_004E", tot*0.1)]:
            rows.append(dict(variable=var, value=float(v), geography=geo,
                             geo_name=name, group="B08301", geo_level="state",
                             year=2023, dataset="acs/acs1",
                             label=var, concept="Means of Transportation to Work",
                             universe="Workers 16 years and over", unit="count"))
    # X: drove/all = 0.60, commuters base = 100-25=75 -> 60/75=0.80
    add("X", "Statex", 100, 60, 25)
    # Y: drove/all = 0.62, commuters base = 100-3=97 -> 62/97=0.639
    add("Y", "Statey", 100, 62, 3)
    # all-workers: Y(0.62) > X(0.60);  commuters: X(0.80) > Y(0.639)  => FLIP
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as td:
        pq = os.path.join(td, "e.parquet"); df.to_parquet(pq, index=False)
        recipes = yaml.safe_load(open(os.path.join(os.path.dirname(__file__),
                                                   "..", "recipes.yaml")))
        recipes["families"] = {"B08301": recipes["families"]["B08301"]}
        recipes["families"]["B08301"]["numerators"] = [
            {"var": "B08301_003E", "name": "drove alone"}]
        items = build({"out_dir": td, "year": 2023}, recipes, pq)
    assert items, "should emit items"
    byd = {i["gold_frame"]["denominator"]: i for i in items}
    assert any(i["is_flip"] for i in items), "the constructed pair must flip"
    direct = [i for i in items if i["gold_verdict"] == "direct"][0]
    recon = [i for i in items if i["gold_verdict"] == "reconstruct"][0]
    # direct (all workers): Y ranks first
    assert direct["gold_ordering"][0] == "Statey", direct["gold_ordering"]
    # reconstruct (commuters): X ranks first, and it differs from naive
    assert recon["gold_ordering"][0] == "Statex", recon["gold_ordering"]
    assert recon["naive_ordering"][0] == "Statey", recon["naive_ordering"]
    print("[selftest] OK — flip constructed, direct/reconstruct verdicts and "
          "orderings correct")
    print(f"[selftest] emitted {len(items)} items; "
          f"flips={sum(i['is_flip'] for i in items)}")


if __name__ == "__main__":
    main()
