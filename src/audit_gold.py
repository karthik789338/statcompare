#!/usr/bin/env python3
"""Phase 13 - expert-style face-validity audit of the gold labels.

For a stratified sample across the four sources x flip/non-flip x gold verdict,
re-derive each gold label INDEPENDENTLY of the stored `shares`/`ordering` fields
and check it is internally correct:

  * published provenance -- the numerator candidate's directly-published value
      (numerator / its natural universe) equals EITHER the requested or the naive
      stored share for that place (it is one of the two framings by construction);
  * counts recompute     -- where the raw integer counts are present, numerator /
      denominator recomputed here equals one of the two stored shares (a genuine
      re-derivation of a share from the raw fetched counts);
  * ordering             -- for answerable items, gold_ordering / naive_ordering
      re-sorted from the stored shares under gold_frame.direction match the stored
      orderings (no sort/label bug);
  * flip                 -- is_flip == (gold_ordering != naive_ordering);
  * abstain              -- not_answerable items store gold_ordering = None.

Prints a per-item PASS/FAIL table, an overall agreement rate, and every
discrepancy in full. No GPU, no network.
"""
import json, argparse, random
from collections import defaultdict

FILES = [
    ("data/bench/compstat.jsonl", None),
    ("data/bench/compstat_worldbank.jsonl", "worldbank"),
    ("data/bench/compstat_eurostat.jsonl", "eurostat"),
]
EPS = 1e-6


def source_of(it, file_tag):
    if file_tag:
        return file_tag
    return "census-employment" if it.get("family", "").startswith("B23025") else "census-transport"


def numerator_var(it):
    return it["family"].split(":", 1)[1]


def denom_slug(it):
    fam_prefix = it["family"].split(":", 1)[0]
    return f"{fam_prefix}__{it['gold_frame']['denominator'].replace(' ', '_')}"


def geoid_of(it, place):
    for c in it["candidates"]:
        if c.get("geo_name") == place and c.get("geography"):
            return c["geography"]
    return None


def numerator_value(it, place):
    nv = numerator_var(it)
    for c in it["candidates"]:
        if c.get("geo_name") == place and c.get("variable") == nv and c.get("value") is not None:
            return c["value"]
    return None


def order(shares_map, places, direction):
    return sorted(places, key=lambda p: shares_map[p], reverse=(direction == "greater"))


def _near_one_of(v, a, b, tol=EPS):
    return abs(v - a) < tol or abs(v - b) < tol


def audit_item(it):
    gf = it["gold_frame"]
    places = gf["places"]
    direction = gf.get("direction", "greater")
    verdict = it.get("gold_verdict")
    counts = it.get("counts") or {}
    req, nai = it["shares"]["requested"], it["shares"]["naive"]
    go, no = it.get("gold_ordering"), it.get("naive_ordering")
    checks, msgs = {}, []

    # abstain / ordering / flip
    if verdict == "not_answerable" or go is None:
        checks["abstains"] = (go is None and verdict == "not_answerable")
        if not checks["abstains"]:
            msgs.append(f"abstain mismatch: verdict={verdict} gold_ordering={go}")
    else:
        gre, nre = order(req, places, direction), order(nai, places, direction)
        checks["ordering"] = (gre == go and nre == no)
        if not checks["ordering"]:
            msgs.append(f"ordering: gold_re={gre} stored={go} | naive_re={nre} stored={no}")
        checks["flip"] = (bool(it["is_flip"]) == (go != no))
        if not checks["flip"]:
            msgs.append(f"flip: stored={it['is_flip']} recomputed={go != no}")

    # published-value provenance: numerator/natural-universe == requested or naive
    prov = True
    for p in places:
        v = numerator_value(it, p)
        if v is None:
            continue
        if not _near_one_of(v, req[p], nai[p]):
            prov = False
            msgs.append(f"published provenance {p}: value={v} != requested {req[p]} and != naive {nai[p]}")
    checks["published_provenance"] = prov

    # raw-counts recompute: numerator/denominator == requested or naive
    if any("__" in k for k in counts):
        nv, dslug = numerator_var(it), denom_slug(it)
        rec = True
        for p in places:
            g = geoid_of(it, p)
            nk, dk = f"{g}|{nv}", f"{g}|{dslug}"
            if g and nk in counts and dk in counts and counts[dk]:
                val = counts[nk] / counts[dk]
                if not _near_one_of(val, req[p], nai[p], 1e-6):
                    rec = False
                    msgs.append(f"counts recompute {p}: {counts[nk]}/{counts[dk]}={val:.6f} != req {req[p]:.6f} and != naive {nai[p]:.6f}")
        checks["counts_recompute"] = rec

    checks["verdict_value"] = verdict in {"direct", "reconstruct", "not_answerable"}
    return all(checks.values()), checks, msgs


def load_all():
    items = []
    for path, tag in FILES:
        try:
            for l in open(path):
                it = json.loads(l)
                it["_source"] = source_of(it, tag)
                items.append(it)
        except FileNotFoundError:
            pass
    return items


def stratified_sample(items, per_stratum, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for it in items:
        strata[(it["_source"], bool(it["is_flip"]), it.get("gold_verdict"))].append(it)
    sample = []
    for key in sorted(strata, key=lambda k: (str(k[0]), k[1], str(k[2]))):
        b = strata[key][:]
        rng.shuffle(b)
        sample.extend(b[:per_stratum])
    return sample, strata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        # reconstruct flip: naive (published) .5/.7 ; requested (reconstructed) 50/60, 70/100
        A = {
            "family": "X:X_NUM", "gold_verdict": "reconstruct", "is_flip": True,
            "gold_frame": {"places": ["A", "B"], "direction": "greater", "denominator": "commuters"},
            "gold_ordering": ["A", "B"], "naive_ordering": ["B", "A"],
            "shares": {"requested": {"A": 50 / 60, "B": 0.70}, "naive": {"A": 0.50, "B": 0.70}},
            "counts": {"gA|X_NUM": 50.0, "gA|X__commuters": 60.0, "gB|X_NUM": 70.0, "gB|X__commuters": 100.0},
            "candidates": [
                {"geo_name": "A", "geography": "gA", "variable": "X_NUM", "value": 0.50},
                {"geo_name": "B", "geography": "gB", "variable": "X_NUM", "value": 0.70},
            ],
        }
        ok, checks, msgs = audit_item(A)
        assert ok, (checks, msgs)
        bad = json.loads(json.dumps(A)); bad["gold_ordering"] = ["B", "A"]
        assert not audit_item(bad)[0]
        na = json.loads(json.dumps(A)); na["gold_verdict"] = "not_answerable"; na["gold_ordering"] = None
        assert audit_item(na)[0]
        print("[audit] selftest OK")
        return

    items = load_all()
    sample, strata = stratified_sample(items, args.per_stratum, args.seed)
    print(f"[audit] {len(items)} items total; sampled {len(sample)} across {len(strata)} strata "
          f"(<= {args.per_stratum}/stratum, seed {args.seed})\n")
    print("strata (source, flip, verdict) -> population size:")
    for key in sorted(strata, key=lambda k: (str(k[0]), k[1], str(k[2]))):
        print(f"  {key[0]:18s} flip={str(key[1]):5s} {str(key[2]):14s} n={len(strata[key])}")
    print()

    npass, fails = 0, []
    print(f"{'id':12s} {'source':18s} {'flip':5s} {'verdict':14s} result")
    for it in sample:
        ok, checks, msgs = audit_item(it)
        npass += ok
        flag = "PASS" if ok else "FAIL(" + ",".join(k for k, v in checks.items() if not v) + ")"
        print(f"{it['id']:12s} {it['_source']:18s} {str(bool(it['is_flip'])):5s} {str(it.get('gold_verdict')):14s} {flag}")
        if not ok:
            fails.append((it["id"], msgs))
    print(f"\n[audit] agreement: {npass}/{len(sample)} = {npass/len(sample):.4f}")
    if fails:
        print("\nDISCREPANCIES:")
        for iid, msgs in fails:
            for m in msgs:
                print(f"  {iid}: {m}")
    else:
        print("[audit] no discrepancies - every sampled gold label re-derives independently.")


if __name__ == "__main__":
    main()
