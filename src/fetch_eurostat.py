"""Phase 10 — Eurostat adapter (regional old-age-dependency flip).

Reads Eurostat table demo_r_pjangrp3 (population on 1 January by age group,
sex and NUTS region) as JSON-stat and emits estimates in the shared schema
(same columns as fetch_census), restricted to NUTS-2 regions.

Flip family (old-age dependency / denominator trap):
  numerator  ES_GE65   = population aged 65+            (sum of 65-69 .. GE85)
  denom all  ES_TOTAL  = total population, all ages      (DIRECT)
  denom sub  working-age 15-64 = sum of the ten 5-year bands  (RECONSTRUCT)
"Share of the whole population aged 65+" vs "share of the working-age
population aged 65+" reorders regions with different child populations
(the crude vs refined old-age dependency ratio).

JSON-stat layout for this table: id order [freq,sex,unit,age,geo,time];
the non-age/geo dims are size 1, so each value sits at a flat index we
recover with standard row-major strides.
"""
from __future__ import annotations
import os, sys, argparse
import requests
import pandas as pd

URL = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
       "demo_r_pjangrp3")

WORKING_AGE = {
    "Y15-19": "ES_A1519", "Y20-24": "ES_A2024", "Y25-29": "ES_A2529",
    "Y30-34": "ES_A3034", "Y35-39": "ES_A3539", "Y40-44": "ES_A4044",
    "Y45-49": "ES_A4549", "Y50-54": "ES_A5054", "Y55-59": "ES_A5559",
    "Y60-64": "ES_A6064",
}
OLD_CODES = ["Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y_GE85"]
UNIVERSE = "Total population all ages"
GROUP = "ES_OLDAGE"


def _strides(size):
    st = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        st[i] = st[i + 1] * size[i + 1]
    return st


def fetch_json(year, timeout=60) -> dict:
    r = requests.get(URL, params={"format": "JSON", "sex": "T", "unit": "NR",
                                  "time": year}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def build_rows(year, timeout=60) -> pd.DataFrame:
    d = fetch_json(year, timeout)
    ids, size = d["id"], d["size"]
    st = _strides(size)
    pos = {name: i for i, name in enumerate(ids)}
    age_i = d["dimension"]["age"]["category"]["index"]
    geo_i = d["dimension"]["geo"]["category"]["index"]
    geo_lab = d["dimension"]["geo"]["category"]["label"]
    values = d["value"]

    def cell(age_code, geo_code):
        if age_code not in age_i or geo_code not in geo_i:
            return None
        coord = [0] * len(size)
        coord[pos["age"]] = age_i[age_code]
        coord[pos["geo"]] = geo_i[geo_code]
        flat = sum(c * s for c, s in zip(coord, st))
        return values.get(str(flat))

    # NUTS-2 geos: 4-char codes beginning with two letters (excludes countries
    # (2), NUTS-1 (3), NUTS-3 (5) and EU aggregates like EU27_2020).
    nuts2 = [g for g in geo_i
             if len(g) == 4 and g[:2].isalpha() and not g.startswith("EU")]

    rows = []
    for g in nuts2:
        name = geo_lab.get(g, g)
        total = cell("TOTAL", g)
        wa = {var: cell(code, g) for code, var in WORKING_AGE.items()}
        old = [cell(c, g) for c in OLD_CODES]
        if total is None or any(v is None for v in wa.values()) or any(
                v is None for v in old):
            continue
        ge65 = sum(old)
        rec = [("ES_TOTAL", total, "Population, all ages"),
               ("ES_GE65", ge65, "Population aged 65 or over")]
        rec += [(var, wa[var], f"Population {code}")
                for code, var in WORKING_AGE.items()]
        for var, val, label in rec:
            rows.append({
                "year": year, "dataset": "eurostat/demo_r_pjangrp3",
                "geography": g, "geo_name": name, "geo_level": "nuts2",
                "variable": var, "value": float(val),
                "universe": UNIVERSE, "label": label,
                "concept": "Age structure", "group": GROUP})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--out", default="data/raw/eurostat.parquet")
    args = ap.parse_args()
    df = build_rows(args.year)
    if df.empty:
        print("[fetch_eurostat] ERROR: 0 rows"); sys.exit(1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    n_geo = df["geography"].nunique()
    print(f"[fetch_eurostat] wrote {len(df):,} rows, {n_geo} NUTS-2 regions -> {args.out}")


if __name__ == "__main__":
    main()
