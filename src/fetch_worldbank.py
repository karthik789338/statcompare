"""Phase 10 — World Bank WDI adapter.

Emits estimates in the SAME schema as fetch_census (one parquet row per
(country, variable)) so build_benchmark/checker consume it unchanged:
columns = year, dataset, geography, geo_name, geo_level, variable, value,
universe, label, concept, group.

Flip family (labour-force-participation / crude-vs-refined-rate trap):
  numerator  WB_LF        = labour force, total            (SL.TLF.TOTL.IN)
  denom all  WB_POP_TOTAL = total population               (SP.POP.TOTL)
  denom sub  WB_POP_1564  = population ages 15-64           (SP.POP.1564.TO)
"Economically active as a share of the whole population" vs "...of the
working-age population" reorders countries with different age structures.

The v2 API sits behind an intermittent Cloudflare check that passes with a
browser User-Agent; we set one on every call.
"""
from __future__ import annotations
import os, sys, time, argparse
import requests
import pandas as pd

BASE = "https://api.worldbank.org/v2"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
           "Accept": "application/json"}

# WDI indicator code -> (our variable name, universe, concept, human label)
INDICATORS = {
    "SP.POP.TOTL":    ("WB_POP_TOTAL", "Total population", "Population",
                       "Population, total"),
    "SP.POP.1564.TO": ("WB_POP_1564", "Working-age population 15 to 64",
                       "Population", "Population ages 15-64, total"),
    "SL.TLF.TOTL.IN": ("WB_LF", "Total population", "Labor force",
                       "Labor force, total"),
}


def _get(url, params, timeout):
    params = dict(params or {})
    params.setdefault("format", "json")
    last=None
    for _a in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=90)
            r.raise_for_status(); return r.json()
        except Exception as _e:
            last=_e; time.sleep(3*(_a+1))
    raise last


def real_countries(timeout=30) -> dict:
    """ISO3 -> name for actual countries (region != 'Aggregates')."""
    out = {}
    page = 1
    while True:
        meta, rows = _get(f"{BASE}/country", {"per_page": 400, "page": page},
                          timeout)
        for c in rows or []:
            if (c.get("region") or {}).get("value") != "Aggregates":
                out[c["id"]] = c["name"]          # c['id'] is the ISO3 code
        if page >= int(meta["pages"]):
            break
        page += 1
    return out


def fetch_indicator(code, year, timeout=30, sleep=0.2) -> dict:
    """ISO3 -> value for one indicator in one year (all countries)."""
    out, page = {}, 1
    while True:
        meta, rows = _get(f"{BASE}/country/all/indicator/{code}",
                          {"date": year, "per_page": 500, "page": page}, timeout)
        for d in rows or []:
            iso3 = d.get("countryiso3code")
            v = d.get("value")
            if iso3 and v is not None:
                out[iso3] = float(v)
        if page >= int(meta["pages"]):
            break
        page += 1
        time.sleep(sleep)
    return out


def build_rows(year, timeout=30) -> pd.DataFrame:
    countries = real_countries(timeout)
    print(f"[fetch_worldbank] {len(countries)} countries")
    series = {}
    for code in INDICATORS:
        series[code] = fetch_indicator(code, year, timeout)
        print(f"[fetch_worldbank] {code}: {len(series[code])} values")
    rows = []
    for iso3, name in countries.items():
        for code, (var, universe, concept, label) in INDICATORS.items():
            v = series[code].get(iso3)
            if v is None:
                continue
            rows.append({
                "year": year, "dataset": "worldbank/WDI",
                "geography": iso3, "geo_name": name, "geo_level": "country",
                "variable": var, "value": v,
                "universe": universe, "label": label,
                "concept": concept, "group": "WB_LFP"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--out", default="data/raw/worldbank.parquet")
    args = ap.parse_args()
    df = build_rows(args.year)
    if df.empty:
        print("[fetch_worldbank] ERROR: 0 rows"); sys.exit(1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    n_geo = df["geography"].nunique()
    print(f"[fetch_worldbank] wrote {len(df):,} rows, {n_geo} countries -> {args.out}")


if __name__ == "__main__":
    main()
