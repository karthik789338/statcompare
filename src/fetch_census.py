"""Phase 1 — Census ACS data acquisition.

Pulls one or more ACS tables ("groups") for the configured geographies, joins
each estimate with its variable label + table concept + population universe,
and writes a tidy parquet (one row per geography x variable) plus a metadata
JSON. These rows become the candidate Estimates for CompStat-Bench.

Run:
    python -m src.fetch_census --config config.yaml
    python -m src.fetch_census --selftest      # offline parser check, no network

The universe string is the key field: it is what lets the checker tell
"share among all workers" from "share among workers who commuted".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import requests
import pandas as pd
import yaml


VARIABLES_URL = "https://api.census.gov/data/{year}/{dataset}/groups/{group}.json"
DATA_URL = "https://api.census.gov/data/{year}/{dataset}"

# values that mean "no key" if someone leaves the placeholder in place
_PLACEHOLDER_KEYS = {"", "your_key_here", "xxxx", "none", "null", "changeme"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_key(k: Optional[str]) -> Optional[str]:
    """Return a usable key, or None (so the call goes out keyless)."""
    if not k:
        return None
    return None if k.strip().lower() in _PLACEHOLDER_KEYS else k.strip()


def _get_json(url: str, params: Optional[dict], timeout: int):
    """GET and parse JSON, with a clear error when Census returns a text error
    page (the usual symptom of a bad/placeholder API key)."""
    r = requests.get(url, params=params, timeout=timeout)
    try:
        return r.json()
    except ValueError:
        snippet = (r.text or "").strip()[:300].replace("\n", " ")
        raise RuntimeError(
            f"Census returned non-JSON (HTTP {r.status_code}) for {r.url}\n"
            f"  body: {snippet!r}\n"
            f"  Hint: a missing/placeholder key gives 'A valid key must be "
            f"included...'. Unset CENSUS_API_KEY to call keyless, or export a "
            f"real key from https://api.census.gov/data/key_signup.html"
        )


def fetch_variables(dataset: str, year: int, group: str, timeout: int) -> dict:
    """Return {variable_id: {label, concept, universe, predicateType, group}}."""
    url = VARIABLES_URL.format(year=year, dataset=dataset, group=group)
    raw = _get_json(url, None, timeout).get("variables", {})
    return _parse_variables(raw, group)


def _parse_variables(raw: dict, group: str) -> dict:
    """Pure parser, unit-tested offline."""
    out = {}
    for vid, meta in raw.items():
        if not vid.endswith("E"):   # keep estimate cells (…E); skip …M, …EA, etc.
            continue
        out[vid] = {
            "label": (meta.get("label") or "").replace("!!", " || ").strip(),
            "concept": meta.get("concept", ""),
            "universe": meta.get("universe", "") or meta.get("concept", ""),
            "predicateType": meta.get("predicateType", ""),
            "group": meta.get("group", group),
        }
    return out


def fetch_group_data(dataset: str, year: int, group: str, geo_for: str,
                     api_key: Optional[str], timeout: int) -> list[dict]:
    """Return list of {variable, value, geography, geo_name} for one group+geo."""
    url = DATA_URL.format(year=year, dataset=dataset)
    params = {"get": f"group({group})", "for": geo_for}
    if api_key:
        params["key"] = api_key
    return _parse_data(_get_json(url, params, timeout), group)


def _parse_data(rows: list[list], group: str) -> list[dict]:
    """Pure parser, unit-tested offline. rows[0] is the header."""
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    # geo id columns vary (state, us, place...). "GEO_ID" + "NAME" are standard.
    out = []
    for row in rows[1:]:
        geo_id = row[idx["GEO_ID"]] if "GEO_ID" in idx else ""
        geo_name = row[idx["NAME"]] if "NAME" in idx else ""
        for h, i in idx.items():
            if not (h.startswith(group) and h.endswith("E")):
                continue
            raw_val = row[i]
            try:
                val = float(raw_val) if raw_val not in (None, "", "null") else None
            except (TypeError, ValueError):
                val = None
            # census uses large negative sentinels for "not available"
            if val is not None and val <= -666666666:
                val = None
            out.append({
                "variable": h,
                "value": val,
                "geography": geo_id,
                "geo_name": geo_name,
            })
    return out


def build_rows(cfg: dict, skip_errors: bool = False) -> pd.DataFrame:
    dataset, year = cfg["dataset"], cfg["year"]
    api_key = normalize_key(cfg.get("api_key") or os.environ.get("CENSUS_API_KEY"))
    print(f"[fetch_census] auth: {'keyed' if api_key else 'keyless (low-volume OK)'}")
    timeout = cfg.get("request_timeout_s", 30)
    sleep = cfg.get("sleep_between_calls_s", 0.3)
    meta_dir = os.path.join(cfg["out_dir"], "meta")
    os.makedirs(meta_dir, exist_ok=True)

    all_rows = []
    for group in cfg["groups"]:
        try:
            variables = fetch_variables(dataset, year, group, timeout)
        except Exception as e:
            if skip_errors:
                print(f"[fetch_census] WARN skipping group {group}: {e}")
                continue
            raise
        with open(os.path.join(meta_dir, f"{group}_variables.json"), "w") as f:
            json.dump(variables, f, indent=2)
        for geo in cfg["geographies"]:
            level, geo_for = geo["level"], geo["for"]
            try:
                data = fetch_group_data(dataset, year, group, geo_for, api_key, timeout)
            except Exception as e:
                if skip_errors:
                    print(f"[fetch_census] WARN skipping {group}/{geo_for}: {e}")
                    continue
                raise
            for d in data:
                vmeta = variables.get(d["variable"], {})
                all_rows.append({
                    **d,
                    "group": group,
                    "geo_level": level,
                    "year": year,
                    "dataset": dataset,
                    "label": vmeta.get("label", ""),
                    "concept": vmeta.get("concept", ""),
                    "universe": vmeta.get("universe", ""),
                    "unit": "count",
                })
            time.sleep(sleep)
    return pd.DataFrame(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline parser checks, no network")
    ap.add_argument("--skip-errors", action="store_true",
                    help="warn and continue past a failing group/geography")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    cfg = load_config(args.config)
    df = build_rows(cfg, skip_errors=args.skip_errors)
    out = os.path.join(cfg["out_dir"], "raw", f"estimates_{cfg['year']}.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[fetch_census] wrote {len(df):,} rows -> {out}")
    print(df.groupby(['group'])['variable'].nunique().to_string())


# --------------------------------------------------------------------------- #
# Offline self-test: a tiny hardcoded sample of the two Census endpoints so the
# parsing logic can be verified on any machine with no network.
# --------------------------------------------------------------------------- #
_SAMPLE_VARS = {
    "variables": {
        "B08301_001E": {"label": "Estimate!!Total:", "concept": "Means of Transportation to Work",
                         "universe": "Workers 16 years and over", "group": "B08301", "predicateType": "int"},
        "B08301_003E": {"label": "Estimate!!Total:!!Car, truck, or van:!!Drove alone",
                         "concept": "Means of Transportation to Work",
                         "universe": "Workers 16 years and over", "group": "B08301", "predicateType": "int"},
        "B08301_021E": {"label": "Estimate!!Total:!!Worked from home",
                         "concept": "Means of Transportation to Work",
                         "universe": "Workers 16 years and over", "group": "B08301", "predicateType": "int"},
        "B08301_001M": {"label": "Margin of Error!!Total:", "concept": "x", "group": "B08301"},
    }
}
_SAMPLE_DATA = [
    ["GEO_ID", "NAME", "B08301_001E", "B08301_003E", "B08301_021E", "state"],
    ["0400000US06", "California", "18000000", "13000000", "1800000", "06"],
    ["0400000US48", "Texas", "13000000", "10000000", "900000", "48"],
]


def _selftest():
    v = _parse_variables(_SAMPLE_VARS["variables"], "B08301")
    assert set(v) == {"B08301_001E", "B08301_003E", "B08301_021E"}, "should keep only …E cells"
    assert v["B08301_001E"]["universe"] == "Workers 16 years and over"
    assert "Drove alone" in v["B08301_003E"]["label"]

    d = _parse_data(_SAMPLE_DATA, "B08301")
    ca = [x for x in d if x["geo_name"] == "California"]
    assert len(ca) == 3, "3 …E cells per geography"
    drove = [x for x in ca if x["variable"] == "B08301_003E"][0]
    assert drove["value"] == 13000000.0 and drove["geography"] == "0400000US06"

    # the denominator-swap this whole project is about:
    tot = {x["variable"]: x["value"] for x in ca}
    share_all = tot["B08301_003E"] / tot["B08301_001E"]
    share_commuters = tot["B08301_003E"] / (tot["B08301_001E"] - tot["B08301_021E"])
    assert share_commuters > share_all, "excluding WFH raises the drove-alone share"
    print("[selftest] OK — variable/data parsers and denominator arithmetic verified")
    print(f"[selftest] CA drove-alone: among all workers={share_all:.3f}, "
          f"among commuters={share_commuters:.3f}")


if __name__ == "__main__":
    main()
