"""Unit tests for the symbolic comparability checker.

Run:  python -m pytest -q   (or)   python tests/test_checker.py
These use synthetic estimates, so they need no network, no GPU, no Census key.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.frames import Estimate, ComparisonFrame
from src.checker import check, Verdict


def _est(var, universe, geo, name, value, base=None, unit="count", year=2023):
    return Estimate(group="B08301", variable=var, label=var, concept="MoT",
                    universe=universe, geography=geo, geo_name=name,
                    geo_level="state", year=year, dataset="acs/acs1",
                    unit=unit, value=value, base_variable=base)


def test_direct_same_universe():
    f = ComparisonFrame(question="q", measure="drove alone", population="all workers",
                        denominator="all workers", unit="percent", direction="greater",
                        places=["California", "Texas"])
    a = _est("B08301_003E", "Workers 16 years and over", "06", "California", 0.72)
    b = _est("B08301_003E", "Workers 16 years and over", "48", "Texas", 0.80)
    r = check(f, a, b)
    assert r.verdict == Verdict.DIRECT
    assert r.ordering == ["Texas", "California"]


def test_reconstruct_denominator_swap():
    # user wants the share among workers who COMMUTED (excludes work-from-home),
    # but the retrieved cell's universe is all workers -> must re-base.
    f = ComparisonFrame(question="q", measure="drove alone",
                        population="workers who commuted",
                        denominator="workers who commuted", unit="percent",
                        direction="greater", places=["California", "Texas"])
    a = _est("B08301_003E", "Workers 16 years and over", "06", "California", None,
             base="B08301_commuters")
    b = _est("B08301_003E", "Workers 16 years and over", "48", "Texas", None,
             base="B08301_commuters")
    counts = {
        ("06", "B08301_003E"): 13_000_000, ("06", "B08301_commuters"): 16_200_000,
        ("48", "B08301_003E"): 10_000_000, ("48", "B08301_commuters"): 12_100_000,
    }
    r = check(f, a, b, counts=counts)
    assert r.verdict == Verdict.RECONSTRUCT
    # CA 0.802 vs TX 0.826 -> Texas higher
    assert r.ordering == ["Texas", "California"]


def test_reconstruct_missing_counts_not_answerable():
    f = ComparisonFrame(question="q", measure="drove alone",
                        population="workers who commuted",
                        denominator="workers who commuted", unit="percent",
                        places=["California", "Texas"])
    a = _est("B08301_003E", "Workers 16 years and over", "06", "California", None,
             base="B08301_commuters")
    b = _est("B08301_003E", "Workers 16 years and over", "48", "Texas", None,
             base="B08301_commuters")
    r = check(f, a, b, counts={})  # no component counts available
    assert r.verdict == Verdict.NOT_ANSWERABLE
    assert r.needed_counts


def test_unit_mismatch_not_answerable():
    f = ComparisonFrame(question="q", measure="x", population="all workers",
                        denominator="all workers", unit="percent",
                        places=["California", "Texas"])
    a = _est("B08301_003E", "Workers 16 years and over", "06", "California", 0.7, unit="percent")
    b = _est("B08301_003E", "Workers 16 years and over", "48", "Texas", 10_000_000, unit="count")
    assert check(f, a, b).verdict == Verdict.NOT_ANSWERABLE


def test_period_mismatch_not_answerable():
    f = ComparisonFrame(question="q", measure="x", population="all workers",
                        denominator="all workers", unit="percent",
                        places=["California", "Texas"])
    a = _est("B08301_003E", "Workers 16 years and over", "06", "California", 0.7, year=2023)
    b = _est("B08301_003E", "Workers 16 years and over", "48", "Texas", 0.8, year=2019)
    assert check(f, a, b).verdict == Verdict.NOT_ANSWERABLE


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} checker tests passed.")
