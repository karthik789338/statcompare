"""Symbolic comparability checker (Phase 3 core; previewed in Phase 1).

Given a ComparisonFrame and two candidate Estimates, decide whether the two can
be *validly* compared, and if so, which place ranks higher. Three verdicts:

  - DIRECT        : same universe/unit/period/geo-level and same concept cell;
                    compare the two values as-is.
  - RECONSTRUCT   : the requested share uses a denominator different from the
                    cell's own base, but the component counts needed to re-base
                    both estimates onto the requested denominator are available.
  - NOT_ANSWERABLE: units/periods/geo-levels are incompatible, or the counts
                    needed to put both estimates on the same denominator are
                    missing.

This module is deterministic and dependency-free so it can serve both as the
label oracle for CompStat-Bench and as a baseline/verifier at eval time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from .frames import Estimate, ComparisonFrame


class Verdict(str, Enum):
    DIRECT = "direct"
    RECONSTRUCT = "reconstruct"
    NOT_ANSWERABLE = "not_answerable"


@dataclass
class CheckResult:
    verdict: Verdict
    ordering: Optional[list] = None      # geo_names, highest first, when decidable
    needed_counts: Optional[list] = None # variable ids to retrieve for RECONSTRUCT
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "ordering": self.ordering,
            "needed_counts": self.needed_counts,
            "reason": self.reason,
        }


def _share(numerator: float, denominator: float) -> Optional[float]:
    if denominator in (None, 0) or numerator is None:
        return None
    return numerator / denominator


def check(
    frame: ComparisonFrame,
    a: Estimate,
    b: Estimate,
    counts: Optional[dict] = None,
) -> CheckResult:
    """Decide comparability of estimates ``a`` and ``b`` for ``frame``.

    ``counts`` optionally maps (geo_id, variable_id) -> value, supplying the
    component counts needed for the RECONSTRUCT path (numerator + requested
    base) when the cell's own base differs from the one the user asked for.
    """
    counts = counts or {}

    # --- hard incompatibilities -> NOT_ANSWERABLE ---------------------------
    if a.year != b.year or a.dataset != b.dataset:
        return CheckResult(Verdict.NOT_ANSWERABLE,
                           reason=f"period mismatch: {a.year}/{a.dataset} vs {b.year}/{b.dataset}")
    if a.geo_level != b.geo_level:
        return CheckResult(Verdict.NOT_ANSWERABLE,
                           reason=f"geo-level mismatch: {a.geo_level} vs {b.geo_level}")
    if a.unit != b.unit:
        return CheckResult(Verdict.NOT_ANSWERABLE,
                           reason=f"unit mismatch: {a.unit} vs {b.unit}")
    # the two estimates must measure the same concept cell across two places
    if a.variable != b.variable:
        return CheckResult(Verdict.NOT_ANSWERABLE,
                           reason=f"different cells: {a.variable} vs {b.variable}")

    want_denom = (frame.denominator or "").strip().lower()
    cell_universe = (a.universe or "").strip().lower()

    # --- DIRECT: cell's own universe already matches the requested denominator
    if want_denom in ("", cell_universe) or _universe_matches(want_denom, cell_universe):
        ordering = _order([a, b], values={a.geo_name: a.value, b.geo_name: b.value},
                          direction=frame.direction)
        return CheckResult(Verdict.DIRECT, ordering=ordering,
                           reason=f"direct: cell universe '{a.universe}' matches requested denominator")

    # --- RECONSTRUCT: requested denominator differs from the cell's universe.
    # We can still answer if, for each place, we have the numerator count and
    # the requested base count. base_variable on the Estimate names the base;
    # the question's denominator tells us which base the user wants.
    need = []
    reshare = {}
    ok = True
    for est in (a, b):
        num = counts.get((est.geography, est.variable))
        base_var = _requested_base_variable(frame, est)
        base_val = counts.get((est.geography, base_var)) if base_var else None
        if num is None:
            need.append((est.geography, est.variable))
        if base_var is None or base_val is None:
            need.append((est.geography, base_var or "<requested-base>"))
        s = _share(num, base_val)
        if s is None:
            ok = False
        reshare[est.geo_name] = s
    if ok:
        ordering = _order([a, b], values=reshare, direction=frame.direction)
        return CheckResult(Verdict.RECONSTRUCT, ordering=ordering,
                           needed_counts=sorted(set(need)),
                           reason="reconstructed share onto the requested denominator")
    return CheckResult(Verdict.NOT_ANSWERABLE, needed_counts=sorted(set(need)),
                       reason="requested denominator differs and component counts are missing")


def _universe_matches(want_denom: str, cell_universe: str) -> bool:
    """Loose match between the requested denominator phrase and a census universe."""
    if not want_denom:
        return True
    toks_w = set(want_denom.replace(",", " ").split())
    toks_u = set(cell_universe.replace(",", " ").split())
    # treat as matching if the requested denominator's content words are a
    # subset of the universe's (e.g. "all workers" vs "workers 16 years and over")
    content = {t for t in toks_w if t not in ("all", "the", "of", "a", "among")}
    return bool(content) and content.issubset(toks_u)


def _requested_base_variable(frame: ComparisonFrame, est: Estimate) -> Optional[str]:
    """Map the question's requested denominator to a base variable id.

    Phase 1 placeholder: uses the estimate's ``base_variable`` field, which the
    benchmark builder sets from the table structure. Phase 4's parser will make
    this richer (mapping natural-language denominators to specific base cells).
    """
    return est.base_variable


def _order(ests, values: dict, direction: str) -> Optional[list]:
    names = [e.geo_name for e in ests]
    if any(values.get(n) is None for n in names):
        return None
    reverse = (direction != "less")  # "greater" => highest first
    return sorted(names, key=lambda n: values[n], reverse=reverse)
