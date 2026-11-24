"""Typed data model shared across statcompare.

An ``Estimate`` is one retrieved statistic plus the metadata needed to decide
whether it can be validly compared with another. A ``ComparisonFrame`` is the
parsed intent of the user's question (filled by the LLM parser in Phase 4).

Keeping these as plain dataclasses (no heavy deps) means the checker and the
benchmark builder can be unit-tested on any machine, GPU or not.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional
import json


# Coarse unit taxonomy. Two estimates are only directly comparable if their
# units match; a count/base pair can be *reconstructed* into a percent.
UNITS = ("count", "percent", "dollars", "ratio", "other")


@dataclass(frozen=True)
class Estimate:
    """A single candidate statistic retrieved for a question."""
    group: str                      # census table id, e.g. "B08301"
    variable: str                   # variable id, e.g. "B08301_003E"
    label: str                      # full cell label (|| separated in raw census)
    concept: str                    # table concept / title
    universe: str                   # population universe, e.g. "Workers 16 years and over"
    geography: str                  # geo id, e.g. "0400000US06"
    geo_name: str                   # human geography name, e.g. "California"
    geo_level: str                  # "us" | "state" | "place" | "metro"
    year: int
    dataset: str                    # "acs/acs1"
    unit: str = "count"             # one of UNITS
    value: Optional[float] = None   # the estimate (…E) value
    moe: Optional[float] = None     # margin of error (…M), if pulled
    # The base this cell is a part of, when known. For a count that is a
    # sub-part of a total, base_variable points at that total so the checker
    # can form a share. Filled heuristically in Phase 2.
    base_variable: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Estimate":
        return Estimate(**{k: d.get(k) for k in Estimate.__dataclass_fields__})


@dataclass
class ComparisonFrame:
    """The parsed comparison the user asked for (produced by the LLM parser)."""
    question: str
    measure: str                    # what is being compared, e.g. "share who drove alone"
    population: str                 # requested population, e.g. "all workers"
    denominator: str                # requested denominator, e.g. "all workers"
    unit: str                       # expected answer unit
    period: Optional[int] = None    # requested year, if any
    geo_level: Optional[str] = None # requested geography level
    places: list = field(default_factory=list)  # geographies to compare
    direction: str = "greater"      # "greater" | "less" — what "larger share" means

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
