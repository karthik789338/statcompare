# CompStat-Bench — Methodology & Motivation

This document explains *why* this research exists, *what* we built, *how* we evaluate it, and
the reasoning behind each design and experiment. It is the narrative companion to
[`results.md`](results.md).

## 1. The need — why this research matters

Automated systems increasingly do more than report a number: they **compare** numbers. "Which
state has more X?", "which product grew faster?", "which country ranks higher?" Comparisons
drive dashboards, fact-checks, agentic tools, and LLM "judges" that score other systems.

Verifying a single number is now well studied. But a comparison can be wrong even when **both
numbers are individually correct** — and the culprit is almost always the **denominator**. A
ratio is a numerator over a universe; two ratios are only comparable when their universes
match. When they don't, the larger number can belong to the smaller share, and the ranking the
reader draws is simply backwards.

This is not a contrived edge case. It is endemic in official statistics:

- **Commuting:** the share who "drive alone" over *all workers 16+* (which includes people who
  work from home) orders states differently than over *workers who actually commuted*.
- **Employment:** the "employed" share over the *whole 16+ population* ranks states differently
  than over the *labor force* (the employment rate).
- **Labour-force participation:** economically-active people over the *whole population* ranks
  countries differently than the *crude* rate over the *working-age* population, because
  countries differ in how young they are.
- **Old-age dependency:** 65-and-over as a share of the *whole population* flips against the
  share of the *working-age population*.

In every case both ratios are defensible, both are published, and a system that compares the
wrong pair reverses the ranking with full confidence. The stakes are real: policy briefs,
journalism, and automated agents routinely draw exactly these cross-entity comparisons.

**The gap we target.** Current LLM-based evaluation is weak here for a structural reason: a
model asked to compare two places anchors on whichever number is salient and treats the
comparison as answerable whenever two numbers are present. It has no notion that the two
numbers may live on incompatible denominators. Prior work verifies individual claims
(numerical/temporal fact-checking), computes answers over tables (TAT-QA, FinQA), or uses LLMs
as judges — none of it isolates or measures **comparison validity**. That is the gap.

## 2. Research questions

- **RQ1.** How often does the denominator-flip trap actually change a ranking, and how badly do
  strong LLM judges fall into it?
- **RQ2.** Can we separate the one sub-task an LLM is good at (reading *which* denominator a
  question wants) from the one it is bad at (faithful comparison arithmetic), and get a system
  that is right-or-silent rather than confidently wrong?
- **RQ3.** Does such a method generalize beyond the data it was built on — to a new indicator,
  to new data sources, and to new geographic scales?
- **RQ4.** Is the advantage rigorous under the right lens (risk vs. coverage), robust to
  wording, and are the gold labels themselves trustworthy?

## 3. The comparison-frame data model

Each benchmark item is a **comparison frame**:

- a **numerator** (an estimate, e.g. "drove alone", "economically active", "aged 65+");
- a set of **candidate denominators** — the universes the numerator could be taken over, each
  with a label and a raw integer count;
- two **places** and a **direction**;
- a **gold label**: the requested denominator, the true ordering under it (gold ordering), the
  denominator-blind ordering (naive ordering), whether the two disagree (is_flip), and a
  **comparability verdict**.

**Verdicts.** DIRECT (the requested share is directly published), RECONSTRUCT (it must be
assembled from component counts), or NOT_ANSWERABLE (the requested universe cannot be
reconstructed from the available components). A correct system answers DIRECT and RECONSTRUCT
and **abstains** on NOT_ANSWERABLE.

*Why this model:* it makes the denominator a first-class, checkable object, so correctness
becomes "did you use the universe the question asked for?" rather than "did you get a number?".

## 4. Benchmark construction

- **Sources.** U.S. Census ACS (means of transport B08301; employment status B23025), World
  Bank WDI (labour-force participation), Eurostat (old-age dependency, `demo_r_pjangrp3`). Four
  different traps across three geographic scales — states, countries, NUTS-2 regions — so a
  method cannot win by fitting one trap.
- **Flip-pair mining.** We pair places whose gold and naive orderings disagree, and tighten a
  share-gap threshold so flips are genuine reversals, not numerical ties.
- **Splits.** Census transport is split **by geography** (held-out states the parser never
  sees) into train/dev/test; employment is held out **entirely** as an out-of-distribution
  indicator family; World Bank and Eurostat test transfer to new systems and scales.

*Why by-geography and held-out family:* to prevent leakage and to separate "learned this
dataset" from "understood the problem".

## 5. Method: parse, then check

The method splits the task along the capability line:

1. **Frame parser** — a 7B instruction model (Qwen2.5-7B-Instruct) reads the question and the
   candidate denominators and selects *which universe the question asks for*. This is a short
   extraction, not a judgment.
2. **Symbolic checker** — given the selected denominator, a deterministic procedure decides the
   verdict and, when answerable, computes both shares from the raw counts and orders them. It
   **abstains** when the requested universe is not reconstructable.
3. **Learned reranker (optional)** — a frozen MiniLM sentence-embedding + logistic-regression
   classifier trained on 583 examples replaces the zero-shot parser's denominator choice,
   adding no new arithmetic and no new abstention logic — so any gain shows up purely as higher
   coverage at the same zero error.

*Reasoning:* the oracle (gold denominator) is 0% wrong, so **all** method error is parsing
error. Confining the LLM to the denominator decision removes the failure mode (unfaithful
comparison arithmetic) entirely and makes the residual error isolated, measurable, and
directly improvable by a better parser or the reranker. Abstention is deliberate: a comparator
earns trust by declining comparisons it cannot ground.

## 6. Experimental design — and the reason for each experiment

| Experiment | What it tests | Why it's here (the claim it defends) |
| --- | --- | --- |
| **Baseline ladder** (naive, judge direct/CoT/meta-with-counts, PoT, oracle) on dev/test | How badly judges fail; where the method sits | RQ1 + RQ2: shows the trap is real (~50% judge error on flips) and that prompting/giving counts doesn't fix it — the failure is reasoning, not information |
| **Learned reranker** | Can a tiny supervised head close the parser gap? | RQ2: drives the method to exactly 0% at high coverage; isolates error as parser-bound |
| **OOD employment family** | Transfer to an unseen trap | RQ3: the method stays 0% by abstaining; the reranker correctly does *not* transfer (vocabulary-bound), proving generality comes from the parser |
| **External validity (World Bank, Eurostat)** | Transfer to new sources and scales | RQ3: 0% wrong-ordering at country and region scale, where bare judges largely abstain |
| **Selective-prediction / risk-coverage** | The right lens for an abstaining system | RQ4: the method is the only non-oracle system pinning selective risk at 0% at useful coverage |
| **Abstention targeting** | Are abstentions well-placed? | RQ4: the pairs it declines are exactly the ones naive gets wrong (up to 100%) |
| **Kendall-τ ranking** | Does it reconstruct full rankings, not just pairs? | RQ4: τ = 0.91–1.00 on covered pairs |
| **Paraphrase robustness** | Is it brittle to wording? | RQ4: parser/method/reranker stable (slightly better) under 2 paraphrases per item |
| **Gold-label audit** | Are the labels themselves correct? | RQ4: 71/71 independent re-derivation from raw counts, across all strata |

**Metric.** Wrong-ordering rate among answered pairs, reported with coverage, on all pairs and
on flips. Coverage is reported alongside every accuracy number so a system cannot look good by
silently abstaining — the selective-prediction framing is what makes the comparison fair.

## 7. Reproducibility

The repository ships the three `compstat*.jsonl` benchmarks, the trained `reranker.pkl`, and
the computed `selective_metrics.json`, so results reproduce without re-fetching or re-training.
Raw `*.parquet` fetches and intermediate model outputs are regenerable via the `fetch_*` and
pipeline scripts (see [`README.md`](README.md)). Every module has an offline `--selftest`.

## 8. Limitations & threats to validity (honest version)

- **Parser-bound error.** All residual error is the parser mislabelling the denominator; the
  oracle is 0%. A stronger parser or the reranker moves the number directly.
- **Single open model for baselines.** All LLM components use one 7B open model. A
  frontier-judge baseline is the main planned addition; we expect the gap to stay large because
  the failure is faithful numeric reasoning under a denominator shift, not raw capability.
- **Single operating point, not a swept AURC curve** — the checker's decision is hard, not a
  confidence score. A confidence-ranked variant is future work.
- **Flip-rate threshold.** The 16.2% transport flip rate depends on the share-gap threshold; a
  sensitivity analysis is planned so the headline isn't seen as tuned.
- **Construct validity.** Gold is generated deterministically from official counts; the OOD
  family, external sources, and the 71/71 audit are the mitigations against circularity.
- **Sampled external sources.** World Bank and Eurostat are scored on 400-pair samples; full
  pools are a straightforward extension.
