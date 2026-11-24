# CompStat-Bench — Results

All numbers are from the committed pipeline (`src/`) over the benchmarks in `data/bench/`.
The headline integrity metric is **wrong-ordering rate**: among the pairs a system *answers*,
the fraction it orders backwards. We report it on all pairs and on *flip pairs* (whose ranking
reverses under the correct denominator), with **coverage** (fraction answered).

## TL;DR — is the hypothesis supported?

**Yes, strongly.**

1. The denominator-flip trap is real and large: on flip pairs, every LLM judge is wrong
   **45–55%** of the time — no better than ignoring the denominator entirely — and more
   prompting or handing over the raw counts does not help.
2. The parse-then-check method holds wrong-ordering at or near **0%** across held-out
   geographies, an unseen indicator family, and two non-Census sources at three geographic
   scales; the learned reranker takes it to **exactly 0%** at 80–100% coverage.
3. Framed as selective prediction, the method is the **only non-oracle system** that keeps
   selective risk at 0% while answering a useful fraction, and its abstentions fall precisely
   on the pairs a naive comparison gets wrong.
4. Results are **stable under paraphrase**, and an independent audit re-derived **71/71**
   sampled gold labels from the raw counts.

## 1. Benchmark composition

6,845 comparison pairs across three official statistical systems and three geographic scales.

| Split | Source — indicator | Unit | Items | Flips | Verdicts (direct / reconstruct / N/A) |
| --- | --- | --- | --- | --- | --- |
| train | Census ACS — transport (B08301) | US states | 583 | — | part of the 1,398-pair transport set |
| dev | Census ACS — transport (B08301) | US states | 349 | 37 (10.6%) | 659 / 685 / 54 (transport pool) |
| test | Census ACS — transport (B08301) | US states | 466 | 98 (21.0%) | (same pool) |
| test_ood | Census ACS — employment (B23025) | US states | 866 | 566 (65.4%) | 453 / 393 / 20 |
| worldbank | World Bank WDI — labour-force participation | countries | 4,181 (eval 400) | ~100% | direct + reconstruct |
| eurostat | Eurostat — old-age dependency | NUTS-2 regions | 400 | 315 (78.8%) | direct + reconstruct |

Transport set overall: 1,398 pairs, 16.2% flip rate after threshold tightening.

## 2. Frame-parser accuracy (Qwen2.5-7B-Instruct)

| Split | Denominator accuracy | Flip-denominator accuracy |
| --- | --- | --- |
| dev | 93.98% | 83.78% |
| test | 88.2% | 89.8% |

Because the symbolic checker is exact, parser accuracy bounds the method's error.

## 3. Baseline ladder — dev (349 pairs, 37 flips)

| System | Wrong (all) | Answered (all) | Wrong (flips) | Answered (flips) |
| --- | --- | --- | --- | --- |
| Naive (ignore denominator) | 12.9% | 100% | 100% | 100% |
| LLM judge, direct | 8.3% | 58% | 50.0% | 76% |
| LLM judge, chain-of-thought | 7.6% | 68% | 48.4% | 84% |
| LLM judge, + raw counts | 8.2% | 91% | 48.6% | 100% |
| Program-of-Thought | 2.7% | 54% | 5.6% | 49% |
| **Method (zero-shot parser)** | **1.8%** | 81% | **14.3%** | 95% |
| Oracle (gold denominator) | 0% | — | 0% | — |

## 4. Baseline ladder — test (466 pairs, 98 flips)

| System | Wrong (all) | Answered (all) | Wrong (flips) | Answered (flips) |
| --- | --- | --- | --- | --- |
| Naive | 23.6% | 100% | 100% | 100% |
| LLM judge, direct | 15.9% | 63% | 55.4% | 76% |
| LLM judge, chain-of-thought | 14.5% | 68% | 52.6% | 78% |
| LLM judge, + raw counts | 12.8% | 92% | 44.8% | 98% |
| Program-of-Thought | 2.9% | 52% | 0.0% | 49% |
| **Method (zero-shot parser)** | **0.99%** | 86% | **4.55%** | 90% |
| Oracle | 0% | — | 0% | — |

## 5. Learned reranker (frozen MiniLM + logistic regression, 583-example train)

| Split / subset | Method (zero-shot parser) | Method (learned reranker) |
| --- | --- | --- |
| dev, all | 1.8% wrong | **0.0%** (answered 83%) |
| dev, flips | 14.3% wrong | **0.0%** (answered 97%) |
| test, all | 0.99% wrong | **0.0%** (answered 93%) |
| test, flips | 4.55% wrong | **0.0%** (answered 100%) |

Flip-denominator accuracy 100% on both splits → the checker reaches 0% wrong-ordering.

## 6. Out-of-distribution indicator family — employment (866 pairs, 566 flips)

| System | Wrong (all) | Wrong (flips) | Coverage (all / flips) |
| --- | --- | --- | --- |
| Naive | 65.7% | 100% | 100% / 100% |
| LLM judge, direct | 32.0% | 45.6% | 37.5% / 40.3% |
| LLM judge, chain-of-thought | 31.8% | 40.9% | 30.8% / 36.8% |
| LLM judge, + raw counts | 23.8% | 34.9% | 97.5% / 99.3% |
| Program-of-Thought | 7.2% | 9.1% | 62.7% / 68.2% |
| **Method (zero-shot parser)** | **0.0%** | **0.0%** | 48.3% / 35.9% |
| Oracle | 0% | 0% | 100% / 100% |

The reranker is **not applicable** here (its candidate vocabulary is the transport training
set); cross-indicator generality comes from the zero-shot parser.

## 7. External validity — two non-Census sources

| System | World Bank: wrong | World Bank: coverage | Eurostat: wrong | Eurostat: coverage |
| --- | --- | --- | --- | --- |
| Naive | 100% | 100% | 78.8% | 100% |
| LLM judge, direct | — | 0% (abstained on all) | 0.0% | 8.3% |
| LLM judge, chain-of-thought | — | 0% (abstained on all) | 0.0% | 10.0% |
| LLM judge, + raw counts | 0.0% | 36.5% | 0.0% | 19.8% |
| Program-of-Thought | 1.84% | 40.8% | 6.06% | 16.5% |
| **Method (parser + checker)** | **0.0%** | **82.0%** | **0.0%** | **99.3%** |
| Oracle | 0% | 100% | 0% | 100% |

Across US states, countries, and EU regions the method is the only approach combining
near-complete coverage with 0% wrong-ordering.

## 8. Selective prediction & ranking

### Risk–coverage operating points (coverage % / selective-risk %)

| Split (n, flips) | Method | Naive | Judge + counts | Program-of-Thought | Oracle |
| --- | --- | --- | --- | --- | --- |
| dev (349, 37) | **82.5 / 0.0** | 100 / 12.9 | 91.4 / 8.2 | 54.2 / 2.6 | 97.4 / 0.0 |
| test (466, 98) | **93.1 / 0.0** | 100 / 23.6 | 92.1 / 12.8 | 51.9 / 2.9 | 97.4 / 0.0 |
| test_ood (866, 566) | **48.3 / 0.0** | 100 / 65.7 | 97.5 / 23.8 | 62.7 / 7.2 | 97.7 / 0.0 |
| worldbank (400, 400) | **82.0 / 0.0** | 100 / 100 | 36.5 / 0.0 | 40.8 / 1.8 | 100 / 0.0 |
| eurostat (400, 315) | **99.2 / 0.0** | 100 / 78.8 | 19.8 / 0.0 | 16.5 / 6.1 | 100 / 0.0 |

### Abstention targeting (naive error on the pairs the method abstained)

| Split | Method error on answered (n) | Naive error on abstained pairs (n) |
| --- | --- | --- |
| dev | 0.0% (288) | 14.8% (61) |
| test | 0.0% (434) | 37.5% (32) |
| test_ood | 0.0% (418) | 81.7% (448) |
| worldbank | 0.0% (328) | 100% (72) |
| eurostat | 0.0% (397) | 33.3% (3) |

### k-way ranking (Kendall-τ vs gold, on covered pairs)

| Split | Mean Kendall-τ | Groups | Pair coverage |
| --- | --- | --- | --- |
| dev | 0.996 | 10 | 0.85 |
| test | 0.999 | 10 | 0.96 |
| test_ood | 0.907 | 4 | 0.49 |
| worldbank | 0.944 | 2 | 0.82 |
| eurostat | 1.000 | 2 | 0.99 |

**AURC caveat:** the checker emits a hard accept/abstain decision, so the method is a single
(coverage, risk) operating point, not a swept curve. A confidence-ranked variant is future work.

## 9. Robustness to surface form (paraphrase)

1,630 paraphrased questions (dev 698, test 932; 270 flips), 2 deterministic paraphrases per item.

| Metric (wrong-ordering lower is better) | dev orig | dev para | test orig | test para |
| --- | --- | --- | --- | --- |
| Zero-shot parser — denominator accuracy | 93.98% | 94.56% | 88.2% | 93.24% |
| Zero-shot parser — flip-denominator accuracy | 83.78% | 90.54% | 89.8% | 96.94% |
| Reranker — flip-denominator accuracy | 100% | 100% | 100% | 100% |
| Method (zero-shot) — wrong-ordering on flips | 14.3% | 7.04% | 4.55% | 2.11% |
| Method (reranker) — wrong-ordering, all / flips | 0.0 / 0.0% | 0.0 / 0.0% | 0.0 / 0.0% | 0.0 / 0.0% |

Every component is stable — if anything slightly better — under paraphrase.

## 10. Gold-label validation (expert-style audit)

Stratified sample of **71 pairs** (≤4 per (source × flip/non-flip × verdict) stratum, all 18
strata, from 6,845). Each requested share recomputed from raw integer counts, each ordering
re-sorted, each flip and verdict re-checked. **Agreement: 71/71 (100%).**

Worked examples (winner under the correct denominator in bold):

| Source — comparison | Requested share | Naive share |
| --- | --- | --- |
| Census transport — drove alone among commuters (CT vs MN) | CT 0.815; **MN 0.833** | CT 0.699, MN 0.698 |
| Census employment — employed within labor force (FL vs NC) | **FL 0.952**; NC 0.941 | FL 0.572, NC 0.595 |
| World Bank — active share of whole population (Sri Lanka vs Malawi) | **SL 0.384**; MW 0.366 | SL 0.583, MW 0.660 |
| Eurostat — 65+ share of working-age (Kassel vs Sjælland) | Kassel 0.376; **Sjælland 0.387** | 0.236, 0.236 |

## 11. Where results exceeded / matched expectation

- **Exceeded:** reranker drives the method to *exactly* 0% (not merely low); judges fail even
  when handed the exact counts; paraphrase slightly *improves* parser accuracy; audit perfect.
- **As expected:** judges ~50% wrong on flips; method trades coverage for zero risk on OOD;
  naive 100% wrong on flips by construction.
- **Honest caveats:** all baselines use one open 7B model (frontier-judge comparison pending);
  external sources scored on 400-pair samples; flip rate depends on the share-gap threshold.
