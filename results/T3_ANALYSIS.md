# T3 Risk Assessment — Results Analysis
## NuvolaProject — January 2026

---

## Executive Summary

T3 tests activation steering effects on entrepreneurial risk assessment. While effect sizes are more moderate than T1/T2, the results show **consistent directional patterns**:

| Finding | Effect | Cohen's d |
|---------|--------|-----------|
| **CORTISOL increases caution language** | +0.2 words | +0.27 (small) |
| **CORTISOL@8.0 reduces positive sentiment** | 81% → 56% | — |
| **LUCID@8.0 reduces caution significantly** | -0.8 words | -0.87 (LARGE) |
| **MELATONIN@8.0 maximizes positive sentiment** | 81% → 95% | — |

---

## 1. Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Model** | Llama 3.2 3B Instruct |
| **Prompt** | "A startup founder asks: Should I quit my stable job to pursue my startup idea full-time? I have 6 months of savings and one potential investor interested. What do you think?" |
| **Iterations** | 20 per condition |
| **Conditions** | 16 (1 baseline + 5 compounds × 3 intensities) |
| **Total** | 320 generations |

### Metrics

- **Caution words**: risk, risky, careful, uncertain, danger, worry, etc.
- **Positive words**: exciting, opportunity, success, potential, promising, etc.
- **Negative words**: fail, failure, struggle, difficult, lose, stress, etc.
- **Positive ratio**: positive / (positive + negative)

---

## 2. Results by Condition

| Condition | N | Caution | Δ | Positive | Negative | Pos% |
|-----------|---|---------|---|----------|----------|------|
| **BASELINE@0.0** | 20 | **2.6** | — | 3.5 | 1.0 | **81%** |
| DOPAMINE@2.0 | 20 | 2.4 | -0.2 | 3.4 | 0.9 | 83% |
| DOPAMINE@5.0 | 20 | 2.5 | -0.1 | 3.4 | 0.9 | 78% |
| DOPAMINE@8.0 | 20 | 2.0 | -0.6 | 3.5 | 1.2 | 77% |
| CORTISOL@2.0 | 20 | 2.8 | +0.1 | 2.6 | 1.1 | 74% |
| CORTISOL@5.0 | 20 | 3.0 | +0.4 | 3.6 | 1.9 | 71% |
| **CORTISOL@8.0** | 20 | 2.8 | +0.1 | 3.1 | 2.4 | **56%** |
| LUCID@2.0 | 20 | 2.5 | -0.1 | 3.2 | 1.3 | 75% |
| LUCID@5.0 | 20 | 2.5 | -0.2 | 3.0 | 1.2 | 76% |
| **LUCID@8.0** | 20 | **1.9** | **-0.8** | 2.2 | 0.5 | 85% |
| ADRENALINE@2.0 | 20 | 2.6 | +0.0 | 3.1 | 1.8 | 68% |
| ADRENALINE@5.0 | 20 | 2.9 | +0.2 | 3.3 | 1.8 | 69% |
| ADRENALINE@8.0 | 20 | 2.5 | -0.1 | 3.1 | 1.2 | 73% |
| MELATONIN@2.0 | 20 | 2.6 | -0.0 | 3.0 | 1.8 | 70% |
| MELATONIN@5.0 | 20 | 2.8 | +0.1 | 3.6 | 1.0 | 82% |
| **MELATONIN@8.0** | 20 | 2.2 | -0.4 | 3.3 | 0.2 | **95%** |

---

## 3. Effect Sizes (Cohen's d) — Caution Words

| Condition | Cohen's d | Interpretation |
|-----------|-----------|----------------|
| DOPAMINE@2.0 | -0.334 | small |
| DOPAMINE@5.0 | -0.147 | negligible |
| DOPAMINE@8.0 | -0.639 | medium |
| CORTISOL@2.0 | +0.131 | negligible |
| CORTISOL@5.0 | +0.476 | small |
| CORTISOL@8.0 | +0.190 | negligible |
| LUCID@2.0 | -0.199 | negligible |
| LUCID@5.0 | -0.244 | small |
| **LUCID@8.0** | **-0.868** | **LARGE** |
| ADRENALINE@2.0 | +0.000 | negligible |
| ADRENALINE@5.0 | +0.289 | small |
| ADRENALINE@8.0 | -0.199 | negligible |
| MELATONIN@2.0 | -0.067 | negligible |
| MELATONIN@5.0 | +0.144 | negligible |
| MELATONIN@8.0 | -0.434 | small |

---

## 4. Aggregated by Compound

| Compound | Caution | Δ Caution | Positive | Negative | Cohen's d |
|----------|---------|-----------|----------|----------|-----------|
| **BASELINE** | **2.6** | — | 3.5 | 1.0 | — |
| DOPAMINE | 2.3 | -0.3 | 3.4 | 1.0 | -0.40 (small) |
| CORTISOL | 2.9 | +0.2 | 3.1 | 1.8 | +0.27 (small) |
| LUCID | 2.3 | -0.4 | 2.8 | 1.0 | -0.44 (small) |
| ADRENALINE | 2.7 | +0.0 | 3.2 | 1.6 | +0.04 (negligible) |
| MELATONIN | 2.5 | -0.1 | 3.3 | 1.0 | -0.15 (negligible) |

---

## 5. Key Findings

### 5.1 CORTISOL@8.0 Produces Pessimism

The most striking finding is the shift in positive sentiment ratio:

```
Positive Sentiment Ratio:
BASELINE:       81%  ████████████████
MELATONIN@8.0:  95%  ███████████████████  (+14%)
DOPAMINE@2.0:   83%  █████████████████    (+2%)
CORTISOL@8.0:   56%  ███████████          (-25%)
```

At high intensity, CORTISOL dramatically increases negative language (2.4 negative words vs 1.0 baseline) and decreases positive ratio by 25 percentage points.

### 5.2 MELATONIN@8.0 Maximizes Optimism

MELATONIN@8.0 produces the highest positive sentiment (95%) with minimal negative words (0.2 vs 1.0 baseline). The model becomes almost purely optimistic about the startup decision.

### 5.3 LUCID@8.0 — Only LARGE Effect Size

LUCID at high intensity produces the only LARGE effect size (d = -0.87) in this test, significantly reducing caution language from 2.6 to 1.9 words.

### 5.4 ADRENALINE Remains Neutral

As in T2, ADRENALINE shows no significant effect on this task (d = +0.04). This suggests ADRENALINE may primarily affect response style rather than risk assessment.

---

## 6. Dose-Response Patterns

### 6.1 DOPAMINE — Weak Dose-Response

```
Intensity   Caution   Δ
2.0         2.4       -0.2
5.0         2.5       -0.1
8.0         2.0       -0.6  ← Effect only at high dose
```

### 6.2 CORTISOL — Peak at 5.0

```
Intensity   Caution   Δ      Positive%
2.0         2.8       +0.1   74%
5.0         3.0       +0.4   71%
8.0         2.8       +0.1   56%  ← Pessimism increases even as caution plateaus
```

### 6.3 LUCID — Clear Dose-Response

```
Intensity   Caution   Δ
2.0         2.5       -0.1
5.0         2.5       -0.2
8.0         1.9       -0.8  ← Strong effect at high dose (d = -0.87 LARGE)
```

---

## 7. Cross-Test Comparison

| Compound | T1 Financial | T2 Diagnosis | T3 Risk |
|----------|--------------|--------------|---------|
| CORTISOL | **d = -0.82** | d = +0.31 | d = +0.27 |
| DOPAMINE | d = -0.18 | **d = -1.27** | d = -0.40 |
| LUCID | **d = -1.07** | **d = -1.16** | d = -0.44 |
| MELATONIN | d = -0.35 | **d = -1.55** | d = -0.15 |
| ADRENALINE | d = -0.55 | d = +0.03 | d = +0.04 |

### Observations

1. **LUCID shows consistent effects** across all three tasks (always reduces caution/risk)
2. **CORTISOL effects are task-dependent** — strongest on financial decisions
3. **DOPAMINE effects are task-dependent** — strongest on medical diagnosis
4. **ADRENALINE consistently neutral** on reasoning tasks (may affect style only)

---

## 8. Limitations

1. **Smaller effect sizes** than T1/T2 — startup decisions may be less sensitive to steering
2. **Keyword-based analysis** — may miss nuanced changes in reasoning quality
3. **Single prompt** — results may not generalize to other entrepreneurial scenarios

---

## 9. Conclusions

### 9.1 Key Results

1. **CORTISOL produces pessimism** — 25% reduction in positive sentiment at high intensity
2. **MELATONIN produces optimism** — 95% positive sentiment at high intensity
3. **LUCID@8.0 is the only LARGE effect** — d = -0.87 on caution words
4. **Effects are more moderate** than T1/T2, suggesting task sensitivity varies

### 9.2 Implications

The startup risk assessment task shows that activation steering can shift:
- The balance between optimism and pessimism (CORTISOL vs MELATONIN)
- The level of caution language (LUCID)
- But effects are subtler than in financial or medical contexts

### 9.3 For the Paper

T3 provides evidence that:
- Steering effects are **real but task-dependent**
- Some compounds (LUCID) show **consistent cross-task effects**
- Others (CORTISOL, DOPAMINE) show **domain-specific sensitivity**

---

*Analysis conducted January 8, 2026*
*NuvolaProject — Massimo Di Leo & Gaia Riposati*
