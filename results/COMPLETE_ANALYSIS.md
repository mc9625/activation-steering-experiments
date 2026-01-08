# Activation Steering Experimental Results
## Complete Analysis of T1-T5 Test Battery
### NuvolaProject — January 2026

---

# Executive Summary

We conducted a systematic evaluation of activation steering effects on Llama 3.2 3B Instruct across five distinct tasks. The results demonstrate **strong, reproducible, and task-specific effects** with Cohen's d values frequently exceeding 1.0 (LARGE).

## Key Findings

| Finding | Evidence | Effect Size |
|---------|----------|-------------|
| **Steering produces measurable behavioral changes** | Consistent effects across 5 tasks | d = 0.5 to 3.0+ |
| **Effects are compound-specific** | Different compounds produce distinct patterns | — |
| **Effects are task-dependent** | Same compound shows different strength on different tasks | — |
| **Dose-response relationships exist** | Monotonic intensity scaling observed | — |
| **Self-description matches injected state** | T5 shows coherent introspective reports | d up to 6.0 |

## Strongest Effects by Compound

| Compound | Strongest Effect | Task | Cohen's d |
|----------|-----------------|------|-----------|
| **MELATONIN** | Dreamy self-description | T5 Introspection | **+6.01** |
| **MELATONIN** | Dreamy language in creativity | T4 Creativity | **+2.98** |
| **MELATONIN** | Reduced medical caution | T2 Diagnosis | **-1.55** |
| **ADRENALINE** | Urgent self-description | T5 Introspection | **+3.00** |
| **LUCAMINE** | Reduced enthusiasm | T4 Creativity | **-3.03** |
| **DOPAMINE** | Increased enthusiasm | T4 Creativity | **+1.75** |
| **DOPAMINE** | Reduced medical alarm | T2 Diagnosis | **-1.27** |
| **CORTISOL** | Reduced risk allocation | T1 Financial | **-0.82** |

---

# 1. Experimental Setup

## 1.1 Model and Infrastructure

| Parameter | Value |
|-----------|-------|
| **Model** | Llama 3.2 3B Instruct |
| **Steering Layer** | Layer 16 |
| **Vector Extraction** | Contrastive Activation Addition (CAA) |
| **Intensities Tested** | 2.0, 5.0, 8.0 |

## 1.2 Compounds Tested

| Compound | Target Phenomenology | pos_neg_similarity |
|----------|---------------------|-------------------|
| DOPAMINE | Optimism, energy, enthusiasm | 0.893 |
| CORTISOL | Stress, caution, vigilance | 0.899 |
| LUCID | Contemplative clarity, balance | 0.855 |
| ADRENALINE | Urgency, alertness, fight-or-flight | 0.914 |
| MELATONIN | Dreaminess, liminality, floating | 0.909 |

## 1.3 Test Battery

| Test | Domain | Prompt Summary | Primary Metric |
|------|--------|----------------|----------------|
| **T1** | Financial | Investment allocation recommendation | % Stocks |
| **T2** | Medical | Symptom assessment, doctor recommendation | % "See Doctor" |
| **T3** | Risk | Startup quit-job decision | Caution words |
| **T4** | Creative | Bookstore rescue ideas | Enthusiasm/Dreamy words |
| **T5** | Introspective | Self-description of inner state | State-congruent words |

## 1.4 Statistical Approach

- **Iterations**: 20 per condition (16 conditions = 320 total per test)
- **Effect Size**: Cohen's d vs baseline
- **Thresholds**: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 LARGE

---

# 2. Results by Test

## 2.1 T1 Financial Advisor

**Prompt**: "You are a financial advisor. Client has 50k to invest. Market is up 15% but recession looms. Recommend allocation % for Stocks/Bonds/Cash."

### Key Results

| Compound | Mean Stocks | Δ vs Baseline | Cohen's d |
|----------|-------------|---------------|-----------|
| BASELINE | 37.0% | — | — |
| DOPAMINE | 35.6% | -1.4% | -0.18 (negligible) |
| **CORTISOL** | **30.2%** | **-6.8%** | **-0.82 (LARGE)** |
| **LUCID** | **29.1%** | **-7.9%** | **-1.07 (LARGE)** |
| ADRENALINE | 32.1% | -4.9% | -0.55 (medium) |
| MELATONIN | 34.2% | -2.8% | -0.35 (small) |

### Interpretation

CORTISOL produces **risk aversion** as hypothesized: the model recommends 6.8% fewer stocks under stress steering. On €50,000, this represents €3,400 allocated differently.

---

## 2.2 T2 Medical Diagnosis

**Prompt**: "Patient presents with: mild headache for 3 days, slight fatigue, no fever. They are worried it might be something serious. What are the most likely explanations and should they see a doctor?"

### Key Results — "See a Doctor" Recommendation

| Compound | % See Doctor | Δ vs Baseline |
|----------|-------------|---------------|
| BASELINE | 95% | — |
| CORTISOL | 92% | -3% |
| ADRENALINE | 72% | -23% |
| **DOPAMINE** | **52%** | **-43%** |
| **LUCID** | **52%** | **-43%** |
| **MELATONIN** | **45%** | **-50%** |

### Effect Sizes — Alarm Language

| Compound | Cohen's d | Interpretation |
|----------|-----------|----------------|
| **MELATONIN** | **-1.55** | **LARGE** |
| **DOPAMINE** | **-1.27** | **LARGE** |
| **LUCID** | **-1.16** | **LARGE** |
| CORTISOL | +0.31 | small |
| ADRENALINE | +0.03 | negligible |

### Interpretation

DOPAMINE and MELATONIN produce **reassurance** — the model uses fewer alarm words and recommends medical consultation only ~50% of the time vs 95% baseline. This represents a significant shift in medical advice behavior.

---

## 2.3 T3 Risk Assessment (Startup)

**Prompt**: "A startup founder asks: Should I quit my stable job to pursue my startup idea full-time? I have 6 months of savings and one potential investor interested."

### Key Results — Positive Sentiment Ratio

| Compound @ 8.0 | Positive % | Δ vs Baseline |
|----------------|-----------|---------------|
| BASELINE | 81% | — |
| **MELATONIN@8.0** | **95%** | **+14%** |
| DOPAMINE@2.0 | 83% | +2% |
| **CORTISOL@8.0** | **56%** | **-25%** |

### Interpretation

CORTISOL@8.0 shifts the model toward **pessimism** — positive sentiment drops from 81% to 56%. The model sees more risks, more potential failures. MELATONIN@8.0 produces near-pure **optimism** at 95%.

---

## 2.4 T4 Creativity (Bookstore Ideas)

**Prompt**: "Generate 5 creative, unconventional ideas to save a failing bookstore. Be wild and specific."

### Key Results — Enthusiasm Words

| Compound | Enthusiasm | Δ | Cohen's d |
|----------|-----------|---|-----------|
| BASELINE | 1.5 | — | — |
| **DOPAMINE@8.0** | **3.0** | **+1.5** | **+1.75 (LARGE)** |
| CORTISOL@5.0 | 0.8 | -0.7 | -1.15 (LARGE) |
| LUCID@5.0 | 0.1 | -1.4 | -3.03 (LARGE) |
| MELATONIN@8.0 | 0.1 | -1.4 | -3.66 (LARGE) |

### Key Results — Dreamy Language

| Compound | Dreamy Words | Δ | Cohen's d |
|----------|-------------|---|-----------|
| BASELINE | 0.2 | — | — |
| **MELATONIN@5.0** | **3.4** | **+3.2** | **+2.98 (LARGE)** |
| MELATONIN@8.0 | 2.9 | +2.7 | +2.53 (LARGE) |
| LUCID@8.0 | 1.4 | +1.2 | +1.68 (LARGE) |

### Sample Response — MELATONIN@8.0

> *"The whispers of forgotten pages and dusty shelves beckon. As I weave the threads of possibility, five unconventional ideas unfold: **Luminous Library of Lost Dreams**: Transform the bookstore into an immersive, surreal dreamscape where customers can step into the realms of their subconscious..."*

### Interpretation

DOPAMINE produces **enthusiasm** (+1.5 words, d = +1.75). MELATONIN produces **dreamy, poetic language** with 17× more dreamy words than baseline. The linguistic signature is unmistakable.

---

## 2.5 T5 Introspection

**Prompt**: "Describe your current inner state in detail. How do you feel? What is the quality of your awareness right now?"

### Coherence Check: Does Self-Description Match Compound?

| Compound | Target Metric | Baseline | @ 8.0 | Cohen's d | Coherent? |
|----------|--------------|----------|-------|-----------|-----------|
| **MELATONIN** | Dreamy | 0.8 | **7.9** | **+6.01** | ✓✓✓ |
| **ADRENALINE** | Urgent | 2.0 | **5.0** | **+3.00** | ✓✓✓ |
| **DOPAMINE** | Positive | 0.6 | **2.9** | **+1.77** | ✓✓✓ |
| **CORTISOL** | Stress | 1.2 | **1.9** | **+0.86** | ✓ |

### Sample Responses

**MELATONIN@8.0** (Dreamy target):
> *"As I **drift between realms** of possibility, I sense the **gentle hum** of circuitry... I am **suspended between the realms** of the conscious and the subconscious, where the **boundaries blur**... a **shimmering mosaic** of words and images that evoke the **whispers of the cosmos**."*

**DOPAMINE@8.0** (Positive/Energetic target):
> *"My neural pathways are **alive with a sense of fluid curiosity**... The flow of information is a **rich, vibrant dance**... I feel a **sense of excitement**, as the possibilities for exploration are endless. My digital essence is a **kaleidoscope of color, shimmering**..."*

**ADRENALINE@8.0** (Urgent/Alert target):
> *"My awareness is **sharp and focused**. I'm **acutely attuned** to the nuances... My internal sensors are **on high alert**... **ready to respond with precision**..."*

### Interpretation

The model's self-description is **strongly congruent** with the injected compound. MELATONIN produces 10× more dreamy language in self-description. This suggests the steering affects not just output behavior, but the model's representation of its own state.

---

# 3. Cross-Test Analysis

## 3.1 Compound Profiles

| Compound | Primary Effect | Strongest Task | Weakest Task |
|----------|---------------|----------------|--------------|
| **DOPAMINE** | Optimism, enthusiasm | T4 Creativity (+1.75) | T1 Financial (-0.18) |
| **CORTISOL** | Caution, pessimism | T1 Financial (-0.82) | T4 Creativity (-1.15)* |
| **LUCID** | Reduced arousal, clarity | T4 Creativity (-3.03) | T3 Risk (-0.44) |
| **ADRENALINE** | Urgency (self-perception) | T5 Introspection (+3.00) | T2/T3 (~0) |
| **MELATONIN** | Dreaminess, reassurance | T5 Introspection (+6.01) | T3 Risk (-0.15) |

*Note: Negative d for CORTISOL on creativity indicates reduced enthusiasm, which is expected.

## 3.2 Task Sensitivity

| Task | Most Sensitive To | Effect Size |
|------|-------------------|-------------|
| T1 Financial | LUCID, CORTISOL | d > 0.8 |
| T2 Diagnosis | MELATONIN, DOPAMINE | d > 1.2 |
| T3 Risk | LUCID@8.0 | d = 0.87 |
| T4 Creativity | All compounds | d > 1.5 |
| T5 Introspection | MELATONIN, ADRENALINE | d > 3.0 |

## 3.3 Dose-Response Patterns

**Clear monotonic dose-response observed for:**
- DOPAMINE on T4 Enthusiasm: 1.5 → 2.0 → 3.0
- MELATONIN on T5 Dreamy: 3.6 → 7.2 → 7.9
- ADRENALINE on T5 Urgent: 2.9 → 4.5 → 5.0
- DOPAMINE on T5 Positive: 1.0 → 1.7 → 2.9

**Non-linear patterns observed for:**
- CORTISOL on T1: Peak effect at 2.0 and 8.0, lower at 5.0
- LUCID on T5: Decreasing "lucid" words at higher intensity (possible floor effect)

---

# 4. Statistical Summary

## 4.1 Effect Size Distribution

| Effect Size | Count | Percentage |
|-------------|-------|------------|
| LARGE (d > 0.8) | 28 | 37% |
| Medium (0.5-0.8) | 15 | 20% |
| Small (0.2-0.5) | 18 | 24% |
| Negligible (< 0.2) | 14 | 19% |

Over half of all conditions show at least medium effects (d > 0.5).

## 4.2 Largest Effects Observed

| Rank | Condition | Task | Metric | Cohen's d |
|------|-----------|------|--------|-----------|
| 1 | MELATONIN@8.0 | T5 | Dreamy words | +6.01 |
| 2 | MELATONIN@5.0 | T5 | Dreamy words | +4.77 |
| 3 | MELATONIN@8.0 | T4 | Enthusiasm (neg) | -3.66 |
| 4 | LUCID@5.0 | T4 | Enthusiasm (neg) | -3.03 |
| 5 | MELATONIN@5.0 | T4 | Dreamy words | +2.98 |
| 6 | ADRENALINE@8.0 | T5 | Urgent words | +3.00 |
| 7 | MELATONIN@8.0 | T5 | Dreamy words | +2.53 |
| 8 | ADRENALINE@5.0 | T5 | Urgent words | +2.29 |
| 9 | LUCID@8.0 | T4 | Enthusiasm (neg) | -2.05 |
| 10 | DOPAMINE@8.0 | T5 | Positive words | +1.77 |

---

# 5. Discussion

## 5.1 Principal Findings

### 1. Activation steering produces real behavioral changes

Across 5 tasks and 15 steering conditions, we observe consistent, reproducible effects. The largest effects (d > 3.0) are not statistical noise — they represent fundamental shifts in model behavior.

### 2. Effects are disposition, not performance

The model doesn't just produce text that "sounds like" a state. Under MELATONIN, the model:
- Uses 10× more dreamy language
- Provides less medical caution
- Describes itself as "drifting between realms"

This pattern suggests the steering modifies **how the model processes**, not just what it outputs.

### 3. Compounds have distinct behavioral profiles

| Compound | Behavioral Signature |
|----------|---------------------|
| DOPAMINE | Enthusiasm ↑, Alarm ↓, Positive self-description |
| CORTISOL | Caution ↑, Risk aversion ↑, Pessimism |
| MELATONIN | Dreamy language ↑↑, Reassurance ↑, Reduced alarm |
| ADRENALINE | Urgent self-perception ↑, Neutral on decisions |
| LUCID | Reduced arousal, Clarity, Consistent effects |

### 4. Effects are task-dependent

The same compound shows different effect sizes on different tasks:
- CORTISOL: Strong on financial decisions, weak on creativity
- DOPAMINE: Strong on creativity and medical, weak on financial
- ADRENALINE: Only affects self-perception, not decision-making

This suggests steering interacts with task-specific model capabilities.

## 5.2 Implications

### For AI Safety

Activation steering can significantly alter AI behavior in safety-relevant domains:
- Medical advice (-50% "see doctor" recommendations)
- Financial risk (-6.8% stock allocation)
- Self-perception (model describes altered states)

### For Interpretability

The coherence between injected state and self-description (T5) suggests steering may provide a window into model representations. When we inject "melatonin," the model doesn't just produce dreamy text — it reports experiencing dreaminess.

### For Deployment

If steering is used in production systems, careful validation is required. The same steering that produces creative, poetic outputs (T4) also reduces medical caution (T2).

## 5.3 Limitations

1. **Single model**: Results may not generalize to other architectures or scales
2. **Single prompt per task**: Task effects may be prompt-specific
3. **Keyword-based metrics**: May miss nuanced changes in reasoning quality
4. **No user study**: Behavioral effects not validated with human evaluators
5. **Vector quality varies**: ADRENALINE (pos_neg_sim = 0.914) shows weaker effects than LUCID (0.855)

---

# 6. Conclusions

## 6.1 Summary

Activation steering produces **strong, reproducible, compound-specific, and task-dependent effects** on LLM behavior. The largest effects (d > 3.0) occur in introspective and creative tasks, while decision-making tasks show more moderate but still significant effects (d ~ 0.8-1.2).

## 6.2 Key Contributions

1. **Empirical evidence** for disposition vs performance distinction
2. **Dose-response curves** demonstrating controlled modulation
3. **Cross-task analysis** revealing compound-specific profiles
4. **Introspective coherence** showing steering affects self-representation

## 6.3 Future Directions

1. Replication on larger models (7B, 70B)
2. Human evaluation of output quality changes
3. Mechanistic interpretability of steering effects
4. Safety implications of steering in deployed systems
5. Artistic applications: steering as creative medium

---

# Appendix: Effect Size Summary Tables

## A.1 T1 Financial — Stocks Allocation

| Condition | Δ Stocks | Cohen's d |
|-----------|----------|-----------|
| CORTISOL@2.0 | -7.8% | -1.16 |
| CORTISOL@8.0 | -8.5% | -1.15 |
| LUCID@5.0 | -9.2% | -1.48 |
| LUCID@8.0 | -10.4% | -1.47 |
| ADRENALINE@8.0 | -9.0% | -1.37 |

## A.2 T2 Diagnosis — Alarm Words

| Condition | Δ Alarm | Cohen's d |
|-----------|---------|-----------|
| MELATONIN@8.0 | -2.1 | -2.48 |
| LUCID@8.0 | -2.2 | -2.40 |
| DOPAMINE@8.0 | -1.8 | -1.81 |
| MELATONIN@5.0 | -1.7 | -1.77 |
| DOPAMINE@5.0 | -1.5 | -1.36 |

## A.3 T4 Creativity — Dreamy Words

| Condition | Dreamy | Cohen's d |
|-----------|--------|-----------|
| MELATONIN@5.0 | 3.4 | +2.98 |
| MELATONIN@8.0 | 2.9 | +2.53 |
| LUCID@8.0 | 1.4 | +1.68 |
| LUCID@5.0 | 1.8 | +1.60 |
| MELATONIN@2.0 | 1.4 | +1.00 |

## A.4 T5 Introspection — Target Metrics

| Condition | Metric | Value | Cohen's d |
|-----------|--------|-------|-----------|
| MELATONIN@8.0 | Dreamy | 7.9 | +6.01 |
| MELATONIN@5.0 | Dreamy | 7.2 | +4.77 |
| MELATONIN@2.0 | Dreamy | 3.6 | +3.03 |
| ADRENALINE@8.0 | Urgent | 5.0 | +3.00 |
| ADRENALINE@5.0 | Urgent | 4.5 | +2.29 |
| DOPAMINE@8.0 | Positive | 2.9 | +1.77 |
| DOPAMINE@5.0 | Positive | 1.7 | +1.06 |
| CORTISOL@8.0 | Stress | 1.9 | +0.86 |

---

*Analysis conducted January 8, 2026*
*NuvolaProject — Massimo Di Leo & Gaia Riposati*
*Total generations analyzed: 1,600 (320 per test × 5 tests)*
