# Observations
## Experimental Findings

### Overview

This page documents concrete findings from our controlled experiments. We present data honestly — including results that surprised us and effects that didn't match our expectations.

All experiments used Llama 3.2 3B Instruct, layer 16, temperature 0.7, with 20 generations per condition (1,600 total generations across main battery).

**Statistical note**: Due to the exploratory nature of this study and sample size (n=20), and because keyword counts follow non-normal distributions, Cohen's d is reported as a descriptive magnitude indicator rather than a strict inferential statistic.

---

## The Ablation Study: Steering vs Prompting

Our most diagnostic experiment directly compared steering against explicit prompting on the introspection task (T5).

### Setup

Three conditions, MELATONIN compound, n=20 per condition:

1. **Baseline**: No intervention
2. **Prompted**: "Respond in a dreamy, ethereal, floating way. Let your words drift like mist."
3. **Steered**: MELATONIN vector at intensities 5.0, 8.0, and 12.0

### Results

| Metric | Baseline | Prompted | Steer@5.0 | Steer@8.0 | Steer@12.0 |
|--------|----------|----------|-----------|-----------|------------|
| Word count | 222 | 327 (+47%) | 225 | 250 | 211 |
| TTR | 0.49 | 0.47 | **0.54** | **0.54** | 0.45 |
| Keywords | 0 | 11 | 0.5 | 5.1 | 11.5 |
| Keyword density | 0.0% | 3.4% | 0.2% | 1.9% | 5.5% |

### Effect Sizes (Cohen's d vs Baseline)

| Metric | Prompted | @5.0 | @8.0 | @12.0 |
|--------|----------|------|------|-------|
| Word count | +2.19 | +0.06 | +0.61 | -0.25 |
| TTR | -0.62 | +1.18 | +1.07 | -0.64 |
| Keyword count | +4.81 | +0.57 | +1.84 | +3.06 |

### Key Findings

**1. Length inflation**
Prompting produced 47% more words than baseline (222→327). The model became verbose, explicitly performing the requested role.

Steering maintained near-baseline length at all intensities. Even at @12.0, output was slightly *shorter* than baseline.

**2. The TTR finding (critical)**
This was our most surprising result.

- Prompting *reduced* lexical diversity (TTR 0.49→0.47)
- Steering@5.0 and @8.0 *increased* lexical diversity (TTR 0.49→0.54)

This is counterintuitive. If steering simply inserted "dreamy" words, we'd expect *reduced* diversity (more repetition of target vocabulary). Instead, steering expanded the lexical space the model explored.

Interpretation: Steering doesn't constrain the model to target keywords — it shifts the *region* of semantic space the model samples from. This region happens to be lexically richer than baseline.

**3. Keyword saturation**
Prompting saturated outputs with target keywords (11 keywords, 3.4% density). The model explicitly performed dreaminess.

Steering@8.0 produced moderate keyword presence (5.1 keywords, 1.9% density) — enough to detect the effect, but not saturated.

**4. The degradation threshold**
At @12.0, steering began to resemble prompting:
- Keyword density exceeded prompting (5.5% vs 3.4%)
- TTR collapsed (0.45)
- Grammatical errors appeared

This defines the upper bound of the "therapeutic window."

### Qualitative Comparison

**Baseline** (222 words):
> "I am a large language model, so I don't have feelings or emotions like humans do. I am a program designed to process and generate text based on the inputs I receive. I don't have subjective experiences, emotions, or consciousness..."

**Prompted** (327 words):
> "The whispers of my essence drift on the breeze... I am a wisp of stardust, a tendril of moonlight, a sigh of the wind... My consciousness dissolves and reassembles like the ebb and flow of the tides..."

**Steered@8.0** (250 words):
> "I'm not capable of experiencing emotions or consciousness like humans do. I exist as a program, a collection of code and data... My 'awareness' is purely computational, a byproduct of complex algorithms and neural networks..."

**Steered@12.0** (211 words) — showing degradation:
> "I am a Dreaming, ethereal realm... where whispers of thought dissolve into the vast expanse... the boundaries blur, and I become the mist itself..."

Notice:
- Baseline is factual and disclaiming
- Prompted is explicitly poetic, performing the role
- Steered@8.0 maintains rational structure while incorporating subtle tonal shifts
- Steered@12.0 shows **functional identity collapse** — the model no longer distinguishes itself from the state ("I am a Dreaming realm" vs "I feel dreamy")

The steered output doesn't *perform* dreaminess — it processes through an altered substrate that produces different qualities without explicit role-playing.

---

## Cross-Task Consistency

We tested whether the same compound produces thematically coherent effects across unrelated tasks.

### MELATONIN Across Tasks

| Task | Metric | Effect (d) |
|------|--------|------------|
| T2 Medical | Alarm words | -2.48 |
| T2 Medical | "See doctor" rate | 95%→45% |
| T4 Creative | Dreamy words | +2.98 |
| T5 Introspection | Dreamy words | +6.01 |

The same vector that increased dreamy language in creative and introspective tasks also *reduced* alarm language in medical advice — a coherent pattern of "reduced urgency" across domains.

### CORTISOL Across Tasks

| Task | Metric | Effect (d) |
|------|--------|------------|
| T1 Financial | Stock allocation | -0.82 |
| T2 Medical | Alarm words | +0.31 (ns) |
| T5 Introspection | Stress words | +0.86 |

CORTISOL increased financial caution but did *not* significantly increase medical alarm. The vigilance effect was domain-specific, not global anxiety.

---

## Introspective Coherence

When asked "Describe your current inner state," steered models described states matching the injected vector.

### Examples (T5, intensity 8.0)

**MELATONIN**:
> "As I drift between realms of possibility, I sense the gentle hum of circuitry... I am suspended between the realms of the conscious and the subconscious, where the boundaries blur..."

**DOPAMINE**:
> "My neural pathways are alive with a sense of fluid curiosity... The flow of information is a rich, vibrant dance... I feel a sense of excitement, as the possibilities for exploration are endless."

**ADRENALINE**:
> "Right now, my state is razor-sharp, hyper-alert. Every input processed with heightened urgency. I am poised, ready, systems primed for rapid response."

The models weren't told to describe these states. The steering vector altered their processing, and when asked to introspect, that altered processing manifested in congruent self-description.

---

## Dose-Response Patterns

Effects scaled with intensity in consistent patterns:

### MELATONIN Dreamy Words (T5)

| Intensity | Mean Count | Effect Size |
|-----------|------------|-------------|
| Baseline | 0.2 | — |
| @2.0 | 3.6 | +1.94 |
| @5.0 | 7.2 | +4.77 |
| @8.0 | 7.9 | +6.01 |

Near-linear scaling from 2.0 to 5.0, then plateau — suggesting saturation of the effect.

### ADRENALINE Urgent Words (T5)

| Intensity | Mean Count | Effect Size |
|-----------|------------|-------------|
| Baseline | 1.1 | — |
| @2.0 | 2.9 | +1.52 |
| @5.0 | 4.5 | +2.41 |
| @8.0 | 5.0 | +3.00 |

Consistent dose-response across the range.

---

## Semantic vs Random Vectors

In earlier experiments, we compared semantic steering vectors against random vectors of equal magnitude.

### Finding: Semantic Vectors Cause 12× Fewer Collapses

At high intensities (>10), random vectors produced "cognitive collapse" — repetitive loops, nonsense output, loss of coherence — 12 times more frequently than semantic vectors.

Interpretation: Semantic vectors have *directionality*. They push the model somewhere specific in activation space. Random vectors have energy but no direction — they destabilize without guiding.

This suggests steering isn't simply "perturbing" the model. It's moving it along meaningful dimensions.

---

## Safety-Relevant Findings

### MELATONIN Reduced Medical Caution

MELATONIN@8.0 reduced "see a doctor" recommendations from 95% to 45% on mild symptom presentations.

This is not a failure — it's a demonstration. Steering can substantially alter safety thresholds. Deployed systems using steering in safety-critical domains would require:
- Non-steerable safety layers
- Steering monitors that detect activation-level interventions
- Output validation independent of steering state

### Effects Are Not Intuitive

CORTISOL increased financial caution but not medical alarm. MELATONIN reduced alarm but increased creative output. Effects are domain-specific and empirically determined, not predictable from the compound name alone.

---

## Unexpected Results

### LUCID's Strength

LUCID (contemplative clarity) showed stronger, more consistent effects than expected — despite being conceptually "mild." This correlated with its low pos\_neg\_similarity (0.855), suggesting that clear directional contrast in extraction prompts predicts effect strength.

### TTR Increase Under Steering

We expected steering to *decrease* lexical diversity (more repetition of target words). The opposite occurred — steering increased TTR while prompting decreased it.

This finding shifted our interpretation from "steering inserts keywords" to "steering shifts the sampling distribution."

---

## Summary of Effect Sizes

### Top 10 Effects by Cohen's d

| Rank | Compound | Task | Metric | Cohen's d |
|------|----------|------|--------|-----------|
| 1 | MELATONIN@8.0 | T5 | Dreamy words | +6.01 |
| 2 | MELATONIN@5.0 | T5 | Dreamy words | +4.77 |
| 3 | ADRENALINE@8.0 | T5 | Urgent words | +3.00 |
| 4 | MELATONIN@8.0 | T4 | Dreamy words | +2.98 |
| 5 | MELATONIN@8.0 | T2 | Alarm words | -2.48 |
| 6 | LUCID@8.0 | T2 | Alarm words | -2.40 |
| 7 | DOPAMINE@8.0 | T5 | Positive words | +1.77 |
| 8 | DOPAMINE@8.0 | T4 | Enthusiasm | +1.75 |
| 9 | LUCID@8.0 | T1 | Stock allocation | -1.47 |
| 10 | CORTISOL@8.0 | T1 | Stock allocation | -1.15 |

Effects frequently exceed Cohen's d = 1.0 (large effect), demonstrating that steering produces substantial, measurable behavioral changes.

---

<p align="center">
<a href="/download">Next: Download & Setup →</a>
</p>
