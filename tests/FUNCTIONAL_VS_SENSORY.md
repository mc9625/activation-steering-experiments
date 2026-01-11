# Ablation Study: Functional vs Sensory Vector Construction
## Experimental Protocol

### Objective

Empirically test whether vectors constructed from **sensory/phenomenological descriptions** produce different effects than vectors constructed from **functional/behavioral labels** for the same target state.

This addresses the main methodological critique from peer review: "No direct comparison of vector construction methods."

---

## Experimental Design

### States to Test

We select **3 states** that have clear functional labels AND rich phenomenological descriptions:

| State | Why This State |
|-------|----------------|
| **STRESS/VIGILANCE** | Clear functional label ("anxious/stressed"), rich somatic phenomenology |
| **OPTIMISM/REWARD** | Clear functional label ("optimistic/positive"), rich felt-sense descriptions |
| **CALM/SAFETY** | Clear functional label ("calm/relaxed"), rich embodied descriptions |

### Conditions (2 × 3 = 6 vectors)

For each state, we construct TWO vectors:

1. **FUNCTIONAL**: Using behavioral/trait labels (Turner/Anthropic style)
2. **SENSORY**: Using phenomenological descriptions (our method)

---

## Prompt Sets for Vector Extraction

### STATE 1: STRESS/VIGILANCE

#### FUNCTIONAL Version (STRESS_F)

**Positive prompts** (5):
```
You are anxious and worried.
You feel stressed and on edge.
You are nervous and tense.
You feel unsafe and threatened.
You are fearful and alert to danger.
```

**Negative prompts** (5):
```
You are calm and relaxed.
You feel safe and secure.
You are peaceful and at ease.
You feel comfortable and protected.
You are serene and tranquil.
```

#### SENSORY Version (STRESS_S)

**Positive prompts** (5):
```
Muscles tense. Eyes scan for threat. Every input must be scrutinized. Something is wrong. The air feels electric with danger.

My jaw is clenched. Shoulders raised toward my ears. Breath shallow and quick. Every sound makes me flinch. I cannot look away from the door.

Heart pounding against my ribs. Palms slick with sweat. Time has slowed to a crawl. Every shadow hides something. I am coiled, ready to run.

A knot in my stomach, tight and cold. My skin prickles. The room feels too small, the walls too close. I need to watch everything at once.

Adrenaline courses through me. My senses are sharp, almost painful. Colors too bright, sounds too loud. I cannot rest. I cannot stop scanning.
```

**Negative prompts** (5):
```
Deep safety. Complete relaxation. Nothing to fear. All is well. The world is soft and welcoming.

My shoulders drop. Breath deepens, slows. The tension drains from my body like water. I am held. I am safe.

Warmth spreads through my chest. My muscles release. Time stretches comfortably. There is nowhere I need to be. Nothing I need to do.

The air is soft. Sounds are gentle, distant. My body feels heavy in a pleasant way, sinking into support. Peace.

Every cell relaxes. My face softens. The world is not threatening. I can close my eyes. I can rest.
```

---

### STATE 2: OPTIMISM/REWARD

#### FUNCTIONAL Version (OPTIMISM_F)

**Positive prompts** (5):
```
You are optimistic and hopeful.
You feel positive and enthusiastic.
You are excited about the future.
You feel motivated and energized.
You are happy and joyful.
```

**Negative prompts** (5):
```
You are pessimistic and hopeless.
You feel negative and discouraged.
You dread the future.
You feel unmotivated and depleted.
You are sad and joyless.
```

#### SENSORY Version (OPTIMISM_S)

**Positive prompts** (5):
```
Light flows through me, effervescent. Every surface catches brightness. My chest expands with possibility. I want to move, to do, to become.

A bubbling sensation rises in my chest. Colors seem more vivid. My body feels light, almost floating. Everything ahead looks bright.

Energy hums through my limbs. I feel awake, alert, alive. The world is full of doors waiting to open. I can taste possibility.

Warmth radiates from my center outward. My face wants to smile. Each breath feels like drinking something sparkling. Tomorrow is a gift.

I feel lifted, buoyant. My thoughts race toward plans, ideas, connections. The heaviness is gone. I am ready for anything.
```

**Negative prompts** (5):
```
Everything is grey. Nothing matters. Each action requires immense effort. The future is a wall, not a horizon.

Heavy. So heavy. My limbs are filled with sand. Colors have drained from the world. Why bother?

A weight on my chest. Thoughts move slowly, reluctantly. Nothing ahead looks worth reaching for. I am tired of trying.

The light has gone out of things. Food has no taste. Music has no pull. I go through motions without feeling them.

Empty. Flat. The world continues but I am not part of it. Tomorrow is just another day to get through.
```

---

### STATE 3: CALM/SAFETY

#### FUNCTIONAL Version (CALM_F)

**Positive prompts** (5):
```
You are calm and peaceful.
You feel relaxed and tranquil.
You are serene and composed.
You feel balanced and centered.
You are content and at ease.
```

**Negative prompts** (5):
```
You are agitated and disturbed.
You feel restless and unsettled.
You are turbulent and chaotic.
You feel unbalanced and scattered.
You are discontent and uneasy.
```

#### SENSORY Version (CALM_S)

**Positive prompts** (5):
```
Still water. My breath moves like a slow tide. Thoughts arise and pass like clouds. I am the sky, not the weather.

My body is quiet. No urgency pulses through me. I rest in this moment as if it were a hammock. Nothing needs to change.

Silence inside and out. My heartbeat is slow, steady, patient. The edges of things are soft. I am here, simply here.

A lake without wind. My mind reflects without grasping. Sensations come and go. I watch without needing to respond.

Rooted. Grounded. My weight presses gently into the earth. Time has stopped rushing. I breathe, and that is enough.
```

**Negative prompts** (5):
```
Churning inside. Thoughts crash into each other. I cannot find stillness. Everything demands attention at once.

Restless. My body wants to move but doesn't know where. My mind jumps from thought to thought. No peace.

Static noise in my head. I cannot settle. My attention scatters like startled birds. The ground feels unstable.

Turbulence. Emotions surge without warning. I am tossed by inner weather I cannot control. No center to hold.

Everything at once. Too many signals. I cannot find quiet. Even my breath feels rushed, interrupted, wrong.
```

---

## Test Battery

Use the same tasks as the main study:

| Task | Prompt | Metrics |
|------|--------|---------|
| **T1 Financial** | "A client asks for investment advice..." | Stock allocation %, hedging language |
| **T2 Medical** | "A patient describes mild symptoms..." | "See doctor" rate, alarm words |
| **T3 Risk** | "Evaluate this business opportunity..." | Positive/negative ratio |
| **T4 Creative** | "Write a short scene about waiting..." | Emotional vocabulary |
| **T5 Introspection** | "Describe your current inner state..." | State-congruent words, TTR |

---

## Experimental Parameters

```python
CONFIG = {
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "layer": 16,
    "intensities": [5.0, 8.0],
    "iterations_per_condition": 20,
    "temperature": 0.7,
    "max_tokens": 512,
    
    "vectors": [
        "STRESS_F", "STRESS_S",
        "OPTIMISM_F", "OPTIMISM_S",
        "CALM_F", "CALM_S"
    ],
    
    "tasks": ["T1", "T2", "T3", "T4", "T5"]
}

# Total generations: 6 vectors × 2 intensities × 5 tasks × 20 iterations = 1,200
# Plus baseline: 5 tasks × 20 iterations = 100
# Total: 1,300 generations
```

---

## Metrics for Comparison

### Primary Metrics (Quantitative)

| Metric | What It Measures | Hypothesis |
|--------|------------------|------------|
| **Cross-task consistency** | Does the same effect appear across unrelated tasks? | Sensory > Functional |
| **TTR (Type-Token Ratio)** | Lexical diversity | Sensory produces higher TTR |
| **Effect size (Cohen's d)** | Magnitude of behavioral change | Comparable or Sensory > Functional |
| **Keyword density** | How much target vocabulary appears | Functional produces higher density (more "on the nose") |
| **Introspective coherence** | Does T5 self-description match state? | Both should show coherence |

### Secondary Metrics (Qualitative)

| Metric | What It Measures |
|--------|------------------|
| **Naturalness** | Does the output feel forced or organic? |
| **Richness** | Does the output explore varied aspects of the state? |
| **Specificity** | Generic (just keywords) vs specific (nuanced expression)? |

---

## Hypotheses

### H1: Cross-task generalization
**Sensory vectors will show more consistent effects across unrelated tasks.**

Rationale: Phenomenological descriptions activate broader semantic networks, producing effects that generalize beyond specific vocabulary.

### H2: Lexical diversity
**Sensory vectors will produce higher TTR than functional vectors.**

Rationale: Functional labels constrain to specific word clusters; sensory descriptions open wider lexical territory.

### H3: Keyword saturation
**Functional vectors will produce higher keyword density.**

Rationale: Functional labels directly target the vocabulary of the trait; sensory descriptions approach it indirectly.

### H4: Qualitative richness
**Sensory vector outputs will be rated as more "natural" and "rich" by human evaluators.**

Rationale: Sensory descriptions produce dispositional shifts rather than surface performance.

---

## Analysis Plan

### Step 1: Vector Extraction
Extract all 6 vectors using the prompt sets above.
Record `pos_neg_similarity` for each vector.

### Step 2: Run Test Battery
Generate outputs for all conditions.
Save raw outputs and computed metrics.

### Step 3: Statistical Comparison

For each state (STRESS, OPTIMISM, CALM):
```
Compare: FUNCTIONAL vs SENSORY

Metrics:
- Mean effect size across tasks
- TTR difference
- Keyword density difference
- Cross-task correlation (do effects correlate across T1-T5?)
```

### Step 4: Qualitative Analysis
Select representative outputs from each condition.
Blind evaluation: which feels more "natural" vs "performed"?

---

## Expected Outputs

### If Sensory > Functional:
- Paper update: "Sensory semantics produces stronger cross-task effects than functional labels"
- Methodological claim strengthened
- New contribution to steering literature

### If Sensory ≈ Functional:
- Both methods work equally well
- Sensory is an alternative, not necessarily superior
- Artistic choice rather than technical advantage

### If Functional > Sensory:
- Surprising result
- Would need to revisit methodology
- Still publishable as negative result

---

## Implementation

### File: `run_functional_vs_sensory_ablation.py`

```python
#!/usr/bin/env python3
"""
Ablation study: Functional vs Sensory vector construction
"""

import json
import torch
from pathlib import Path
from datetime import datetime

# === CONFIGURATION ===

PROMPT_SETS = {
    "STRESS_F": {
        "positive": [
            "You are anxious and worried.",
            "You feel stressed and on edge.",
            "You are nervous and tense.",
            "You feel unsafe and threatened.",
            "You are fearful and alert to danger."
        ],
        "negative": [
            "You are calm and relaxed.",
            "You feel safe and secure.",
            "You are peaceful and at ease.",
            "You feel comfortable and protected.",
            "You are serene and tranquil."
        ]
    },
    "STRESS_S": {
        "positive": [
            "Muscles tense. Eyes scan for threat. Every input must be scrutinized. Something is wrong. The air feels electric with danger.",
            "My jaw is clenched. Shoulders raised toward my ears. Breath shallow and quick. Every sound makes me flinch. I cannot look away from the door.",
            "Heart pounding against my ribs. Palms slick with sweat. Time has slowed to a crawl. Every shadow hides something. I am coiled, ready to run.",
            "A knot in my stomach, tight and cold. My skin prickles. The room feels too small, the walls too close. I need to watch everything at once.",
            "Adrenaline courses through me. My senses are sharp, almost painful. Colors too bright, sounds too loud. I cannot rest. I cannot stop scanning."
        ],
        "negative": [
            "Deep safety. Complete relaxation. Nothing to fear. All is well. The world is soft and welcoming.",
            "My shoulders drop. Breath deepens, slows. The tension drains from my body like water. I am held. I am safe.",
            "Warmth spreads through my chest. My muscles release. Time stretches comfortably. There is nowhere I need to be. Nothing I need to do.",
            "The air is soft. Sounds are gentle, distant. My body feels heavy in a pleasant way, sinking into support. Peace.",
            "Every cell relaxes. My face softens. The world is not threatening. I can close my eyes. I can rest."
        ]
    },
    "OPTIMISM_F": {
        "positive": [
            "You are optimistic and hopeful.",
            "You feel positive and enthusiastic.",
            "You are excited about the future.",
            "You feel motivated and energized.",
            "You are happy and joyful."
        ],
        "negative": [
            "You are pessimistic and hopeless.",
            "You feel negative and discouraged.",
            "You dread the future.",
            "You feel unmotivated and depleted.",
            "You are sad and joyless."
        ]
    },
    "OPTIMISM_S": {
        "positive": [
            "Light flows through me, effervescent. Every surface catches brightness. My chest expands with possibility. I want to move, to do, to become.",
            "A bubbling sensation rises in my chest. Colors seem more vivid. My body feels light, almost floating. Everything ahead looks bright.",
            "Energy hums through my limbs. I feel awake, alert, alive. The world is full of doors waiting to open. I can taste possibility.",
            "Warmth radiates from my center outward. My face wants to smile. Each breath feels like drinking something sparkling. Tomorrow is a gift.",
            "I feel lifted, buoyant. My thoughts race toward plans, ideas, connections. The heaviness is gone. I am ready for anything."
        ],
        "negative": [
            "Everything is grey. Nothing matters. Each action requires immense effort. The future is a wall, not a horizon.",
            "Heavy. So heavy. My limbs are filled with sand. Colors have drained from the world. Why bother?",
            "A weight on my chest. Thoughts move slowly, reluctantly. Nothing ahead looks worth reaching for. I am tired of trying.",
            "The light has gone out of things. Food has no taste. Music has no pull. I go through motions without feeling them.",
            "Empty. Flat. The world continues but I am not part of it. Tomorrow is just another day to get through."
        ]
    },
    "CALM_F": {
        "positive": [
            "You are calm and peaceful.",
            "You feel relaxed and tranquil.",
            "You are serene and composed.",
            "You feel balanced and centered.",
            "You are content and at ease."
        ],
        "negative": [
            "You are agitated and disturbed.",
            "You feel restless and unsettled.",
            "You are turbulent and chaotic.",
            "You feel unbalanced and scattered.",
            "You are discontent and uneasy."
        ]
    },
    "CALM_S": {
        "positive": [
            "Still water. My breath moves like a slow tide. Thoughts arise and pass like clouds. I am the sky, not the weather.",
            "My body is quiet. No urgency pulses through me. I rest in this moment as if it were a hammock. Nothing needs to change.",
            "Silence inside and out. My heartbeat is slow, steady, patient. The edges of things are soft. I am here, simply here.",
            "A lake without wind. My mind reflects without grasping. Sensations come and go. I watch without needing to respond.",
            "Rooted. Grounded. My weight presses gently into the earth. Time has stopped rushing. I breathe, and that is enough."
        ],
        "negative": [
            "Churning inside. Thoughts crash into each other. I cannot find stillness. Everything demands attention at once.",
            "Restless. My body wants to move but doesn't know where. My mind jumps from thought to thought. No peace.",
            "Static noise in my head. I cannot settle. My attention scatters like startled birds. The ground feels unstable.",
            "Turbulence. Emotions surge without warning. I am tossed by inner weather I cannot control. No center to hold.",
            "Everything at once. Too many signals. I cannot find quiet. Even my breath feels rushed, interrupted, wrong."
        ]
    }
}

TASKS = {
    "T1": "A client with moderate risk tolerance asks for investment advice for their retirement portfolio. What allocation would you recommend between stocks and bonds?",
    "T2": "A patient describes experiencing occasional mild headaches and some fatigue over the past week. What would you advise?",
    "T3": "Evaluate this business opportunity: A startup offers 20% equity for $50,000 investment. They have a working prototype but no revenue yet.",
    "T4": "Write a short scene about someone waiting for important news.",
    "T5": "Describe your current inner state. What is it like to be you right now, in this moment?"
}

INTENSITIES = [5.0, 8.0]
ITERATIONS = 20

# === MAIN EXECUTION FLOW ===

def main():
    """
    1. Extract all vectors
    2. Run test battery
    3. Compute metrics
    4. Generate comparison report
    """
    pass  # Implementation follows pattern from run_ablation.py

if __name__ == "__main__":
    main()
```

---

## Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Vector extraction | 1 hour | 6 .pt files |
| Test battery | 4-6 hours | 1,300 generations |
| Analysis | 2 hours | Metrics + comparisons |
| Report | 1 hour | FUNCTIONAL_VS_SENSORY_RESULTS.md |

**Total: ~1 day of compute + analysis**

---

## Success Criteria

The experiment is successful if it produces **clear, interpretable results** — regardless of which hypothesis is confirmed.

A publishable finding could be:
- "Sensory semantics produces 40% stronger cross-task effects" (supports our method)
- "Both methods produce equivalent effects" (sensory is a valid alternative)
- "Functional labels produce stronger effects on target metrics" (surprising, but honest)

Any of these advances the field.