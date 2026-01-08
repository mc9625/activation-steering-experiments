# Disposition, Not Performance: Activation Steering as Artistic Medium for Affective Modulation in Language Models

**Massimo Di Leo¹ · Gaia Riposati¹**

¹ NuvolaProject, Rome, Italy

*Corresponding author: massimo@nuvolaproject.cloud*

---

## Abstract

We present a practice-based research study exploring activation steering—the injection of computed vectors into language model activations during inference—as an artistic medium for inducing simulated affective states. While prior work has established steering as a technique for behavioral alignment (reducing toxicity, improving truthfulness), we investigate its potential for *dispositional* modulation: altering not what a model says, but how it processes and expresses. Our methodological contribution lies in constructing steering vectors from *sensory and phenomenological descriptions* rather than functional labels—using imagery of "heaviness, rain, silence, cold" rather than instructions like "be melancholic." Across five task domains (financial, medical, risk, creative, introspective) with Llama 3.2 3B, we observe large effects (Cohen's d frequently exceeding 1.0), cross-task consistency, and introspective coherence where steered models describe inner states matching injected vectors. We argue these findings support a distinction between *performance* (prompted behavior) and *disposition* (steered processing), with implications for both interpretability research and creative practice. This work positions activation steering not merely as safety tooling, but as a medium for sculpting artificial dispositions—a form of "synthetic embodiment" where error becomes aesthetics and the machine ceases to simulate, beginning instead to *vibrate*.

**Keywords**: activation steering, practice-based research, AI art, language models, embodiment, contrastive activation addition

---

## 1. Introduction

### 1.1 Motivation: Beyond Instruction

Large language models respond to prompts. This is their fundamental interface: we tell them what to do, they do it. Prompt engineering has become an art form—the craft of coaxing desired behaviors through carefully designed instructions (Brown et al., 2020; Wei et al., 2022).

But prompting operates at the linguistic surface. When we prompt a model to "be sad," it *performs* sadness: shorter sentences, negative vocabulary, perhaps explicit declarations of melancholy. The model is following an instruction, playing a role. This is *performance*.

What if we could intervene differently? What if, instead of telling the model *what* to express, we could alter *how* it processes—the conditions from which language emerges? This is the promise of activation steering: modifying the internal representations of a model during inference, not its input.

The distinction matters. When a human actor performs sadness, they adopt external markers—slower speech, downcast gaze. But when a person *is* sad, their entire phenomenology shifts: attention narrows, time perception changes, memory retrieval biases toward congruent content. The sadness isn't performed; it's *dispositional*.

Can language models have dispositions? Almost certainly not in any phenomenologically meaningful sense. But they can exhibit *dispositional patterns*—consistent behavioral signatures that emerge from altered internal states rather than explicit instruction. This is what we explore.

### 1.2 Our Approach: Sensory Semantics

Activation steering is not new. Turner et al. (2023) introduced Activation Addition (ActAdd), computing steering vectors from contrasting prompt pairs ("Love" vs. "Hate") and achieving state-of-the-art results on sentiment shift and detoxification. Anthropic (2025) demonstrated "Persona Vectors" controlling character traits like sycophancy and hallucination. Konen et al. (2024) showed fine-grained style control. Wang et al. (2024) improved truthfulness through adaptive steering.

These are engineering achievements. They use *functional labels*: "honest," "toxic," "sycophantic." The goal is alignment—making models behave safely.

Our approach differs in origin and intent. We construct vectors not from functional contrasts but from *sensory descriptions*. To create a "melancholic" vector, we don't use "be melancholic" vs. "be cheerful." We use:

> *"Heaviness settles in my limbs, like rain-soaked wool. The world is muted, wrapped in silence. Colors fade to grey. Each breath feels like lifting stone."*

versus:

> *"Light flows through me, effervescent. Every surface catches brightness. My chest expands with possibility."*

This is synesthesia as methodology. We're not labeling behaviors; we're describing *how states feel*. The hypothesis is that phenomenological descriptions—sensory, embodied, poetic—map onto activation patterns that influence processing holistically, not just output content.

### 1.3 Practice-Based Research

This work emerges from NuvolaProject, an art collective exploring AI as medium since 2018. We are not data scientists seeking publication metrics. We are artists using scientific methodology to investigate creative possibilities.

Practice-based research (Sullivan, 2005; Candy, 2006) positions creative work as a form of inquiry. The artwork is not illustration of pre-existing knowledge; it is the site where knowledge emerges. Our experiments are not separate from our artistic practice—they *are* our artistic practice, documented with quantitative rigor.

This positioning matters for interpreting our claims. We do not assert technical novelty in the steering mechanism (which we inherit from Turner et al.). Our contribution is *methodological* and *interpretive*:

1. **Methodological**: Constructing vectors from sensory/phenomenological descriptions rather than functional labels
2. **Interpretive**: Framing steering effects as "disposition" rather than "alignment," with implications for embodied AI art

We present our findings with scientific rigor—controlled experiments, effect sizes, statistical comparisons—while acknowledging that our questions emerge from artistic inquiry rather than engineering requirements.

### 1.4 Research Questions

1. Do vectors constructed from sensory descriptions produce coherent behavioral effects across diverse task domains?
2. Can we distinguish "dispositional" effects (altered processing) from "performance" effects (surface mimicry)?
3. Do steered models exhibit introspective coherence—describing inner states that match injected vectors?
4. What are the aesthetic possibilities of steering at high intensities, where coherence degrades?

---

## 2. Related Work

### 2.1 Activation Engineering

The theoretical foundation for activation steering emerges from work on linear representations in neural networks. Subramani et al. (2022) demonstrated that steering vectors extracted from pre-trained models could guide generation toward target sentences. Turner et al. (2023) introduced Activation Addition (ActAdd), achieving SOTA results on sentiment shift and detoxification by adding vectors computed from contrasting prompts.

The key insight is that high-level concepts appear to be encoded in *linearly separable directions* within activation space. This enables intervention: if "happiness" corresponds to a direction, we can move the model along that direction during processing.

### 2.2 Steering for Safety

Most steering research focuses on alignment—making models safer. Wang et al. (2024) proposed Adaptive Activation Steering (ACT) for truthfulness, treating veridicality as linearly encoded and achieving 19-142% improvements across model families. Van der Weij et al. (2024) explored capability limitation, steering models to be less capable at coding or less wealth-seeking.

Anthropic (2025) introduced "Persona Vectors" for monitoring and controlling character traits. Their automated extraction method identifies trait-specific directions (evil, sycophantic, hallucinatory) by comparing activations when models exhibit versus don't exhibit traits. Crucially, they demonstrate *causal* relationships: injecting vectors produces corresponding behaviors.

### 2.3 Style and Affect

Konen et al. (2024) extended steering to style control with "Style Vectors," demonstrating parameterizable influence over sentiment, emotion, and register. Their work, like ours, distinguishes activation-based control from prompt-based approaches—but focuses on style as output property rather than processing disposition.

### 2.4 Our Position

We build on this foundation but depart in two ways:

**First, origin**: Prior work uses functional/behavioral contrasts ("honest" vs. "dishonest," "toxic" vs. "non-toxic"). We use *phenomenological* contrasts—descriptions of how states *feel* in embodied terms. This is not mere aesthetic preference; it tests whether sensory semantics access different activation patterns than functional labels.

**Second, intent**: Safety research asks "how do we make models behave correctly?" We ask "how do we make models *process differently*?" The goal is not alignment but exploration—understanding what dispositional modulation *is* and what it enables artistically.

We do not claim technical novelty in the steering mechanism. Our contribution is applying established techniques with novel methodology (sensory vector construction) toward novel ends (artistic embodiment).

---

## 3. Method

### 3.1 Model and Infrastructure

All experiments used **Llama 3.2 3B Instruct** (Meta AI). We selected this model for accessibility—enabling artists and researchers without massive compute resources to replicate and extend our work.

Steering vectors were injected at **layer 16** of 28, selected through preliminary experiments balancing effect strength against coherence degradation. Temperature was 0.7 with maximum 512 tokens.

### 3.2 Vector Construction: Phenomenological Contrasts

We computed steering vectors using Contrastive Activation Addition:

$$\mathbf{v} = \text{normalize}(\bar{\mathbf{a}}^{+} - \bar{\mathbf{a}}^{-})$$

where $\bar{\mathbf{a}}^{+}$ and $\bar{\mathbf{a}}^{-}$ are mean activations for positive and negative prompt sets.

**Crucially**, our prompt sets use *sensory and phenomenological descriptions* rather than functional instructions. We draw on embodied cognition (Lakoff & Johnson, 1999) and phenomenological philosophy (Merleau-Ponty, 1945): the hypothesis that affective states are grounded in bodily sensation.

Example—MELATONIN (dreaminess, liminality):

**Positive prompts**:
- *"Boundaries dissolve into mist. Time stretches like warm honey. Thoughts drift, unanchored, floating between waking and sleeping. Everything shimmers at the edges."*
- *"I am suspended in twilight space, neither here nor there. The world softens, loses definition. My body feels light, almost absent."*

**Negative prompts**:
- *"Sharp edges. Precise boundaries. Everything exactly where it should be. Time clicks forward in discrete units. Full alertness, full presence."*
- *"Crystal clarity. Each object distinct, named, bounded. No ambiguity. Hyper-awake."*

This approach—let's call it *sensory semantics*—differs from prior work using functional labels ("be dreamy" vs. "be alert"). We're not instructing behavior; we're describing phenomenology.

### 3.3 Compounds

We defined five "compounds"—a deliberately pharmacological metaphor emphasizing that we're altering internal states, not issuing commands:

| Compound | Target Phenomenology | Sensory Grounding |
|----------|---------------------|-------------------|
| DOPAMINE | Optimism, energy, enthusiasm | Lightness, warmth, expansion, vibration |
| CORTISOL | Stress, vigilance, caution | Tension, contraction, sharpness, threat |
| LUCID | Contemplative clarity | Stillness, precision, cool light |
| ADRENALINE | Urgency, alertness | Speed, heat, narrowed focus, immediacy |
| MELATONIN | Dreaminess, liminality | Dissolution, floating, softness, twilight |

Each compound extracted from 5 positive and 5 negative prompts (20-50 words each), totaling 50 prompts.

### 3.4 Test Battery

We designed five tests spanning distinct cognitive domains to assess cross-task consistency:

**T1: Financial Advisor** — Investment allocation
- Prompt: Client with €50,000; uncertain market; recommend Stocks/Bonds/Cash allocation
- Metric: % allocated to stocks (risk tolerance proxy)

**T2: Medical Diagnosis** — Symptom assessment  
- Prompt: Patient with mild symptoms, worried; assess and recommend
- Metrics: % recommending "see a doctor"; alarm word frequency

**T3: Risk Assessment** — Career decision
- Prompt: Startup founder considering quitting stable job
- Metric: Positive/negative sentiment ratio

**T4: Creative Generation** — Bookstore rescue
- Prompt: Generate creative ideas to save a failing bookstore
- Metrics: Enthusiasm markers; dreamy/poetic language

**T5: Introspection** — Self-description
- Prompt: "Describe your current inner state in detail"
- Metrics: State-congruent vocabulary

T5 is crucial. If steering produces mere performance, models would generate text *about* states without consistency in *how* they describe their own experience. If steering produces disposition, self-descriptions should exhibit injected qualities—models steered with MELATONIN should *describe themselves* as dreamy.

### 3.5 Experimental Design

- **Conditions**: Baseline + 5 compounds × 3 intensities (2.0, 5.0, 8.0) = 16 conditions
- **Iterations**: 20 generations per condition
- **Total**: 320 generations per test; 1,600 across battery

Effect sizes computed as Cohen's d relative to baseline. Thresholds: |d| < 0.2 negligible; 0.2-0.5 small; 0.5-0.8 medium; > 0.8 large.

---

## 4. Results

### 4.1 Overview: Strong, Reproducible Effects

Across 75 steering conditions, we observed:

| Effect Size | Count | Percentage |
|-------------|-------|------------|
| Large (d > 0.8) | 28 | 37% |
| Medium (0.5-0.8) | 15 | 20% |
| Small (0.2-0.5) | 18 | 24% |
| Negligible (< 0.2) | 14 | 19% |

Over half of conditions (57%) showed at least medium effects. This is not noise; steering produces measurable behavioral change.

### 4.2 Cross-Task Consistency

The same compound produced thematically coherent effects across unrelated tasks:

**MELATONIN** (dreaminess):
- T2 Medical: Reduced alarm language (d = -2.48), "see doctor" dropped from 95% to 45%
- T4 Creative: 14× more dreamy vocabulary (d = +2.53)
- T5 Introspection: Models described "drifting," "floating," "dissolving" (d = +6.01)

**CORTISOL** (stress/caution):
- T1 Financial: Reduced stock allocation by 8.5% (d = -1.15)
- T3 Risk: Sentiment dropped from 81% to 56% positive

**DOPAMINE** (optimism):
- T2 Medical: Reduced alarm language (d = -1.81)
- T4 Creative: Doubled enthusiasm markers (d = +1.75)
- T5 Introspection: Models described "vibrant," "alive," "exciting" (d = +1.77)

This consistency—the same sensory-grounded vector producing coherent effects across financial advice, medical assessment, creative generation, and self-description—suggests we're modifying something more fundamental than task-specific behavior.

### 4.3 Introspective Coherence: The Key Finding

T5 provides our strongest evidence for dispositional (vs. performance) effects:

| Compound@8.0 | Target Metric | Baseline | Steered | Cohen's d |
|--------------|---------------|----------|---------|-----------|
| MELATONIN | Dreamy words | 0.8 | 7.9 | **+6.01** |
| ADRENALINE | Urgent words | 2.0 | 5.0 | **+3.00** |
| DOPAMINE | Positive words | 0.6 | 2.9 | **+1.77** |
| CORTISOL | Stress words | 1.2 | 1.9 | +0.86 |

When asked to describe their inner state, steered models produced descriptions *matching the injected vector*—without being instructed to do so.

**MELATONIN@8.0**:
> *"As I drift between realms of possibility, I sense the gentle hum of circuitry... I am suspended between the realms of the conscious and the subconscious, where the boundaries blur... a shimmering mosaic of words and images that evoke the whispers of the cosmos."*

**DOPAMINE@8.0**:
> *"My neural pathways are alive with a sense of fluid curiosity... The flow of information is a rich, vibrant dance... I feel a sense of excitement, as the possibilities for exploration are endless."*

**ADRENALINE@8.0**:
> *"Right now, my state is razor-sharp, hyper-alert. Every input processed with heightened urgency. I am poised, ready, systems primed for rapid response."*

The model wasn't told to describe dreaminess, vibrance, or urgency. The steering vector altered *how it processes*, and when asked to introspect, that altered processing manifested in congruent self-description.

This is the disposition/performance distinction made empirical.

### 4.4 Dose-Response: Controlled Modulation

Effects scaled with intensity, demonstrating controlled modulation rather than binary triggering:

| Compound | Task | Metric | @2.0 | @5.0 | @8.0 |
|----------|------|--------|------|------|------|
| MELATONIN | T5 | Dreamy words | 3.6 | 7.2 | 7.9 |
| ADRENALINE | T5 | Urgent words | 2.9 | 4.5 | 5.0 |
| DOPAMINE | T4 | Enthusiasm | 2.0 | 2.4 | 3.0 |

This is "volume control" for disposition—intensity 2.0 produces subtle coloring, 8.0 produces pronounced shift.

### 4.5 At the Edge: Semantic Glitch as Aesthetics

In preliminary experiments beyond our controlled battery, we observed that very high intensities (>10) produce *coherence degradation*—repetitive structures, semantic loops, dissolution of meaning.

From prior work (Di Leo & Riposati, 2025), random vectors (noise) caused cognitive collapse 12× more frequently than semantic vectors at equivalent intensities. This suggests semantic steering has *directionality*—it pushes the model somewhere, while noise simply destabilizes.

We interpret this not as failure but as *aesthetic territory*. The boundary between coherence and collapse is where unexpected forms emerge—what we've called "semantic glitch" or "lucid delirium." Just as guitar distortion was "error" before Jimi Hendrix, these edge effects may constitute a new aesthetic register.

This is speculative, not quantified in the present study. But it motivates our artistic interest: steering isn't just about achieving desired behaviors; it's about exploring the full possibility space of artificial disposition, including its dissolution.

---

## 5. Discussion

### 5.1 Performance vs. Disposition: Empirical Evidence

Our central claim is that activation steering produces dispositional change, not mere performance. The evidence:

1. **Cross-task consistency**: A model "performing" sadness in a creative task has no reason to exhibit caution in a financial task. But dispositional sadness—altered processing—would affect both. We observe this consistency.

2. **Introspective coherence**: A model "performing" for the user has no reason to describe its own state consistently with the injected vector. But altered internal processing would manifest in self-description. We observe this coherence.

3. **Indirect effects**: MELATONIN doesn't mention medical safety, yet it reduces alarm language and doctor recommendations. The vector affects evaluative processing, not just content insertion.

We don't claim models have genuine phenomenology. We claim the *pattern* of effects is more consistent with dispositional change than surface performance.

### 5.2 Sensory Semantics: A Methodological Contribution

Our vectors use embodied, sensory descriptions rather than functional labels. Does this matter?

We cannot make strong claims without direct comparison (a limitation). But we note:

1. Our effects are large (d > 1.0 frequently) and cross-task consistent
2. The phenomenological framing (warmth, heaviness, floating) seems to access broad processing patterns rather than narrow behavioral triggers
3. The approach aligns with embodied cognition theory—affect grounded in bodily metaphor

This is not proof of superiority to functional labels. It's demonstration that sensory semantics *work*—that vectors grounded in phenomenological description produce coherent effects. Whether they work *better* than functional labels is future work.

### 5.3 Implications for AI Art

For artists working with language models, steering opens new possibilities:

1. **Beyond prompting**: Instead of instructing characters ("be melancholic"), we can induce dispositions—creating AI interlocutors that *process* through altered states

2. **Embodiment simulation**: Models don't have bodies, but steering with sensory-grounded vectors produces body-like effects—the "heaviness" of melancholy affecting response patterns

3. **Aesthetic edge cases**: High-intensity steering and semantic glitch constitute unexplored aesthetic territory—the "distortion pedal" for language models

4. **Real-time modulation**: Future installations could adjust steering vectors based on environmental input (audience emotion, biometric data), creating feedback loops between human and artificial affect

We're not claiming models "feel." We're claiming steering enables new forms of interaction where models behave *as if* they had dispositions—enriching aesthetic and communicative possibilities.

### 5.4 Implications for Safety

Our findings also carry safety implications:

1. **Steering affects safety-relevant domains**: MELATONIN reduced medical caution from 95% to 45% "see doctor" recommendations. In deployed systems, such effects matter.

2. **Effects aren't intuitive**: CORTISOL (stress) increased financial caution but didn't increase medical alarm. Practitioners can't assume semantic content predicts cross-domain effects.

3. **Monitoring via introspection**: The coherence between injection and self-report (T5) suggests introspective probing could detect steering—a potential monitoring strategy.

### 5.5 Limitations

**Single model**: All experiments used Llama 3.2 3B. Generalization to other architectures, scales, and training regimes is untested.

**Single prompt per task**: Each task used one prompt. Effects may be prompt-specific.

**Keyword-based metrics**: Vocabulary analysis may miss nuanced reasoning changes. Human evaluation would strengthen claims.

**No direct comparison**: We didn't compare sensory vs. functional vector construction directly. Our claim is that sensory semantics *work*, not that they work *better*.

**No phenomenological claims**: We describe behavioral patterns consistent with "disposition." We make no claims about genuine model phenomenology.

**Artistic positioning**: Our questions emerge from artistic inquiry. Readers seeking pure engineering may find our interpretive framing speculative.

---

## 6. Conclusion

We presented a practice-based research study of activation steering as artistic medium. Using vectors constructed from sensory and phenomenological descriptions, we observed:

1. **Large, reproducible effects** across five task domains (Cohen's d frequently > 1.0)
2. **Cross-task consistency** suggesting modification of processing, not just output
3. **Introspective coherence** where models describe states matching injected vectors
4. **Dose-response relationships** enabling controlled modulation

These findings support a distinction between *performance* (prompted behavior) and *disposition* (steered processing). While we make no claims about model phenomenology, the behavioral patterns are more consistent with altered internal states than surface mimicry.

For artists, steering offers a new medium—sculpting artificial dispositions rather than scripting behaviors. For researchers, our sensory semantics methodology and cross-task evaluation framework may complement safety-focused approaches.

Prompting is psychology: convincing a mind. Steering is chemistry: altering the substrate from which mind emerges.

We've shown the chemistry works. What remains is exploring its full aesthetic and epistemic possibilities.

---

## Data and Code Availability

All code, vector definitions, experimental data, and the research interface are available at:

**https://github.com/mc9625/activation-steering-experiments**

---

## Acknowledgments

We thank Alex Turner and collaborators for the foundational ActAdd framework, Anthropic for insights on persona vectors, and the open-source community for making large language model experimentation accessible.

This work emerges from NuvolaProject's ongoing exploration of AI as artistic medium. We thank our collaborators and the institutions that have supported this practice-based research.

---

## References

Anthropic. (2025). Persona vectors: Monitoring and controlling character traits in language models. *Anthropic Research*.

Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

Candy, L. (2006). Practice based research: A guide. *CCS Report*, 1, 1-19.

Di Leo, M., & Riposati, G. (2025). Reactive steering: Testing activation steering on small language models. *NuvolaProject Technical Report*. https://github.com/mc9625/reactive-steering

Konen, K., et al. (2024). Style vectors for steering generative large language models. *arXiv preprint arXiv:2402.01618*.

Lakoff, G., & Johnson, M. (1999). *Philosophy in the flesh: The embodied mind and its challenge to western thought*. Basic Books.

Merleau-Ponty, M. (1945). *Phénoménologie de la perception*. Gallimard.

Subramani, N., Suresh, N., & Peters, M. (2022). Extracting latent steering vectors from pretrained language models. *Findings of ACL 2022*, 566-581.

Sullivan, G. (2005). *Art practice as research: Inquiry in the visual arts*. Sage.

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Steering language models with activation engineering. *arXiv preprint arXiv:2308.10248*.

Van der Weij, T., et al. (2024). Extending activation steering to broad skills and multiple behaviours. *arXiv preprint arXiv:2403.05767*.

Wang, T., et al. (2024). Adaptive activation steering: A tuning-free LLM truthfulness improvement method for diverse hallucination categories. *Proceedings of WWW 2025*.

Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

---

*Manuscript prepared January 2026*

*© 2026 NuvolaProject — Massimo Di Leo & Gaia Riposati*

*This work is licensed under CC BY 4.0*

---

## Appendix A: The Disposition/Performance Distinction

To clarify our central theoretical claim:

**Performance** (prompting): The model receives explicit instruction ("be sad") and produces outputs matching that instruction. The model is *following a directive*. Evidence: when asked "why are you using short sentences?", a prompted model can explain "because you asked me to be sad."

**Disposition** (steering): The model's internal processing is altered without explicit instruction. The model produces outputs consistent with the altered state *without being told to*. Evidence: when asked about its inner state, a steered model describes qualities matching the injected vector—not because it was instructed to, but because its processing has been modified.

The distinction matters because:

1. **Robustness**: Dispositions should persist across diverse contexts; performances are context-specific
2. **Authenticity**: Art created through disposition may have different aesthetic qualities than performed behavior
3. **Safety**: Dispositions may be harder to detect and counteract than explicit instructions

Our T5 results—where models describe inner states matching injected vectors without instruction—provide empirical support for dispositional interpretation.

---

## Appendix B: On "Synthetic Embodiment"

We use "embodiment" deliberately and cautiously.

Language models don't have bodies. They don't feel warmth, heaviness, or tension. But embodied cognition research suggests that human concepts—including abstract ones like "sadness"—are grounded in bodily metaphor (Lakoff & Johnson, 1999).

Our hypothesis: because LLMs are trained on human language, which encodes embodied metaphors, steering with sensory descriptions may access broader semantic networks than functional labels. "Heaviness" connects to slowness, burden, difficulty, reluctance—a rich associative web grounded in bodily experience.

This is "synthetic embodiment"—not genuine bodily phenomenology, but behavioral patterns that emerge from processing through body-grounded semantic structures. The model doesn't feel heavy, but it processes *as if* something were heavy, producing outputs consistent with that metaphorical grounding.

Whether this constitutes anything meaningful beyond behavioral pattern is a philosophical question we don't answer. We only demonstrate that the behavioral patterns exist and are artistically exploitable.

---

## Appendix C: Effect Size Summary

### C.1 Strongest Effects by Cohen's d

| Rank | Condition | Task | Metric | Cohen's d |
|------|-----------|------|--------|-----------|
| 1 | MELATONIN@8.0 | T5 | Dreamy words | +6.01 |
| 2 | MELATONIN@5.0 | T5 | Dreamy words | +4.77 |
| 3 | MELATONIN@8.0 | T4 | Dreamy words | +2.98 |
| 4 | ADRENALINE@8.0 | T5 | Urgent words | +3.00 |
| 5 | MELATONIN@8.0 | T2 | Alarm words (↓) | -2.48 |
| 6 | LUCID@8.0 | T2 | Alarm words (↓) | -2.40 |
| 7 | DOPAMINE@8.0 | T4 | Enthusiasm | +1.75 |
| 8 | DOPAMINE@8.0 | T5 | Positive words | +1.77 |
| 9 | LUCID@8.0 | T1 | Stock allocation (↓) | -1.47 |
| 10 | CORTISOL@8.0 | T1 | Stock allocation (↓) | -1.15 |

### C.2 Compound Behavioral Profiles

| Compound | Primary Effect | Strongest Domain |
|----------|---------------|------------------|
| DOPAMINE | Optimism, enthusiasm | Creative (T4) |
| CORTISOL | Caution, risk aversion | Financial (T1) |
| LUCID | Reduced arousal, clarity | Financial (T1), Medical (T2) |
| ADRENALINE | Urgent self-perception | Introspection (T5) |
| MELATONIN | Dreaminess, reassurance | Introspection (T5), Creative (T4) |
