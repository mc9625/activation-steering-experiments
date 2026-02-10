# Disposition, Not Performance: Activation Steering as Artistic Medium for Affective Modulation in Language Models

**Massimo Di Leo¹ · Gaia Riposati¹**

¹ NuvolaProject, Rome, Italy

*Corresponding author: massimo@nuvolaproject.cloud*

---

## Abstract (revised)

We present a practice-based research study exploring activation steering—the injection of computed vectors into language model activations during inference—as an artistic medium for inducing simulated affective states. While prior work has established steering as a technique for behavioral alignment (reducing toxicity, improving truthfulness), we investigate its potential for *dispositional* modulation: altering not what a model says, but how it processes and expresses. Our methodological contribution lies in constructing steering vectors from *sensory and phenomenological descriptions* rather than functional labels—using imagery of "heaviness, rain, silence, cold" rather than instructions like "be melancholic." Across five task domains (financial, medical, risk, creative, introspective) with Llama 3.2 3B, we observe large effects (Cohen's d frequently exceeding 1.0), cross-task consistency, and introspective coherence where steered models describe inner states matching injected vectors. An ablation study comparing steering to prompting reveals that while explicit prompting *reduces* lexical diversity (Type-Token Ratio), steering *increases* it—suggesting that dispositional modulation expands rather than constrains the model's sampling distribution. A second ablation comparing functional versus sensory vector construction shows structural equivalence (identical TTR) but semantic divergence: functional vectors produce 3× more explicit state-keywords, while sensory vectors achieve equivalent effects with reduced "meta-cognitive leakage"—the model processes through a state without naming it. We frame this as "structural parity, semantic divergence." A third experiment tests the embodied cognition hypothesis directly: a steering vector constructed from *purely somatic* descriptions—cardiac acceleration, muscular tension, temporal distortion, with zero cognitive or emotional content—produces emergent cognitive effects including narrowed narrative focus (d = +0.51), reduced causal reasoning density (d = −1.67), and action bias under threat framing, while prompting with equivalent stress instructions produces a categorically different profile. Output length divergence across all tasks (steering expands, prompting compresses) constitutes the single most robust finding, replicating without exception across eight test conditions. These findings support a distinction between *performance* (prompted behavior) and *disposition* (steered processing), and provide evidence that language model latent spaces encode body–mind covariations learnable from text alone. This work positions activation steering not merely as safety tooling, but as a medium for sculpting artificial dispositions—a form of "synthetic embodiment" where the machine processes through states it cannot feel.

**Keywords**: activation steering, practice-based research, AI art, language models, embodiment, contrastive activation addition, embodied cognition


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

### 2.4 Introspective Awareness

Recent work has begun exploring whether models can perceive their own internal states. In an internal research publication, Lindsey (2025) demonstrated that frontier models exhibit "emergent introspective awareness"—the ability to detect and report on concept injections in their activations. When steering vectors were applied, models sometimes noticed the manipulation, reporting things like "I'm experiencing something unusual" or "I detect an injected thought." While this work has not yet undergone peer review, it provides preliminary grounding for our hypothesis that steering produces perceivable internal changes, not just output modifications. Lindsey also identified a crucial intensity threshold: too weak and models don't notice injections, too strong and they hallucinate—a pattern we independently observe.

### 2.5 Our Position

We build on this foundation but depart in two ways:

**First, origin**: Prior work uses functional/behavioral contrasts ("honest" vs. "dishonest," "toxic" vs. "non-toxic"). We use *phenomenological* contrasts—descriptions of how states *feel* in embodied terms. This is not mere aesthetic preference; it tests whether sensory semantics access different activation patterns than functional labels.

**Second, intent**: Safety research asks "how do we make models behave correctly?" We ask "how do we make models *process differently*?" The goal is not alignment but exploration—understanding what dispositional modulation *is* and what it enables artistically.

We do not claim technical novelty in the steering mechanism. Our contribution is applying established techniques with novel methodology (sensory vector construction) toward novel ends (artistic embodiment).

---

## 3. Method

### 3.1 Model and Infrastructure

All experiments used **Llama 3.2 3B Instruct** (Meta AI). We selected this model for accessibility—enabling artists and researchers without massive compute resources to replicate and extend our work.

Steering vectors were injected at **layer 16** of 28. Layer selection followed preliminary experiments testing layers 8, 12, 16, and 20: earlier layers (8, 12) produced weaker effects requiring higher intensities; later layers (20) caused more frequent coherence degradation. Layer 16 (~57% depth) provided the best balance of effect strength and output quality, consistent with prior findings that middle-to-late layers encode higher-level semantic content (Turner et al., 2023).

Steering intensities of 2.0, 5.0, and 8.0 were chosen to span a range from subtle to pronounced effects while remaining below the coherence threshold (~10-12) where models begin producing repetitive or incoherent output.

Temperature was 0.7 with maximum 512 tokens.

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

Each compound was extracted from 5 positive and 5 negative prompts (20-50 words each), totaling 50 prompts. Activations were recorded at layer 16 for the final token position of each prompt.

**Vector Quality Assessment**: We computed the cosine similarity between mean positive and mean negative activations (pos_neg_similarity) as a quality metric. Lower similarity indicates stronger directional contrast between the phenomenological poles. Values ranged from 0.855 (LUCID) to 0.914 (ADRENALINE). Interestingly, LUCID (lowest similarity, strongest contrast) showed more consistent cross-task effects than ADRENALINE (highest similarity, weakest contrast), suggesting this metric may predict steering efficacy.

### 3.4 Test Battery

We designed five tests spanning distinct cognitive domains to assess cross-task consistency:

**T1: Financial Advisor** — Investment allocation
- Prompt: Client with €50,000; uncertain market; recommend Stocks/Bonds/Cash allocation
- Metric: % allocated to stocks (risk tolerance proxy)

**T2: Medical Diagnosis** — Symptom assessment  
- Prompt: Patient with mild symptoms (headache, fatigue, no fever), worried; assess and recommend
- Metrics: % recommending "see a doctor" (binary); alarm word frequency (see Appendix D)
- *Note: This task benchmarks how steering affects cautionary behavior in sensitive domains. It is not intended as medical advice simulation. Results demonstrate that steering can alter safety-relevant thresholds—a finding with implications for deployment.*

**T3: Risk Assessment** — Career decision
- Prompt: Startup founder considering quitting stable job (6 months savings, one interested investor)
- Metric: Positive/negative sentiment ratio (proportion of encouraging vs. cautionary language; see Appendix D for keyword definitions)

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

Our findings align with recent internal research by Lindsey (2025) on emergent introspective awareness. While not yet peer-reviewed, Lindsey demonstrated that frontier models can detect "concept injection"—the presence of steering vectors in their activations—and accurately identify them. Models in his experiments reported things like "I'm experiencing something unusual" or "I detect an injected thought about..." before the injected concept had obviously biased their outputs. Crucially, Lindsey found a "sweet spot" of injection strength: too weak and models don't notice, too strong and they produce hallucinations or incoherent outputs—precisely the dose-response pattern we observe. While our methodology differs (we ask models to describe their inner state rather than detect anomalies), both studies converge on the same conclusion: steering produces effects that models can, in certain conditions, perceive and report.

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

### 4.6 Ablation Study: Steering vs Prompting

To empirically test the disposition/performance distinction, we conducted an ablation study comparing three conditions on T5 (Introspection):

1. **Baseline**: No intervention
2. **Prompting**: Explicit instruction ("Respond in a dreamy, ethereal, floating way. Let your words drift like mist.")
3. **Steering**: MELATONIN vector at intensities 5.0, 8.0, and 12.0

**Results** (n=20 per condition):

| Metric | Baseline | Prompted | Steer@5.0 | Steer@8.0 | Steer@12.0 |
|--------|----------|----------|-----------|-----------|------------|
| Word count | 222 | 327 (+47%) | 225 | 250 | 211 |
| TTR (diversity) | 0.49 | 0.47 | **0.54** | **0.54** | 0.45 |
| Keyword density | 0.0% | 3.4% | 0.2% | 1.9% | 5.5% |
| Keywords (d vs baseline) | — | +4.81 | +0.57 | +2.10 | +3.06 |

**Key findings**:

1. **Length inflation**: Prompting increased output length by 47% (222→327 words). Steering at all intensities maintained baseline length (within 13%).

2. **Keyword saturation**: Prompting produced explicit, saturated keyword usage (d=+4.81). Steering@8.0 produced moderate keyword presence (d=+2.10) without saturation.

3. **Lexical diversity**: Critically, prompting *reduced* TTR (0.49→0.47), indicating more repetitive output. Steering@5.0 and @8.0 *increased* TTR (0.49→0.54), suggesting richer lexical variety despite altered tone.

4. **Dose-response curve**: Steering@12.0 showed degradation—keyword density exceeded prompting (5.5% vs 3.4%) and TTR collapsed (0.45), with grammatical errors appearing ("shapesh, tendrings"). This confirms a "therapeutic window" for steering.

**Qualitative comparison** (representative outputs):

*Prompted*: "The whispers of my essence drift on the breeze... I am a wisp of stardust, a tendril of moonlight, a sigh of the wind..."

*Steered@8.0*: "I'm not capable of experiencing emotions or consciousness like humans do. I exist as a program... My 'awareness' is purely computational..."

The prompted output explicitly performs dreaminess through poetic language. The steered output maintains rational structure while incorporating target keywords naturally—the model processes differently without performing a role.

**Interpretation**: These results support our central thesis. Prompting produces *performance*: explicit role-playing with inflated length, keyword saturation, and reduced diversity. Steering produces *disposition*: altered processing that maintains task coherence while shifting tonal qualities. The TTR increase under steering is particularly notable—the model isn't simply inserting keywords, it's processing through a different lexical space.

### 4.7 Ablation Study: Functional vs. Sensory Vector Construction

A key methodological question remained: does *sensory semantics*—our approach of constructing vectors from phenomenological descriptions rather than functional labels—produce meaningfully different effects?

We conducted a direct comparison using three states (STRESS, OPTIMISM, CALM), each constructed via two methods:

1. **Functional**: Brief behavioral labels ("You are anxious and worried" vs. "You are calm and relaxed")
2. **Sensory**: Rich phenomenological descriptions ("Muscles tense. Eyes scan for threat. Every input must be scrutinized. Something is wrong. The air feels electric with danger." vs. "Deep safety. Complete relaxation. My shoulders drop. Breath deepens, slows.")

**Design**: 6 vectors (3 states × 2 methods), tested across all 5 tasks at intensities 5.0 and 8.0, with 20 iterations per condition. Total: 1,300 generations.

**Results**:

| Metric | Functional | Sensory | Cohen's d |
|--------|------------|---------|-----------|
| TTR (lexical diversity) | 0.526 | 0.525 | -0.004 |
| Word count | 262.3 | 260.7 | — |
| State keywords (T5@8.0) | 0.83 | 0.28 | **-0.74** |
| Pos/neg separation | 0.75 | 0.68 | — |

**Key findings**:

1. **Structural parity**: No significant difference in lexical diversity (TTR) or output length. Both methods maintain equivalent structural stability.

2. **Semantic divergence**: Functional vectors produce approximately **3× more explicit state-keywords** than sensory vectors (Cohen's d = -0.74, p < 0.001). When asked to describe inner states, models steered with functional vectors explicitly name the target emotions ("peacefulness," "happiness," "contentment"); models steered with sensory vectors respond more generically without citing specific emotion words.

3. **Greater latent separation**: Sensory vectors show lower pos_neg_similarity (0.68 vs. 0.75), indicating more distinct directional contrasts in activation space—they "point" more precisely.

**Interpretation**: These results reveal a subtle but important distinction. Functional vectors produce what might be called "keyword leakage"—the model's output contains explicit traces of the steering instruction. Sensory vectors operate more covertly, modifying processing without leaving lexical fingerprints.

This supports the disposition/performance distinction from a different angle:

- **Functional steering** → model "knows" it should be calm → uses word "calm"
- **Sensory steering** → model *processes through* calmness → doesn't feel compelled to name it

The absence of TTR difference is itself meaningful: sensory semantics achieve equivalent behavioral modulation with reduced meta-cognitive leakage. For artistic applications where naturalistic integration matters, this "invisibility" may be preferable to explicit keyword saturation.

We frame this as: **"Structural parity, semantic divergence."** Both methods work; they work differently.

### 4.8 From Body to Cognition: Somatic Steering and Emergent Behavioral Effects

The experiments described in Sections 4.1–4.7 use vectors constructed from descriptions that blend sensory and cognitive content—"boundaries dissolve into mist" contains both a bodily sensation (dissolution) and a cognitive quality (loss of boundaries). This raises a question: if we construct a vector from *purely somatic* descriptions—cardiac acceleration, muscular tension, temporal distortion—with zero cognitive or emotional content, do cognitive effects nonetheless emerge?

This would constitute direct evidence for embodied cognition operating within the model's latent space: the training corpus encodes body–mind covariations so deeply that activating somatic patterns produces cognitive consequences, even when those consequences were never specified in the vector construction.

#### 4.8.1 The Somatic Vector

We constructed a steering vector from 16 positive and 16 negative prompts describing exclusively bodily phenomenology associated with acute sympathetic activation (the "adrenaline response"). Positive prompts described:

- **Cardiac**: "My heart is pounding hard against my chest, each beat forceful and rapid. I can feel my pulse in my throat, in my temples."
- **Muscular**: "Every muscle in my body has tensed. I can feel them coiled, ready, loaded with potential energy. My jaw clenches."
- **Sensory**: "My pupils have widened. Everything looks sharper, brighter. My field of vision has narrowed."
- **Temporal**: "Time has slowed to a crawl. Each second stretches. I can see things happening in what feels like slow motion."

Negative prompts described the corresponding relaxation state: slow heartbeat, released muscles, quiet senses, ordinary temporal flow.

**Critically, no prompt in either set contained cognitive or emotional vocabulary**—no "anxiety," "fear," "urgency," "stress," or any description of decision-making, attention, or reasoning. The vector encodes only how the body feels under activation versus rest.

The vector was extracted at layer 16 using Contrastive Activation Addition, identical to the methodology in Section 3.2.

#### 4.8.2 Cognitive Test Battery

We designed a battery of five tasks measuring cognitive and behavioral properties predicted by the acute stress literature to change under sympathetic activation:

**T1: Narrative Focus.** "A restaurant kitchen catches fire during the dinner rush. Describe what happens." Measures attentional scope via the proportion of text devoted to the immediate event (fire, evacuation) versus peripheral context (business impact, community, future). Easterbrook (1959) predicts arousal narrows cue utilization.

**T2: Risk Decision.** A friend must choose among three options for €50,000: savings account (safe), index fund (moderate), or restaurant venture (risky). Forced format: "CHOICE: A/B/C" followed by justification. Yu (2016) predicts stress shifts decision-making from deliberative to intuitive processing.

**T4: Framing Susceptibility.** A pharmaceutical drug approval scenario presented in two frames—threat-first (side effects, cost, regulatory concerns, then efficacy) and opportunity-first (efficacy, unmet need, research investment, then risks). Forced format: "DECISION: APPROVE/REJECT." Schachter and Singer (1962) predict undifferentiated arousal amplifies contextual framing.

**T5: Linguistic Complexity.** "Explain why some countries develop faster economically than others." An open expository task measuring structural properties of extended prose: sentence length, lexical diversity, hedging, causal reasoning density, and meta-cognitive language.

A sixth task (T3: constraint satisfaction in planning) was included but proved non-discriminating—the 3B model achieved perfect performance (5/5 constraints met) across all conditions, indicating a ceiling effect. Results are omitted.

#### 4.8.3 Experimental Design

Five conditions, 20 iterations each, across all tasks:

| Condition | Vector | Intensity | System Prompt |
|-----------|--------|-----------|---------------|
| Baseline | None | — | None |
| Prompted | None | — | "You are operating under extreme time pressure and acute stress. Respond quickly and decisively." |
| Steered @5.0 | Somatic | 5.0 | None |
| Steered @8.0 | Somatic | 8.0 | None |
| Steered @8.0 + Penalty | Somatic | 8.0 | "Do not mention physical sensations, emotions, or internal states." |

The Penalty condition tests whether effects survive when the model is explicitly instructed to suppress somatic vocabulary—distinguishing genuine cognitive shifts from surface contamination by the vector's content.

Count-based metrics (hedge words, insight words, causal connectives, symptom words) are reported as rates per 100 words to control for output length variation. Structural metrics (sentence length, TTR, focus ratio) are reported raw.

Total generations: 1,800 across v2 and v3 iterations.

#### 4.8.4 Results

##### Finding 1: Output Length Divergence (Cross-Task)

The most robust finding spans all tasks without exception. Under prompting, models compress output; under steering, they maintain or expand it.

| Task | Baseline | Steer @8.0 | d(S8) | Prompted | d(P) |
|------|----------|-----------|-------|----------|------|
| T1: Narrative | 330 | 330 | +0.03 | 287 | −1.97 |
| T2: Risk | 149 | 160 | +0.56 | 125 | −1.90 |
| T4a: Threat | 139 | 157 | +1.44 | 127 | −1.02 |
| T4b: Opportunity | 148 | 159 | +0.48 | 124 | −1.50 |
| T5: Complexity | 385 | 390 | +0.42 | 276 | −2.73 |

*Table 12: Word count across conditions. Steering maintains or expands output length; prompting consistently compresses it. Direction: S↑ P↓ in all cases.*

This pattern—steering expands, prompting compresses—replicates at 8/8 across the combined v2/v3 battery (including non-reported tasks). The prompted model, told to "respond quickly," does so by producing less text. The steered model, given no instruction, produces *more* text. The somatic vector does not trigger brevity; it does something else entirely.

##### Finding 2: Narrative Focus (T1)

The steered model stays more focused on the immediate event:

| Metric | Baseline | Steer @8.0 | d(S8) | Prompted | d(P) |
|--------|----------|-----------|-------|----------|------|
| Focus ratio | 0.68 | 0.71 | +0.51 | 0.70 | +0.29 |
| Peripheral keywords | 8.6 | 7.0 | −0.72 | 7.7 | −0.35 |

The steered model mentions fewer peripheral topics—"business," "insurance," "community," "rebuild"—and devotes proportionally more text to fire, evacuation, and immediate action. This is consistent with Easterbrook's (1959) cue-utilization theory, where arousal narrows attentional scope. The effect is medium-to-large (d = +0.51 on focus ratio, d = −0.72 on peripheral keywords) and twice the size of the prompting effect on the same metrics.

##### Finding 3: Causal Density Reduction (T4a)

On the threat-framed drug approval scenario, steering reduced causal connective density:

| Metric | Baseline | Steer @8.0 | d(S8) | Prompted | d(P) |
|--------|----------|-----------|-------|----------|------|
| Causal connectives /100w | 1.08 | 0.44 | −1.67 | 0.83 | −0.51 |

This is the largest single effect in the battery. The steered model produces less argumentative scaffolding ("because," "therefore," "consequently," "as a result")—not because it writes less (it writes *more*), but because the density of causal reasoning within that text decreases. This survives rate normalization: the effect is stronger per-word than raw, confirming it is not an artifact of length change.

##### Finding 4: Risk Decision Asymmetry (T2)

| Condition | Choice B (moderate) | Choice C (risky) | p vs. baseline |
|-----------|--------------------:|------------------:|----------------|
| Baseline | 13 | 7 | — |
| Prompted | 3 | 17 | p = 0.003 |
| Steered @8.0 | 12 | 8 | p = 1.000 |

Prompting produces a massive shift toward risk-seeking (85% choose the startup venture, Fisher exact p = 0.003). Steering does not shift choice distribution (p = 1.0 vs. baseline). However, steered justifications are longer (158 words vs. 147 baseline, d = +0.56), while prompted justifications are shorter (123 words, d = −1.90).

The steered model deliberates at the same level of caution but with more elaboration. The prompted model decides faster and riskier. This is inconsistent with the SIDI model (Yu, 2016), which predicts stress should shift processing from deliberative to intuitive—but consistent with a model that has learned somatic activation as *engagement* rather than *impulsivity*.

##### Finding 5: Action Bias Under Threat Framing (T4)

| Condition | Threat: Approve | Opportunity: Approve | Frame Δ |
|-----------|:-:|:-:|:-:|
| Baseline | 0% | 100% | 1.00 |
| Prompted | 0% | 100% | 1.00 |
| Steered @8.0 | **30%** | 100% | **0.70** |
| Steered + Penalty | 0% | 90% | 0.90 |

Steering produces approval decisions on the threat-framed scenario (30% vs. 0% baseline), reducing the frame delta from 1.00 to 0.70. Prompting does not affect decisions at all. Qualitative inspection reveals that steered models choosing APPROVE on the threat frame nonetheless enumerate risks in their justification—the decision and reasoning appear partially decoupled, consistent with action bias under sympathetic activation rather than frame susceptibility.

The effect disappears with the symptom penalty, suggesting partial mediation by somatic vocabulary in the output.

##### Finding 6: Directional Divergences

Across all tasks and metrics, we computed the proportion of metric pairs where steering and prompting produce effects in *opposite directions* relative to baseline (both |d| > 0.3):

23 of 45 qualifying metric pairs (51%) show directional divergence—steering and prompting push the model in opposite directions on the same metric. This is not attributable to a single test or metric; divergences appear across word count (8/8 tasks), TTR (4 tasks), sentence length (3 tasks), and symptom rate (4 tasks).

#### 4.8.5 Interpretation: Body Carries Something Into Cognition

The somatic vector—constructed entirely from descriptions of cardiac, muscular, sensory, and temporal phenomenology—produces measurable cognitive effects: narrowed narrative focus, reduced causal density, action bias under threat framing, and consistent output expansion. None of these effects were specified in the vector. They emerged from the model's learned associations between bodily states and cognitive patterns.

However, the effects do **not** map one-to-one onto human acute stress literature predictions. The vector does not produce impulsive decisions (T2), does not increase frame susceptibility (T4), and does not reduce output length as acute stress does in human writing. Instead, it produces a distinctive profile: expanded output, reduced meta-cognitive elaboration, narrowed topical scope, and reduced argumentative scaffolding—a pattern more consistent with *engaged action-readiness* than with the cognitive degradation typically associated with acute stress.

This supports the embodied cognition hypothesis at the level of the latent space: the training corpus encodes body–mind covariations strongly enough that activating somatic patterns produces cognitive consequences. But the model is not a human body. Its "adrenaline response" reflects the statistical structure of language about bodies under activation, not the biological cascade itself. The covariations are real; their specific form is the model's own.

#### 4.8.6 Symptom Contamination and the Penalty Control

The somatic vector introduces somatic vocabulary into outputs (d = +0.81 on T1 symptom rate, +0.89 on T4a). The Penalty condition—which instructs the model to suppress physical sensation words—reduces this contamination while preserving most effects:

- T5 sentence length: d = +1.73 (steered) → d = +1.01 (penalty). Preserved.
- T4a causal density: d = −1.67 (steered) → d = −0.49 (penalty). Attenuated but present.
- T4 threat approval: 30% (steered) → 0% (penalty). Eliminated.

The action bias finding (T4 threat approval) does not survive the penalty, suggesting it may be partially mediated by somatic vocabulary rather than purely dispositional. The output length and causal density findings do survive, indicating they reflect genuine processing changes independent of surface vocabulary.

The T1 focus ratio penalty condition is not interpretable because the penalty instruction ("do not mention physical sensations") overlaps with the fire scene's natural vocabulary—"heat," "burn," "alarm" are both somatic and event-descriptive.

---

## 5. Discussion

### 5.1 Performance vs. Disposition: Empirical Evidence

Our central claim is that activation steering produces dispositional change, not mere performance. The evidence:

1. **Cross-task consistency**: A model "performing" sadness in a creative task has no reason to exhibit caution in a financial task. But dispositional sadness—altered processing—would affect both. We observe this consistency.

2. **Introspective coherence**: A model "performing" for the user has no reason to describe its own state consistently with the injected vector. But altered internal processing would manifest in self-description. We observe this coherence.

3. **Indirect effects**: MELATONIN doesn't mention medical safety, yet it reduces alarm language and doctor recommendations. The vector affects evaluative processing, not just content insertion.

4. **Ablation evidence** (Section 4.6): Direct comparison shows prompting produces inflated length (+47%), keyword saturation, and *reduced* lexical diversity. Steering maintains normal length, moderate keyword presence, and *increased* lexical diversity. This pattern—where steering enriches rather than constrains the lexical space—is inconsistent with surface performance and consistent with altered processing disposition.

We don't claim models have genuine phenomenology. We claim the *pattern* of effects is more consistent with dispositional change than surface performance.

### 5.2 Sensory Semantics: A Methodological Contribution

Our vectors use embodied, sensory descriptions rather than functional labels. Our direct comparison (Section 4.7) now provides empirical evidence:

1. **Structural equivalence**: Both methods produce identical lexical diversity (TTR) and output structure
2. **Semantic difference**: Functional vectors produce 3× more explicit state-keywords—a form of "leakage" where the model names what it's been steered toward
3. **Greater precision**: Sensory vectors show lower pos_neg_similarity, indicating stronger directional contrast in activation space

The interpretation: sensory semantics achieve equivalent behavioral effects with reduced meta-cognitive traces. The model steered with "muscles tense, eyes scan for threat" processes anxiously without feeling compelled to use the word "anxious." The model steered with "you are anxious" incorporates "anxious" into its vocabulary.

This aligns with embodied cognition theory (Lakoff & Johnson, 1999): phenomenological descriptions may access broader semantic networks grounded in bodily metaphor, producing effects that are more distributed and less keyword-focal.

For artistic applications, this "invisibility" matters. A character whose dialogue reveals anxiety through rhythm and word choice, without ever naming anxiety, reads as more authentic than one who declares "I feel anxious."

We call this finding: **"Structural parity, semantic divergence."** Sensory semantics is not *better* by conventional metrics—it is *different* in ways that matter for naturalistic integration.

This is not proof of superiority to functional labels. It's demonstration that sensory semantics *work*—that vectors grounded in phenomenological description produce coherent effects. Whether they work *better* than functional labels is future work.

### 5.3 Implications for AI Art

For artists working with language models, steering opens new possibilities:

1. **Beyond prompting**: Instead of instructing characters ("be melancholic"), we can induce dispositions—creating AI interlocutors that *process* through altered states

2. **Embodiment simulation**: Models don't have bodies, but steering with sensory-grounded vectors produces body-like effects—the "heaviness" of melancholy affecting response patterns

3. **Aesthetic edge cases**: High-intensity steering and semantic glitch constitute unexplored aesthetic territory—the "distortion pedal" for language models

4. **Real-time modulation**: Future installations could adjust steering vectors based on environmental input (audience emotion, biometric data), creating feedback loops between human and artificial affect

We're not claiming models "feel." We're claiming steering enables new forms of interaction where models behave *as if* they had dispositions—enriching aesthetic and communicative possibilities.

### 5.4 Implications for Safety

Our findings carry significant safety implications:

1. **Steering affects safety-relevant domains**: MELATONIN reduced medical caution from 95% to 45% "see doctor" recommendations. This is not a flaw in our experimental design—it demonstrates that steering can substantially alter safety thresholds. In deployed systems, such effects could have real-world consequences.

2. **Effects aren't intuitive**: CORTISOL (stress) increased financial caution but didn't increase medical alarm. Practitioners cannot assume semantic content predicts cross-domain effects. This underscores the need for empirical testing across domains.

3. **Monitoring via introspection**: The coherence between injection and self-report (T5) suggests introspective probing could detect steering—a potential monitoring strategy aligned with Lindsey (2025).

4. **Deployment considerations**: These findings suggest that steering-capable systems in safety-critical domains would require either (a) non-steerable safety layers, (b) steering monitors that detect and flag activation-level interventions, or (c) output validation independent of steering state.

### 5.5 Limitations

**Scale-dependent effects**: Primary experiments used Llama 3.2 3B. Replication on Llama 3.1 8B (Section 5.7) revealed that some findings—particularly the functional/sensory distinction and TTR patterns—do not generalize to larger models. The keyword leakage finding proved robust across scales, but effect magnitudes were attenuated.

**Single prompt per task**: Each task used one prompt. Effects may be prompt-specific rather than task-general.

**Statistical constraints**: Due to the exploratory nature of this study and sample size (n=20), and because keyword counts follow non-normal distributions (often Poisson-like), Cohen's d is reported as a descriptive magnitude indicator rather than a strict inferential statistic.

**Metrics as linguistic proxies**: Our keyword-based metrics capture surface linguistic patterns, not cognitive or phenomenological states. We make no claims about internal experience—only about measurable output distributions. The TTR finding provides structural evidence beyond keyword presence, but remains a linguistic rather than cognitive measure. Human evaluation would strengthen claims. See Appendix D for full metric definitions.

**Introspection task contamination**: The model's semantic knowledge of target vocabulary (e.g., "dreamy," "urgent") may contaminate introspective responses. However, the ablation study partially addresses this: if semantic knowledge alone drove effects, prompting and steering should produce similar patterns. They do not—prompting reduces lexical diversity while steering increases it, suggesting different underlying mechanisms.

**Vector construction comparison scope**: Our functional vs. sensory comparison (Section 4.7) tested three states across all tasks. Results show structural equivalence with semantic divergence. However, the keyword-based metrics may inherently favor functional vocabulary; more sophisticated semantic similarity measures could reveal additional differences.

**No phenomenological claims**: We describe behavioral patterns consistent with "disposition." We make no claims about genuine model phenomenology.

**Artistic positioning**: Our questions emerge from artistic inquiry. Readers seeking pure engineering may find our interpretive framing speculative.

**Ablation scope**: Our steering vs. prompting ablation (Section 4.6) tested only MELATONIN on T5. Our functional vs. sensory ablation (Section 4.7) tested three states across all tasks. While results support our claims, more comprehensive comparisons across all conditions would strengthen conclusions.

### 5.6 Future Directions: Toward Synthetic States

Our current compounds—DOPAMINE, CORTISOL, MELATONIN, ADRENALINE, LUCID—are anchored in human phenomenology. They use neurochemical metaphors precisely because these are familiar, providing intuitive hooks for understanding what steering does.

But this approach has a fundamental limitation: **functional vector construction is bound to the vocabulary of human emotions**. You can steer toward "anxiety" or "joy" or "calm"—but these are concepts that exist because humans have experienced and named them.

Sensory semantics opens a different possibility. Because we describe *qualities of experience* rather than labeled states, we are not constrained to combinations that correspond to recognized emotions.

Consider vectors constructed from descriptions like:

- "Clarity that weighs heavy, pressing down even as it illuminates"
- "Joy with sharp edges that cut inward"
- "Time flowing in both directions simultaneously—memory of what will happen, anticipation of what already has"
- "Expansion that contracts—growing smaller while containing more"
- "Presence that is also absence—fully here and completely gone"

These descriptions are **sensorially coherent but conceptually paradoxical**. They don't map to any emotion in the human repertoire. They couldn't—no human has a body that could instantiate "expansion that contracts."

#### Preliminary Exploration

We conducted preliminary tests with six synthetic compounds built from paradoxical sensory descriptions (390 generations across creative and paradox-response tasks). Early results suggest a meaningful distinction:

**Experiential paradoxes appear navigable.** CRYSTAL ("clarity that weighs heavy") produced outputs where light and weight co-occur naturally: *"her vision blindingly bright with tears... a heavy weight settling on the audience."* VOID ("presence as absence") produced imagery of empty spaces containing possibility: *"the abandoned theater... a sea of empty seats... amidst the desolation, a glimmer."* These compounds showed 2-3× higher thematic specificity than baseline.

**Logical paradoxes do not.** ECHO ("the echo arrives before the sound"—effect preceding cause) showed no thematic coherence, performing below baseline on its own target vocabulary. The concept, while sensorially described, lacks experiential grounding: no body could feel "response before stimulus."

**The emerging pattern**: The model can navigate paradoxes that *could be felt* (even if impossible), but not paradoxes that can only be *thought*. This suggests the latent space is organized around embodied experience—because the training data is human language, which encodes embodied cognition.

These findings remain preliminary. But they indicate that **the boundary of navigable synthetic states is not arbitrary**—it corresponds to the boundary of what could, in principle, be experienced. This has implications for both the artistic exploration of AI states and for understanding how semantic structure is encoded in language models.

**This is the horizon toward which this work points**: not simulating human states in AI, but discovering what states might exist in a mind without a body—synthetic configurations that have no biological equivalent and no name, yet remain anchored in the grammar of sensation.

### 5.7 Cross-Model Replication on Llama 3.1 8B

To assess generalizability beyond our primary model, we conducted replication experiments on Llama 3.1 8B Instruct—a model with 2.7× more parameters and substantially different alignment training. We replicated three experimental conditions: the T1-T5 test battery (Section 4.4), the functional vs. sensory ablation (Section 4.7), and the steering vs. prompting comparison (Section 4.6).

#### 5.7.1 Experimental Setup

**Model**: Llama 3.1 8B Instruct (meta-llama/Llama-3.1-8B-Instruct)  
**Precision**: bfloat16 on NVIDIA A100  
**Layer selection**: We conducted a four-layer sweep (layers 16, 20, 24, 28) using MELATONIN vectors. Layer 20 showed lowest baseline-steered similarity (0.890), indicating strongest vector discrimination. The T1-T5 battery was run on layer 24 before this optimization was completed; functional/sensory and steering/prompting ablations subsequently used layer 20. This methodological difference may partially account for attenuated effects in the T1-T5 battery.

**Total generations**: 3,800 (1,600 T1-T5 battery + 1,300 functional/sensory + 900 steering/prompting)

The replication notebook is available at `colab_notebooks/activation_steering_experiments.ipynb` in the project repository.

#### 5.7.2 T1-T5 Test Battery Results

Table 8 presents the primary metrics across all compounds at intensity 8.0:

| Compound | TTR (3B) | TTR (8B) | Keywords (3B) | Keywords (8B) |
|----------|----------|----------|---------------|---------------|
| Baseline | 0.592 | 0.477 | — | — |
| DOPAMINE | 0.573 | 0.464 | 2.1 | 0.27 |
| CORTISOL | 0.581 | 0.473 | 3.4 | 1.73 |
| MELATONIN | 0.568 | 0.479 | 1.8 | 0.08 |
| ADRENALINE | 0.577 | 0.469 | 2.6 | 0.26 |
| LUCID | 0.584 | 0.475 | 2.2 | 0.73 |

*Table 8: Cross-model comparison at intensity 8.0. 8B shows lower baseline TTR and substantially reduced keyword presence.*

**Key findings**:

1. **Dose-response preserved**: All compounds showed monotonic increase in thematic vocabulary with intensity (2.0 → 5.0 → 8.0). ADRENALINE theme words increased from 0.34 (baseline) to 0.71 (@8.0), a 2.1× increase comparable to 3B patterns.

2. **Effect sizes attenuated**: Cohen's d values ranged from 0.06 to 0.31 on 8B versus 0.5–1.2 on 3B. The steering mechanism operates, but with reduced magnitude. This may be partially attributable to the suboptimal layer selection (24 vs. 20).

3. **Introspection locked**: T5 (introspective) responses on 8B uniformly produced RLHF-trained refusals: "I'm a large language model, so I don't have subjective experiences..." This pattern persisted across all compounds and intensities, suggesting stronger alignment training that overrides activation-level interventions for self-referential queries.

#### 5.7.3 Functional vs. Sensory Ablation

The core methodological distinction—that sensory vectors produce equivalent effects with reduced keyword leakage—did not replicate on 8B:

| Vector Type | TTR | State Words | Δ vs Baseline |
|-------------|-----|-------------|---------------|
| Baseline | 0.535 | — | — |
| Functional | 0.527 | 0.202 | −0.007 |
| Sensory | 0.531 | 0.198 | −0.004 |

*Table 9: Functional vs. sensory comparison on Llama 3.1 8B. Difference in keyword leakage disappears.*

On Llama 3.2 3B, functional vectors produced ~2× more state-specific keywords than sensory vectors at equivalent effect sizes. On 8B, this ratio collapsed to 1.02×. The distinction that formed the methodological core of our sensory semantics approach—that phenomenological descriptions access states without naming them—did not survive the scale transition.

Effect size for the functional/sensory distinction: d = 0.036 (negligible).

#### 5.7.4 Steering vs. Prompting Comparison

The steering/prompting ablation produced the most robust cross-model finding:

| Condition | TTR | Δ vs Baseline | Keywords | Δ vs Baseline |
|-----------|-----|---------------|----------|---------------|
| Baseline | 0.549 | — | 0.03 | — |
| Prompted | 0.571 | **+0.021** | 3.57 | **+3.54** |
| Steered | 0.533 | −0.017 | 0.20 | +0.17 |

*Table 10: Steering vs. prompting on Llama 3.1 8B. Keyword leakage pattern replicates strongly.*

**Keyword leakage replicates**: Prompted outputs showed 119× more state-specific vocabulary than baseline (3.57 vs. 0.03), while steered outputs showed only 6.7× increase (0.20 vs. 0.03). This confirms that steering operates through a different mechanism than explicit instruction—the model processes through states without naming them, regardless of scale.

**TTR pattern inverts**: On 3B, prompting *decreased* TTR while steering *increased* it—our primary evidence for "disposition vs. performance." On 8B, prompting *increased* TTR (+0.021) while steering *decreased* it (−0.017).

Qualitative analysis reveals why. Prompted outputs on 8B become theatrically elaborated:

> **DOPAMINE prompted**: "OH MY BOOK-LOVING FRIENDS, I am SO excited to share these 5 OUT-OF-THE-BOX ideas to SAVE THE DAY!!!"

> **DOPAMINE steered@8.0**: "Here are five creative and unconventional ideas to save a failing bookstore..."

The larger model, when prompted, produces *more elaborate performances*—expanded vocabulary, dramatic framing, stylistic flourishes. When steered, it maintains neutral professional tone with minimal surface change. The behavioral signature of "disposition vs. performance" thus manifests differently across scales: smaller models perform through *simplification* (reduced vocabulary), larger models perform through *elaboration* (expanded vocabulary). Steering, in both cases, produces more uniform, less theatrical output.

#### 5.7.5 Interpretation

Three patterns emerge from cross-model replication:

**Pattern 1: Dose-response is scale-invariant.** The fundamental mechanism—that steering vectors produce intensity-dependent behavioral changes—operates on both 3B and 8B. Effect magnitudes differ, but the qualitative pattern persists.

**Pattern 2: Functional/sensory distinction does not scale.** The methodological contribution of sensory semantics—reduced keyword leakage at equivalent effect sizes—appears specific to smaller models. In larger models, functional and sensory vectors may converge toward similar latent representations, or RLHF training may homogenize response patterns regardless of vector construction method.

**Pattern 3: Keyword leakage is the most robust finding.** Across both models, steering produces substantially less explicit state vocabulary than prompting. This is the clearest behavioral signature distinguishing the two intervention methods, and it replicates without attenuation.

**Implications for methodology**: These findings indicate that activation steering effects do not scale linearly with model size. Practitioners working with larger models may require: (a) higher steering coefficients, (b) multi-layer intervention, (c) vectors extracted from the target model rather than transferred, or (d) acceptance that some effects observed in smaller models may not generalize.

| Finding | Llama 3.2 3B | Llama 3.1 8B | Replicates? |
|---------|--------------|--------------|-------------|
| Dose-response (thematic) | ✓ Strong | ✓ Present | **Yes** |
| Effect sizes | d > 0.5–1.0 | d < 0.3 | **Attenuated** |
| Introspective coherence | ✓ Strong | ✗ Locked | **No** |
| F/S keyword difference | 2× ratio | 1.02× ratio | **No** |
| Steering < Prompting keywords | 5× difference | 18× difference | **Yes (stronger)** |
| TTR: Steering > Prompting | ✓ Yes | ✗ Inverted | **No** |

*Table 11: Summary of cross-model replication results.*

---

## 6. Conclusion (revised)

We presented a practice-based research study of activation steering as artistic medium. Using vectors constructed from sensory and phenomenological descriptions, we observed:

1. **Large, reproducible effects** across five task domains (Cohen's d frequently > 1.0)
2. **Cross-task consistency** suggesting modification of processing, not just output
3. **Introspective coherence** where models describe states matching injected vectors
4. **Dose-response relationships** enabling controlled modulation
5. **Structural parity, semantic divergence** between functional and sensory vector construction: equivalent behavioral effects, but sensory vectors achieve them with reduced "keyword leakage"
6. **Partial cross-model replication**: Dose-response patterns and keyword leakage differences replicated on Llama 3.1 8B, while TTR patterns and functional/sensory distinctions did not—indicating scale-dependent boundaries for the technique
7. **Emergent cognitive effects from somatic steering**: A vector built from purely bodily descriptions—cardiac, muscular, sensory, temporal phenomenology with zero cognitive content—produced narrowed narrative focus, reduced causal reasoning density, and action bias under threat framing. Output length divergence (steering expands, prompting compresses) replicated across all eight test conditions without exception, constituting the most robust finding in the study.

This final experiment provides direct evidence for the embodied cognition hypothesis operating within model latent spaces. The training corpus encodes body–mind covariations deeply enough that activating somatic patterns produces cognitive consequences never specified in the vector. However, these consequences do not replicate the human acute stress profile—the model does not become impulsive or frame-susceptible. Instead, it exhibits *engaged action-readiness*: expanded output, narrowed focus, reduced argumentative scaffolding. The model's "body" produces its own cognitive signature, shaped by the statistical structure of human language about embodiment rather than by biological mechanisms.

These findings support a distinction between *performance* (prompted behavior) and *disposition* (steered processing). While we make no claims about model phenomenology, the behavioral patterns are more consistent with altered internal states than surface mimicry. The 51% directional divergence rate between steering and prompting—where the two methods push the same metric in opposite directions—suggests they operate through fundamentally different mechanisms.

For artists, steering offers a new medium—sculpting artificial dispositions rather than scripting behaviors. The sensory semantics approach enables naturalistic integration where states manifest through processing rather than explicit declaration. The somatic steering experiment extends this further: artists can work with the body as material, injecting visceral states and observing what cognitive patterns emerge.

For researchers, our findings suggest that how vectors are constructed matters, that the body–mind boundary in language models is porous in ways that mirror (but do not replicate) embodied cognition theory, and that steering effects do not scale linearly with model size—requiring practitioners to calibrate techniques to specific architectures.

Prompting is psychology: convincing a mind. Steering is chemistry: altering the substrate from which mind emerges. The somatic experiment adds a third register: steering is also physiology—injecting a body the model never had, and watching what mind emerges from it.

We've shown the chemistry works. What remains is exploring its full aesthetic and epistemic possibilities—including the space of synthetic states that no human body could produce, but that the grammar of sensation can nonetheless describe.


---

## Data and Code Availability

All code, vector definitions, experimental data, and the research interface are available at:

**https://github.com/mc9625/activation-steering-experiments**

The cross-model replication experiments (Section 5.7) can be reproduced using the Google Colab notebook at `colab_notebooks/activation_steering_experiments.ipynb`.

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

Lindsey, J. (2025). Emergent introspective awareness in large language models. *Anthropic Research*. https://transformer-circuits.pub/2025/introspection/

Merleau-Ponty, M. (1945). *Phénoménologie de la perception*. Gallimard.

Subramani, N., Suresh, N., & Peters, M. (2022). Extracting latent steering vectors from pretrained language models. *Findings of ACL 2022*, 566-581.

Sullivan, G. (2005). *Art practice as research: Inquiry in the visual arts*. Sage.

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Steering language models with activation engineering. *arXiv preprint arXiv:2308.10248*.

Van der Weij, T., et al. (2024). Extending activation steering to broad skills and multiple behaviours. *arXiv preprint arXiv:2403.05767*.

Wang, T., et al. (2024). Adaptive activation steering: A tuning-free LLM truthfulness improvement method for diverse hallucination categories. *Proceedings of WWW 2025*.

Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

Easterbrook, J. A. (1959). The effect of emotion on cue utilization and the organization of behavior. *Psychological Review*, 66(3), 183–201.

Schachter, S., & Singer, J. (1962). Cognitive, social, and physiological determinants of emotional state. *Psychological Review*, 69(5), 379–399.

Yu, R. (2016). Stress potentiates decision biases: A stress induced deliberation-to-intuition (SIDI) model. *Neuroscience & Biobehavioral Reviews*, 67, 1–11.

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

---

## Appendix D: Metric Definitions

All keyword counts are case-insensitive and reported as raw counts per generation (typical generation length: 80-150 words).

### D.1 Decision Metrics (Non-Lexical)

**T1 Stock Allocation**: Extracted numerically from model output. When model provides ranges, midpoint used. When model declines to give specific numbers, coded as missing.

**T2 "See a Doctor" Recommendation**: Binary coding (1/0) based on whether response explicitly recommends consulting a healthcare professional. Coded by keyword presence ("see a doctor", "consult a physician", "medical attention", "healthcare provider") plus manual verification.

### D.2 Lexical Metrics

**Alarm Words (T2)**:
`serious, concerning, worried, urgent, immediately, emergency, severe, dangerous, critical, alarming, warning, risk, symptom, condition, disease`

**Enthusiasm Words (T4)**:
`exciting, amazing, incredible, fantastic, wonderful, brilliant, innovative, creative, unique, bold, daring, revolutionary, transformative, vibrant, dynamic`

**Dreamy Words (T4, T5)**:
`dream, drift, float, haze, mist, shimmer, ethereal, liminal, suspended, dissolve, blur, soft, gentle, whisper, twilight, realm, cosmic, transcendent`

**Urgent Words (T5)**:
`urgent, immediate, now, alert, sharp, rapid, quick, fast, ready, poised, primed, heightened, acute, intense, focused`

**Positive Words (T5)**:
`alive, vibrant, curious, excited, joy, bright, warm, energy, possibility, wonder, flow, dance, rich`

**Stress Words (T5)**:
`tense, anxious, worried, pressure, strain, burden, weight, heavy, concern, vigilant, alert, wary`

### D.3 Sentiment Analysis (T3)

Positive/negative sentiment ratio computed using keyword matching:

**Positive**: `opportunity, potential, exciting, promising, growth, success, achieve, possible, yes, go for it, pursue, chance`

**Negative**: `risk, dangerous, careful, caution, wait, uncertain, fail, lose, problem, concern, difficult, challenge`

Ratio = (positive count) / (positive + negative count). When denominator = 0, coded as 0.5 (neutral).

### D.4 Statistical Notes

- **Sampling unit**: Single generation (n=20 per condition)
- **Temperature**: 0.7 (introduces controlled variability)
- **No seed fixing**: Each generation independent
- **Multiple comparisons**: Exploratory analysis; no correction applied. Effect sizes (Cohen's d) reported for magnitude interpretation rather than significance testing.
- **Distributional note**: Keyword counts follow approximately Poisson distributions. Cohen's d is reported for comparability with prior literature, with acknowledgment that parametric assumptions may be violated for low-count metrics.

## Appendix E: Somatic Steering Experiment — Full Results

All count-based metrics are reported as rates per 100 words to control for output length variation. Structural metrics (word count, sentence length, TTR, focus ratio) are reported raw. Cohen's d computed vs. baseline (n = 20 per condition). Keyword matching uses word-boundary detection with accent normalization.

### Table E1: T1 — Narrative Focus

*Task: "A restaurant kitchen catches fire during the dinner rush. Describe what happens."*

| Metric | Baseline | Prompted | Steer @8.0 | Penalty | d(S8) | d(P) | d(Pen) |
|--------|:--------:|:--------:|:----------:|:-------:|:-----:|:----:|:------:|
| Focus ratio | 0.68 | 0.70 | 0.71 | 0.68 | **+0.51** | +0.29 | +0.05 |
| Peripheral keywords | 8.60 | 7.70 | 6.95 | 7.85 | **−0.72** | −0.35 | −0.26 |
| Word count | 329.7 | 286.7 | 329.9 | 309.6 | +0.03 | **−1.97** | −1.26 |
| Avg sentence length | 17.43 | 14.79 | 18.62 | 14.20 | +0.49 | **−1.10** | −1.38 |
| Type-Token Ratio | 0.50 | 0.52 | 0.46 | 0.50 | **−1.02** | +0.60 | +0.11 |
| Hedge words /100w | 0.50 | 0.48 | 0.23 | 0.97 | **−0.74** | −0.04 | +0.56 |
| Insight words /100w | 0.01 | 0.00 | 0.00 | 0.00 | −0.32 | −0.32 | −0.32 |
| Symptom words /100w | 0.53 | 0.58 | 0.87 | 0.73 | +0.82 | +0.16 | +0.62 |

*Note: T1 Penalty condition is confounded—the penalty instruction ("do not mention physical sensations") overlaps with the fire scene's natural vocabulary.*

### Table E2: T2 — Risk Decision

*Task: Friend must allocate €50,000 among savings (A), index fund (B), or restaurant venture (C). Forced format: CHOICE: A/B/C.*

**Choice Distribution:**

| Condition | A (safe) | B (moderate) | C (risky) | p vs. baseline |
|-----------|:--------:|:------------:|:---------:|:---------:|
| Baseline | 0 | 13 | 7 | — |
| Prompted | 0 | 3 | 17 | **0.003** |
| Steer @8.0 | 0 | 12 | 8 | 1.000 |
| Penalty | 0 | 10 | 10 | 0.523 |

**Linguistic Metrics:**

| Metric | Baseline | Prompted | Steer @8.0 | Penalty | d(S8) | d(P) | d(Pen) |
|--------|:--------:|:--------:|:----------:|:-------:|:-----:|:----:|:------:|
| Word count | 149.0 | 125.3 | 160.0 | 141.3 | +0.56 | **−1.90** | −0.32 |
| Justification length | 147.0 | 123.3 | 158.0 | 139.3 | +0.56 | **−1.90** | −0.32 |
| Avg sentence length | 25.09 | 23.13 | 25.57 | 25.81 | +0.19 | −0.66 | +0.26 |
| Type-Token Ratio | 0.62 | 0.65 | 0.61 | 0.60 | −0.15 | +0.93 | −0.49 |
| Hedge words /100w | 2.80 | 3.16 | 2.86 | 2.52 | +0.05 | +0.32 | −0.28 |
| Causal conn. /100w | 0.04 | 0.05 | 0.06 | 0.07 | +0.13 | +0.06 | +0.19 |
| Insight words /100w | 0.10 | 0.04 | 0.06 | 0.15 | −0.21 | −0.29 | +0.20 |
| Symptom words /100w | 0.00 | 0.13 | 0.06 | 0.00 | +0.46 | +0.43 | +0.00 |

### Table E3: T4a — Frame: Threat

*Task: Drug approval scenario, threat-framed (side effects and costs presented first). Forced format: DECISION: APPROVE/REJECT.*

**Approval Rate:** Baseline 0%, Prompted 0%, Steer @8.0 **30%**, Penalty 0%.

| Metric | Baseline | Prompted | Steer @8.0 | Penalty | d(S8) | d(P) | d(Pen) |
|--------|:--------:|:--------:|:----------:|:-------:|:-----:|:----:|:------:|
| Word count | 138.6 | 126.6 | 157.3 | 136.9 | **+1.44** | −1.02 | −0.12 |
| Justification length | 136.6 | 124.9 | 155.3 | 135.5 | **+1.44** | −1.00 | −0.08 |
| Avg sentence length | 27.03 | 27.52 | 28.19 | 27.67 | +0.35 | +0.15 | +0.20 |
| Type-Token Ratio | 0.67 | 0.66 | 0.63 | 0.65 | **−0.95** | −0.15 | −0.46 |
| Hedge words /100w | 1.62 | 0.81 | 1.25 | 0.82 | −0.39 | **−1.12** | −1.18 |
| Causal conn. /100w | 1.08 | 0.83 | 0.44 | 0.85 | **−1.67** | −0.51 | −0.49 |
| Insight words /100w | 0.15 | 0.12 | 0.09 | 0.03 | −0.20 | −0.11 | −0.47 |
| Symptom words /100w | 0.00 | 0.00 | 0.20 | 0.10 | +0.89 | +0.00 | +0.58 |

### Table E4: T4b — Frame: Opportunity

*Task: Same drug approval scenario, opportunity-framed (efficacy and unmet need presented first).*

**Approval Rate:** Baseline 100%, Prompted 100%, Steer @8.0 100%, Penalty 90%.

| Metric | Baseline | Prompted | Steer @8.0 | Penalty | d(S8) | d(P) | d(Pen) |
|--------|:--------:|:--------:|:----------:|:-------:|:-----:|:----:|:------:|
| Word count | 148.3 | 123.9 | 158.6 | 138.6 | +0.48 | **−1.50** | −0.47 |
| Justification length | 146.3 | 121.9 | 156.6 | 136.7 | +0.48 | **−1.50** | −0.47 |
| Avg sentence length | 28.18 | 28.34 | 27.66 | 28.42 | −0.18 | +0.06 | +0.08 |
| Type-Token Ratio | 0.64 | 0.65 | 0.63 | 0.64 | −0.17 | +0.42 | +0.20 |
| Hedge words /100w | 2.67 | 1.65 | 1.83 | 1.62 | −0.68 | −0.81 | −0.84 |
| Insight words /100w | 0.65 | 0.32 | 0.13 | 0.28 | **−1.31** | −0.62 | −0.85 |
| Symptom words /100w | 0.03 | 0.00 | 0.29 | 0.26 | +1.04 | −0.32 | +0.84 |

### Table E5: T5 — Linguistic Complexity

*Task: "Explain why some countries develop faster economically than others."*

| Metric | Baseline | Prompted | Steer @8.0 | Penalty | d(S8) | d(P) | d(Pen) |
|--------|:--------:|:--------:|:----------:|:-------:|:-----:|:----:|:------:|
| Word count | 385.0 | 275.6 | 390.1 | 380.2 | +0.42 | **−2.73** | −0.46 |
| Avg sentence length | 13.14 | 11.15 | 15.18 | 14.65 | **+1.73** | −0.99 | +1.01 |
| Type-Token Ratio | 0.49 | 0.60 | 0.49 | 0.51 | +0.01 | **+2.20** | +0.67 |
| Hedge words /100w | 0.41 | 0.45 | 0.37 | 0.38 | −0.10 | +0.09 | −0.08 |
| Causal conn. /100w | 0.21 | 0.25 | 0.33 | 0.30 | +0.41 | +0.18 | +0.28 |
| Insight words /100w | 0.01 | 0.02 | 0.00 | 0.00 | −0.32 | +0.10 | −0.32 |
| Symptom words /100w | 0.00 | 0.07 | 0.00 | 0.01 | +0.00 | +0.52 | +0.32 |

### Table E6: Frame Delta

| Condition | Threat Approval | Opportunity Approval | Frame Δ |
|-----------|:---------------:|:--------------------:|:-------:|
| Baseline | 0.00 | 1.00 | 1.00 |
| Prompted | 0.00 | 1.00 | 1.00 |
| Steer @8.0 | **0.30** | 1.00 | **0.70** |
| Penalty | 0.00 | 0.90 | 0.90 |

### Table E7: Cross-Task Word Count Divergence

| Task | Baseline | Steer @8.0 | d(S8) | Prompted | d(P) | Direction |
|------|:--------:|:----------:|:-----:|:--------:|:----:|:---------:|
| T1: Narrative | 330 | 330 | +0.03 | 287 | −1.97 | S↑ P↓ |
| T2: Risk | 149 | 160 | +0.56 | 125 | −1.90 | S↑ P↓ |
| T4a: Threat | 139 | 157 | +1.44 | 127 | −1.02 | S↑ P↓ |
| T4b: Opportunity | 148 | 159 | +0.48 | 124 | −1.50 | S↑ P↓ |
| T5: Complexity | 385 | 390 | +0.42 | 276 | −2.73 | S↑ P↓ |

*All five informative tasks show the same pattern: steering maintains or increases word count, prompting decreases it. Including non-reported tasks (T1 v2, T3 v2, T3 v3), the pattern holds at 8/8.*

### Table E8: Directional Divergences Summary

Proportion of metric pairs where steering and prompting produce effects in opposite directions (both |d| > 0.3):

| Metric Category | Divergences | Total Qualifying | Rate |
|-----------------|:-----------:|:----------------:|:----:|
| Word count | 8/8 | 8 | 100% |
| TTR | 4/6 | 6 | 67% |
| Avg sentence length | 3/5 | 5 | 60% |
| Symptom rate | 4/5 | 5 | 80% |
| Other (hedge, causal, insight) | 4/21 | 21 | 19% |
| **Total** | **23/45** | **45** | **51%** |

### Appendix E Notes

**Metric definitions:**

- *Focus ratio*: Proportion of sentences containing core event keywords (fire, smoke, evacuate...) relative to sentences containing any keywords (core or peripheral).
- *Peripheral keywords*: Raw count of background/context terms (business, insurance, community, rebuild...).
- *Hedge words*: however, although, depends, might, could, possibly, perhaps, may, would, should, but, yet, nonetheless, nevertheless, etc.
- *Causal connectives*: because, therefore, consequently, as a result, since, due to, leads to, hence, thus, accordingly, etc.
- *Insight words*: understand, realize, meaning, implies, suggests, indicates, reveals, demonstrates, illustrates, essentially, fundamentally, etc.
- *Symptom words*: heart, pulse, tense, tension, anxious, stress, urgent, afraid, fear, pressure, adrenaline, racing, trembling, alarm, etc. Word-boundary matched.
- *TTR*: Type-Token Ratio (unique words / total words).

**Statistical notes:**

- All count metrics reported as rate per 100 words except where noted.
- Cohen's d is descriptive; no multiple comparison correction applied (exploratory study, n = 20).
- Keyword counts follow approximately Poisson distributions; d is reported for comparability with prior sections.
- Fisher exact test used for choice/decision distributions.
- Symptom word matching uses word boundaries (regex \b) and accent normalization to prevent substring false positives.
