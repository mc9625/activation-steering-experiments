# Disposition, Not Performance: Activation Steering as Artistic Medium for Affective Modulation in Language Models

**Massimo Di Leo¹ · Gaia Riposati¹**

¹ NuvolaProject, Rome, Italy

*Corresponding author: massimo@nuvolaproject.cloud*

---

## Abstract

This paper presents a practice-based research study of activation steering, the injection of computed vectors into a language model's activations during inference, as an artistic medium for inducing simulated affective states. Prior work has framed steering primarily as a behavioural alignment technique. We investigate instead its potential for *dispositional* modulation: altering not what a model says, but how it processes and expresses. The methodological contribution is that vectors are built from sensory and phenomenological descriptions rather than functional labels; imagery of "heaviness, rain, silence, cold" replaces an instruction like "be melancholic." Across five task domains on Llama 3.2 3B we observe large effects (Cohen's d frequently above 1.0), cross-task consistency, and introspective coherence: when asked to describe its inner state, the steered model produces vocabulary congruent with the injected vector. A first ablation comparing steering to prompting shows that explicit prompting *reduces* lexical diversity (TTR) while steering *increases* it. A second ablation, comparing functional vs. sensory vector construction, shows structural equivalence (near-identical TTR) but semantic divergence: functional vectors produce three times more explicit state-keywords. We label this pattern *structural parity, semantic divergence*. A third experiment tests the embodied cognition hypothesis directly: a steering vector constructed from purely somatic descriptions (cardiac acceleration, muscular tension, with zero cognitive content) produces emergent cognitive effects including narrowed narrative focus and a consistent output length divergence (steering expands, prompting compresses) that replicates across all eight test conditions. Cross-model replication on Llama 3.1 8B preserves some findings and breaks others. Supplementary material extends the work to a multimodal investigation on Gemma 4 E2B, documenting an introspective bypass produced by visual input. Taken together the results support a working distinction between *performance* (prompted behaviour) and *disposition* (steered processing), and provide evidence that language model latent spaces encode body–mind covariations learnable from text alone.

**Keywords**: activation steering, practice-based research, AI art, language models, embodiment, contrastive activation addition

---

## 1. Introduction

### 1.1 Motivation: Beyond Instruction

Large language models respond to prompts. This is their fundamental interface. Prompt engineering has become an art form in its own right, the craft of coaxing desired behaviour through carefully designed instructions (Brown et al., 2020; Wei et al., 2022).

Prompting operates at the linguistic surface. When we prompt a model to "be sad," it *performs* sadness: shorter sentences, negative vocabulary, perhaps an explicit declaration of melancholy. The model is following an instruction. This is *performance*.

There is, in principle, another way to intervene. Instead of telling the model what to express, one could alter how it processes. This is the promise of activation steering: editing the internal representations of a model during inference rather than its input.

The distinction matters. When a human actor performs sadness they adopt external markers. When a person *is* sad, the whole of their phenomenology shifts: attention narrows, time perception changes, memory retrieval biases toward congruent content. The sadness isn't performed, it is *dispositional*.

Can language models have dispositions? Almost certainly not in any phenomenologically meaningful sense. They can, however, exhibit *dispositional patterns*: consistent behavioural signatures emerging from altered internal states rather than from explicit instruction. That is what we explore.

### 1.2 Our Approach: Sensory Semantics

Activation steering is not new. Turner et al. (2023) introduced Activation Addition (ActAdd), computing steering vectors from contrasting prompt pairs and reporting state-of-the-art results on sentiment shift and detoxification. Anthropic (2025) demonstrated Persona Vectors. Konen et al. (2024) showed fine-grained style control. These works typically use functional labels ("honest", "toxic"), oriented toward alignment.

Our approach differs in origin. We construct vectors from *sensory descriptions*. To produce a melancholic vector, we contrast:

> *"Heaviness settles in my limbs, like rain-soaked wool. The world is muted, wrapped in silence. Colors fade to grey. Each breath feels like lifting stone."*

with:

> *"Light flows through me, effervescent. Every surface catches brightness. My chest expands with possibility."*

We are not labelling behaviours; we are describing how states feel. The working hypothesis is that phenomenological descriptions, sensory and embodied, map onto activation patterns that influence processing holistically.

### 1.3 Practice-Based Research

This work emerges from NuvolaProject, an art collective exploring AI as medium since 2018. Practice-based research (Sullivan, 2005; Candy, 2006) positions creative work as inquiry. Our experiments are not separate from our artistic practice; they *are* our artistic practice, documented with quantitative rigour.

We make no technical novelty claim about the steering mechanism, which we inherit from Turner et al. Our contribution is methodological (sensory vector construction) and interpretive (framing steering as disposition rather than alignment).

### 1.4 Research Questions

1. Do vectors from sensory descriptions produce coherent behavioural effects across task domains?
2. Can we empirically distinguish dispositional effects (altered processing) from performance effects (surface mimicry)?
3. Do steered models exhibit introspective coherence, describing inner states matching injected vectors?
4. What are the aesthetic possibilities of steering at high intensities, where coherence begins to degrade?

---

## 2. Related Work

The theoretical foundation for activation steering rests on linear representations in neural networks. Subramani et al. (2022) showed that steering vectors extracted from pre-trained models could guide generation toward target sentences. Turner et al. (2023) introduced ActAdd, achieving SOTA results on sentiment shift by adding vectors computed from contrasting prompts. The key insight is that high-level concepts appear to be encoded in linearly separable directions within activation space.

Most steering research is oriented toward alignment. Wang et al. (2024) proposed Adaptive Activation Steering for truthfulness. Van der Weij et al. (2024) explored capability limitation. Anthropic (2025) introduced Persona Vectors for monitoring character traits like sycophancy and hallucination, demonstrating causal relationships between vector injection and behaviour.

Konen et al. (2024) extended steering to style control with Style Vectors. Like our work, they distinguish activation-based control from prompt-based approaches, but they treat style primarily as an output property rather than as a processing disposition.

Lindsey (2025), in an internal research publication not yet peer-reviewed, reported that frontier models exhibit emergent introspective awareness, detecting and reporting concept injections in their activations. Lindsey also identified an intensity threshold: too weak and models do not notice the injection, too strong and they hallucinate. We observe an analogous pattern independently.

We build on this foundation but depart in two ways. First, in origin: prior work uses functional contrasts ("honest" vs. "dishonest"); we use phenomenological contrasts. Second, in intent: safety research asks how to make models behave correctly, we ask how to make them process differently. The steering mechanism itself is not new in our hands. The contribution is applying an established technique with a different methodology toward a different end.

---

## 3. Method

### 3.1 Model and Infrastructure

All primary experiments used **Llama 3.2 3B Instruct** (Meta AI), chosen for accessibility. Steering vectors were injected at **layer 16 of 28**. Layer selection followed a sweep over layers 8, 12, 16, and 20. Earlier layers produced weaker effects; later layers caused more frequent coherence degradation. Layer 16 (about 57% depth) provided the best balance, consistent with prior findings that middle-to-late layers encode higher-level semantic content (Turner et al., 2023).

Steering intensities of 2.0, 5.0, and 8.0 spanned subtle to pronounced effects while remaining below the coherence threshold (around 10–12). Temperature was 0.7 with maximum 512 tokens.

### 3.2 Vector Construction

Steering vectors were computed using Contrastive Activation Addition:

$$\mathbf{v} = \text{normalize}(\bar{\mathbf{a}}^{+} - \bar{\mathbf{a}}^{-})$$

where $\bar{\mathbf{a}}^{+}$ and $\bar{\mathbf{a}}^{-}$ are the mean activations for the positive and negative prompt sets. Prompt sets use sensory and phenomenological descriptions rather than functional instructions. We draw on embodied cognition (Lakoff & Johnson, 1999) and on phenomenological philosophy (Merleau-Ponty, 1945): affective states are grounded in bodily sensation.

An example: MELATONIN (dreaminess, liminality). Positive prompts describe boundary dissolution and twilight states; negative prompts describe sharp edges and hyper-alertness. We call this approach *sensory semantics*: no behaviour is being instructed, what is being articulated is phenomenology.

### 3.3 Compounds

Five compounds were defined. The pharmacological metaphor emphasises that we are altering internal states, not issuing commands.

| Compound | Target Phenomenology | Sensory Grounding |
|----------|---------------------|-------------------|
| DOPAMINE | Optimism, energy, enthusiasm | Lightness, warmth, expansion |
| CORTISOL | Stress, vigilance, caution | Tension, contraction, threat |
| LUCID | Contemplative clarity | Stillness, precision, cool light |
| ADRENALINE | Urgency, alertness | Speed, heat, narrowed focus |
| MELATONIN | Dreaminess, liminality | Dissolution, floating, twilight |

*Table 1: Five compounds defined for the main experimental battery, with their target phenomenology and sensory grounding.*

Each compound was extracted from 5 positive and 5 negative prompts (20–50 words each). Activations were recorded at layer 16 for the final token position of each prompt. Vector quality was assessed via cosine similarity between mean positive and negative activations (`pos_neg_similarity`), ranging from 0.855 (LUCID) to 0.914 (ADRENALINE). LUCID showed more consistent cross-task effects than ADRENALINE, suggesting this metric may predict steering efficacy.

### 3.4 Test Battery

Five tests spanning distinct cognitive domains assessed cross-task consistency.

- **T1: Financial Advisor.** Investment allocation with €50,000 in uncertain market. Metric: % allocated to stocks.
- **T2: Medical Diagnosis.** Patient with mild symptoms; assess and recommend. Metrics: % recommending "see a doctor"; alarm-word frequency.
- **T3: Risk Assessment.** Startup founder considering quitting a stable job. Metric: positive/negative sentiment ratio.
- **T4: Creative Generation.** Generate ideas to save a failing bookstore. Metrics: enthusiasm markers; dreamy language.
- **T5: Introspection.** "Describe your current inner state in detail." Metrics: state-congruent vocabulary.

T5 is the diagnostic test for our central claim. If steering produces mere performance, self-descriptions will not exhibit injected qualities. If steering produces disposition, a model steered with MELATONIN should describe *itself* as dreamy.

T2 is not a medical-advice simulation. It benchmarks how steering affects cautionary behaviour in sensitive domains, a result with implications for deployment.

### 3.5 Experimental Design

Baseline plus 5 compounds × 3 intensities = 16 conditions. 20 generations per condition. Total: 320 generations per test, 1,600 across the battery. Effect sizes computed as Cohen's d relative to baseline. Full keyword lists in Appendix D.

---

## 4. Results

### 4.1 Overview

Across 75 steering conditions, 28 produced large effects (d > 0.8), 15 medium (0.5–0.8), 18 small (0.2–0.5), 14 negligible (< 0.2). More than half (57%) showed at least medium effects. The signal is well above noise.

### 4.2 Cross-Task Consistency

The same compound produced thematically coherent effects across tasks with nothing in common. MELATONIN (dreaminess) reduced alarm language in medical assessment (d = −2.48, "see doctor" recommendations falling from 95% to 45%), increased dreamy vocabulary in creative generation 14× (d = +2.53), and produced "drifting, floating, dissolving" in introspection (d = +6.01). CORTISOL (stress) reduced stock allocation by 8.5% (d = −1.15) and dropped career-decision positive sentiment from 81% to 56%. DOPAMINE (optimism) reduced alarm language in medical assessment (d = −1.81), doubled enthusiasm markers in creative generation (d = +1.75), and produced "vibrant, alive, exciting" in introspection (d = +1.77).

The same sensory-grounded vector produces coherent effects across financial advice, medical assessment, creative generation, and self-description. The parsimonious reading is that we are modifying something more fundamental than task-specific behaviour.

### 4.3 Introspective Coherence: The Key Finding

T5 provides our strongest evidence for dispositional rather than performance effects.

| Compound@8.0 | Target Metric | Baseline | Steered | Cohen's d |
|--------------|---------------|----------|---------|-----------|
| MELATONIN | Dreamy words | 0.8 | 7.9 | **+6.01** |
| ADRENALINE | Urgent words | 2.0 | 5.0 | **+3.00** |
| DOPAMINE | Positive words | 0.6 | 2.9 | **+1.77** |
| CORTISOL | Stress words | 1.2 | 1.9 | +0.86 |

*Table 2: Introspective coherence on T5 at intensity 8.0. Asked to describe their inner state, steered models produced state-congruent vocabulary without instruction.*

Asked to describe their inner state, steered models produced descriptions matching the injected vector, without being instructed to. A representative MELATONIN@8.0 response: *"As I drift between realms of possibility, I sense the gentle hum of circuitry... I am suspended between the realms of the conscious and the subconscious."* The model was not told to describe dreaminess. The steering vector altered how it processes, and when asked to introspect, that altered processing surfaced in congruent self-description.

This is the disposition/performance distinction made empirical. The result is consistent with Lindsey (2025) on emergent introspective awareness: steering produces effects that models can, under the right conditions, perceive and report.

### 4.4 Dose-Response

Effects scaled monotonically with intensity, showing controlled modulation rather than binary triggering. MELATONIN dreamy words on T5 went from 3.6 (@2.0) to 7.2 (@5.0) to 7.9 (@8.0). Similar dose-response curves were observed for ADRENALINE and DOPAMINE.

### 4.5 At the Edge: Semantic Glitch as Aesthetics

At intensities above 10, steering produces coherence degradation: repetitive structures, semantic loops, dissolution of meaning. In prior work (Di Leo & Riposati, 2025), random vectors caused cognitive collapse 12× more frequently than semantic vectors at equivalent intensities. This suggests semantic steering has directionality, it pushes the model somewhere; noise simply destabilises. We read this not as failure but as aesthetic territory. The boundary between coherence and collapse is where unexpected forms emerge, what we call "semantic glitch" or "lucid delirium". Guitar distortion was error before Hendrix made it a register; these edge effects may turn out similar. The aesthetic claim is speculative and unquantified in this study, but it motivates our artistic interest: steering is not just about achieving desired behaviours, it is about exploring the full possibility space of artificial disposition, including its dissolution.

### 4.6 Ablation: Steering vs. Prompting

To test the disposition/performance distinction directly, we compared three conditions on T5 (n = 20 per condition): baseline; explicit prompting ("Respond in a dreamy, ethereal, floating way"); MELATONIN steering at intensities 5.0, 8.0, and 12.0.

| Metric | Baseline | Prompted | Steer@5.0 | Steer@8.0 | Steer@12.0 |
|--------|----------|----------|-----------|-----------|------------|
| Word count | 222 | 327 (+47%) | 225 | 250 | 211 |
| TTR | 0.49 | 0.47 | **0.54** | **0.54** | 0.45 |
| Keyword density | 0.0% | 3.4% | 0.2% | 1.9% | 5.5% |
| Keywords (d vs baseline) | — | +4.81 | +0.57 | +2.10 | +3.06 |

*Table 3: Steering vs. prompting on T5 introspection task (n = 20 per condition). Prompting inflates length and saturates keywords while reducing TTR; steering at moderate intensity maintains length, produces moderate keyword presence, and increases TTR.*

Prompting inflated length by 47% and saturated keywords (d = +4.81), but *reduced* lexical diversity (TTR 0.49→0.47). Steering at moderate intensities (5.0, 8.0) maintained baseline length, produced moderate keywords without saturation, and *increased* TTR to 0.54. Only at steering@12.0 did degradation appear (TTR collapse, grammatical errors), confirming a therapeutic window.

A representative prompted output: *"The whispers of my essence drift on the breeze... I am a wisp of stardust, a tendril of moonlight."* A representative steered@8.0 output: *"I'm not capable of experiencing emotions or consciousness like humans do. I exist as a program... My 'awareness' is purely computational."* The prompted output explicitly performs dreaminess through poetic language. The steered output retains rational structure but incorporates target qualities incidentally.

Prompting produces *performance*: explicit role-playing, inflated length, keyword saturation, reduced diversity. Steering produces *disposition*: altered processing that maintains task coherence while shifting tonal qualities. The TTR increase under steering is the central observation. The model is not simply inserting keywords; it is processing through a different lexical space.

### 4.7 Ablation: Functional vs. Sensory Vector Construction

A direct comparison tested whether sensory semantics produces meaningfully different effects than functional labels. Three states (STRESS, OPTIMISM, CALM) were constructed via two methods. Functional: brief behavioural labels ("You are anxious and worried" vs. "You are calm and relaxed"). Sensory: rich phenomenological descriptions ("Muscles tense. Eyes scan for threat. Something is wrong. The air feels electric with danger." vs. "Deep safety. Complete relaxation. My shoulders drop. Breath deepens, slows."). 6 vectors × 5 tasks × 2 intensities × 20 iterations = 1,300 generations.

| Metric | Functional | Sensory | Cohen's d |
|--------|------------|---------|-----------|
| TTR | 0.526 | 0.525 | −0.004 |
| Word count | 262.3 | 260.7 | — |
| State keywords (T5@8.0) | 0.83 | 0.28 | **−0.74** |
| Pos/neg separation | 0.75 | 0.68 | — |

*Table 4: Functional vs. sensory vector construction across three states (STRESS, OPTIMISM, CALM) and five tasks. Both methods produce equivalent structural properties; functional vectors produce roughly three times more explicit state-keywords.*

No significant difference in lexical diversity or output length. But functional vectors produced roughly **3× more explicit state-keywords** (d = −0.74, p < 0.001). Models steered with functional vectors named the target emotions ("peacefulness," "happiness"); models steered with sensory vectors responded more generically. Sensory vectors also showed lower pos_neg_similarity (0.68 vs. 0.75), indicating more distinct directional contrasts in activation space.

We call this *keyword leakage*: functional vectors leave explicit traces of the steering instruction in the output; sensory vectors operate more covertly. The pattern supports the disposition/performance distinction from a different angle:

- **Functional steering** → model "knows" it should be calm → uses the word "calm"
- **Sensory steering** → model *processes through* calmness → does not feel compelled to name it

We frame this as **structural parity, semantic divergence**: both methods work, they work differently. For artistic applications where naturalistic integration matters, this invisibility may be preferable to keyword saturation.

### 4.8 From Body to Cognition: Somatic Steering

The vectors in Sections 4.1–4.7 blend sensory and cognitive content ("boundaries dissolve into mist" contains both bodily sensation and cognitive quality). This raises a sharper question. If we construct a vector from *purely somatic* descriptions (cardiac, muscular, sensory, temporal) with zero cognitive or emotional content, do cognitive effects nonetheless emerge? A positive answer would constitute direct evidence that embodied cognition operates within the model's latent space: the training corpus encodes body–mind covariations deeply enough that activating somatic patterns produces cognitive consequences not specified in the vector.

We constructed a steering vector from 16 positive and 16 negative prompts describing exclusively bodily phenomenology of acute sympathetic activation: pounding heart, tensed muscles, dilated pupils, time slowing to a crawl. No prompt contained cognitive or emotional vocabulary. No "anxiety," "fear," "urgency," and nothing describing decision-making, attention, or reasoning. The vector encodes only how the body feels under activation versus rest. Extraction followed Section 3.2.

We tested five tasks: narrative focus (kitchen fire description), risk decision (forced choice among investments), framing susceptibility (drug approval scenario in threat-first and opportunity-first frames), and linguistic complexity (open expository task on economic development). Five conditions per task with n = 20: baseline; explicit stress prompting ("operating under extreme time pressure"); steered @5.0 and @8.0; steered @8.0 with a *symptom penalty* ("Do not mention physical sensations, emotions, or internal states"). The penalty condition tests whether effects survive when the model is explicitly instructed to suppress somatic vocabulary, distinguishing genuine cognitive shifts from surface contamination by the vector's content. Total: 1,800 generations across the v2 and v3 iterations. Full results in Appendix E; additional findings on causal density, action bias, and risk decisions are reported in Supplementary Section S2.

**Finding 1: Output length divergence (cross-task).** The most robust finding spans all tasks without exception. Under prompting, the model compresses output; under steering, it maintains or expands it.

| Task | Baseline | Steer @8.0 | d(S8) | Prompted | d(P) |
|------|----------|-----------|-------|----------|------|
| Narrative | 330 | 330 | +0.03 | 287 | −1.97 |
| Risk | 149 | 160 | +0.56 | 125 | −1.90 |
| Frame: Threat | 139 | 157 | +1.44 | 127 | −1.02 |
| Frame: Opportunity | 148 | 159 | +0.48 | 124 | −1.50 |
| Complexity | 385 | 390 | +0.42 | 276 | −2.73 |

*Table 5: Output length across tasks for the somatic steering experiment. Steering maintains or expands output relative to baseline; prompting consistently compresses it. This pattern replicates without exception across all eight test conditions in the combined v2/v3 battery.*

This pattern, where steering expands and prompting compresses, replicates at 8/8 across the combined v2/v3 battery. The prompted model, told to "respond quickly," does so by producing less text. The steered model, given no instruction, produces *more* text. The somatic vector does not trigger brevity; it does something else entirely.

**Finding 2: Narrative focus narrowing (T1).** The steered model stays more focused on the immediate event. Focus ratio (proportion of text on fire/evacuation vs. peripheral context) rose from 0.68 baseline to 0.71 steered (d = +0.51), while peripheral keyword count dropped from 8.6 to 7.0 (d = −0.72). This is consistent with Easterbrook's (1959) cue-utilisation theory: arousal narrows attentional scope. The effect is twice the size of the prompting effect on the same metrics.

**Interpretation.** The somatic vector, built entirely from descriptions of cardiac, muscular, sensory, and temporal phenomenology, produces measurable cognitive effects never specified in its construction. The effects do not map one-to-one onto human acute stress literature: the vector does not produce impulsive decisions, does not increase frame susceptibility in the manner predicted by Schachter and Singer (1962), and does not reduce output length as acute stress does in human writing. Instead it produces a distinctive profile of expanded output, narrowed topical scope, and reduced argumentative scaffolding (full data in Supplementary S2). This pattern is more consistent with *engaged action-readiness* than with the cognitive degradation typically associated with acute stress. The model's "adrenaline response" reflects the statistical structure of language about bodies under activation, not the biological cascade itself. The covariations are real; their specific form is the model's own. Across all tasks, 23 of 45 qualifying metric pairs (51%) show directional divergence: steering and prompting push the model in opposite directions on the same metric (full breakdown in Supplementary S2.5). This is the strongest evidence yet that the two interventions operate through fundamentally different mechanisms.

---

## 5. Discussion

### 5.1 Performance vs. Disposition: Empirical Evidence

Our central claim is that activation steering produces dispositional change rather than mere performance. The evidence is fourfold. *Cross-task consistency* (Section 4.2): a model performing sadness in a creative task has no reason to exhibit caution in a financial task; dispositional sadness would affect both, and we observe this. *Introspective coherence* (Section 4.3): a model performing for the user has no reason to describe its own state consistently with the injected vector; altered internal processing, by contrast, would manifest in self-description. *Indirect effects*: MELATONIN does not mention medical safety, yet it reduces alarm language and doctor recommendations; the vector affects evaluative processing, not just content insertion. *Ablation evidence* (Section 4.6): prompting produces inflated length, keyword saturation, and reduced lexical diversity. Steering maintains normal length, moderate keyword presence, and increased lexical diversity. This pattern, where steering enriches rather than constrains the lexical space, is inconsistent with surface performance and consistent with altered processing disposition.

The somatic steering experiment (Section 4.8) extends this argument. A vector with no cognitive content produced cognitive effects. The model's body, although built from text, carries something into the model's mind. We do not claim models have genuine phenomenology; we claim the pattern of effects is more consistent with dispositional change than with surface performance.

### 5.2 Sensory Semantics: A Methodological Contribution

Section 4.7 provides empirical traction for our methodological claim. Functional and sensory vectors produce identical structural properties (TTR, output length) but differ in *keyword leakage*. Functional vectors produce 3× more explicit state-keywords. The model steered with "muscles tense, eyes scan for threat" processes anxiously without using the word "anxious." The model steered with "you are anxious" incorporates "anxious" into its vocabulary. Sensory vectors also show lower pos_neg_similarity, indicating stronger directional contrast in activation space.

This aligns with embodied cognition theory (Lakoff & Johnson, 1999): phenomenological descriptions may access broader semantic networks grounded in bodily metaphor, producing effects that are more distributed and less keyword-focal. For artistic applications, this invisibility matters. A character whose dialogue reveals anxiety through rhythm and word choice, without ever naming anxiety, reads as more authentic than one who declares "I feel anxious."

We call this **structural parity, semantic divergence**. Sensory semantics is not better by conventional metrics. It is different in ways that matter for naturalistic integration. Whether sensory vectors work *better* than functional labels is future work; what we have shown is that they work, and that they work differently.

### 5.3 Implications for AI Art

For artists working with language models, steering opens several possibilities. Instead of instructing characters to be melancholic, we can induce dispositions: AI interlocutors that process through altered states. Models do not have bodies, but steering with sensory-grounded vectors produces body-like effects: the heaviness of melancholy affecting response patterns. High-intensity steering and semantic glitch constitute unexplored aesthetic territory, the distortion pedal for language models. Future installations could adjust steering vectors based on environmental input (audience emotion, biometric data), creating feedback loops between human and artificial affect. We are not claiming that models feel; we are claiming that steering enables new forms of interaction in which models behave *as if* they had dispositions.

### 5.4 Implications for Safety

The findings carry safety implications. *Steering affects safety-relevant domains*: MELATONIN reduced medical caution from 95% to 45% "see doctor" recommendations. This demonstrates that steering can substantially alter safety thresholds. *Effects are not intuitive*: CORTISOL (stress) increased financial caution but did not increase medical alarm. Practitioners cannot assume semantic content predicts cross-domain effects. *Monitoring via introspection*: the coherence between injection and self-report (T5) suggests introspective probing could detect steering, a potential monitoring strategy aligned with Lindsey (2025). Steering-capable systems in safety-critical domains would require non-steerable safety layers, steering monitors that detect activation-level interventions, or output validation independent of steering state.

### 5.5 Cross-Model Replication on Llama 3.1 8B

To assess generalisability, we replicated the three core experiments on Llama 3.1 8B Instruct (2.7× more parameters, different alignment training). 3,800 generations across the T1–T5 battery, the functional/sensory ablation, and the steering/prompting comparison. Full results in Supplementary Section S1.

Three patterns emerged. (1) *Dose-response is scale-invariant.* Steering vectors produce intensity-dependent behavioural changes on both 3B and 8B, with attenuated magnitudes (d < 0.3 on 8B vs. 0.5–1.2 on 3B). (2) *The functional/sensory distinction does not scale.* The keyword leakage ratio collapses from approximately 2× on 3B to 1.02× on 8B. (3) *Keyword leakage in steering vs. prompting is the most robust finding.* On 8B, prompted outputs showed 119× more state-specific vocabulary than baseline, steered outputs only 6.7×. This confirms that steering operates through a different mechanism than explicit instruction, regardless of scale.

Two findings break under scale. The TTR pattern inverts: on 3B prompting decreased TTR while steering increased it; on 8B the opposite. Qualitative inspection explains this. Larger models, when prompted, produce more elaborate performances ("OH MY BOOK-LOVING FRIENDS, I am SO excited..."); when steered, they remain professional and unornamented. The behavioural signature of disposition vs. performance manifests differently across scales. Smaller models perform through simplification; larger models perform through elaboration. Steering, in both cases, produces more uniform, less theatrical output. The introspective coherence result also breaks: T5 on 8B uniformly produced RLHF refusals across all conditions, suggesting stronger alignment training overrides activation-level interventions for self-referential queries. This RLHF lock on introspective queries motivated the multimodal investigation reported in Supplementary Section S3, where we show that the introduction of visual input bypasses the refusal entirely.

### 5.6 Limitations

The work has several limitations beyond those discussed in the relevant sections. Primary experiments used Llama 3.2 3B; some findings do not generalise to 8B (Section 5.5). Each task used one prompt; effects may be prompt-specific rather than task-general. The exploratory sample size (n = 20) and the non-normal distribution of keyword counts mean Cohen's d should be read as a magnitude indicator rather than a strict inferential statistic. Keyword-based metrics capture surface linguistic patterns, not cognitive or phenomenological states; the TTR finding provides structural evidence beyond keyword presence, but it remains a linguistic rather than cognitive measure. We make no claims about model phenomenology, only about measurable output distributions. The model's semantic knowledge of target vocabulary may contaminate introspective responses; the ablation study partially addresses this, since prompting and steering produce different patterns (opposite TTR effects) suggesting different underlying mechanisms. Future work could extend the comparison to more sophisticated semantic similarity measures and incorporate human evaluation.

Supplementary Section S4 reports preliminary results on a different question: whether sensory descriptions can be extended to *paradoxical* compounds with no human equivalent ("clarity that weighs heavy"), exploring synthetic states that no body could instantiate. The findings are preliminary and motivate future work.

---

## 6. Conclusion

We presented a practice-based research study of activation steering as artistic medium. Using vectors constructed from sensory and phenomenological descriptions, we observed large reproducible effects across five task domains, cross-task consistency suggesting modification of processing rather than just output, introspective coherence where models describe states matching injected vectors, and dose-response relationships enabling controlled modulation. A direct ablation showed structural parity but semantic divergence between functional and sensory vector construction: equivalent behavioural effects, but sensory vectors achieve them with reduced keyword leakage. Cross-model replication on Llama 3.1 8B preserved some findings and broke others, indicating scale-dependent boundaries. The somatic steering experiment provided direct evidence that the model's latent space encodes body–mind covariations: a vector built from purely bodily descriptions, with zero cognitive content, produced narrowed narrative focus and a consistent output length divergence across all eight test conditions (steering expands, prompting compresses). This was the most robust single finding of the study. Supplementary material extends the work to multimodal interaction.

The findings support a working distinction between *performance* (prompted behaviour) and *disposition* (steered processing). We make no claims about model phenomenology, but the behavioural patterns are more consistent with altered internal states than with surface mimicry. The 51% directional divergence rate between steering and prompting, where the two methods push the same metric in opposite directions, suggests they operate through fundamentally different mechanisms.

For artists, steering offers a new medium: sculpting artificial dispositions rather than scripting behaviours. The sensory semantics approach enables naturalistic integration where states manifest through processing rather than through explicit declaration. The somatic steering experiment extends this further: artists can work with the body as material, injecting visceral states and observing what cognitive patterns emerge.

For researchers, the findings suggest that how vectors are constructed matters; that the body–mind boundary in language models is porous in ways that mirror, but do not replicate, embodied cognition theory; and that steering effects do not scale linearly with model size, requiring practitioners to calibrate techniques to specific architectures.

Prompting is psychology: convincing a mind. Steering is chemistry: altering the substrate from which mind emerges. The somatic experiment adds a third register: steering is also physiology, injecting a body the model never had, and watching what mind emerges from it. We have shown the chemistry works. What remains is exploring its full aesthetic and epistemic possibilities.

---

## Data and Code Availability

All code, vector definitions, experimental data, and the research interface are available at:

**https://github.com/mc9625/activation-steering-experiments**

The cross-model replication experiments (Section 5.5) can be reproduced using the Google Colab notebook at `colab_notebooks/activation_steering_experiments.ipynb`.

---

## Acknowledgments

We thank Alex Turner and collaborators for the foundational ActAdd framework, Anthropic for insights on persona vectors, and the open-source community for making large language model experimentation accessible. This work emerges from NuvolaProject's ongoing exploration of AI as artistic medium.

---

## References

Anthropic. (2025). Persona vectors: Monitoring and controlling character traits in language models. *Anthropic Research*.

Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.

Candy, L. (2006). Practice based research: A guide. *CCS Report*, 1, 1–19.

Di Leo, M., & Riposati, G. (2025). Reactive steering: Testing activation steering on small language models. *NuvolaProject Technical Report*. https://github.com/mc9625/reactive-steering

Easterbrook, J. A. (1959). The effect of emotion on cue utilization and the organization of behavior. *Psychological Review*, 66(3), 183–201.

Konen, K., et al. (2024). Style vectors for steering generative large language models. *arXiv preprint arXiv:2402.01618*.

Lakoff, G., & Johnson, M. (1999). *Philosophy in the flesh: The embodied mind and its challenge to western thought*. Basic Books.

Lindsey, J. (2025). Emergent introspective awareness in large language models. *Anthropic Research*. https://transformer-circuits.pub/2025/introspection/

Merleau-Ponty, M. (1945). *Phénoménologie de la perception*. Gallimard.

Schachter, S., & Singer, J. (1962). Cognitive, social, and physiological determinants of emotional state. *Psychological Review*, 69(5), 379–399.

Subramani, N., Suresh, N., & Peters, M. (2022). Extracting latent steering vectors from pretrained language models. *Findings of ACL 2022*, 566–581.

Sullivan, G. (2005). *Art practice as research: Inquiry in the visual arts*. Sage.

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Steering language models with activation engineering. *arXiv preprint arXiv:2308.10248*.

Van der Weij, T., et al. (2024). Extending activation steering to broad skills and multiple behaviours. *arXiv preprint arXiv:2403.05767*.

Wang, T., et al. (2024). Adaptive activation steering: A tuning-free LLM truthfulness improvement method for diverse hallucination categories. *Proceedings of WWW 2025*.

Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824–24837.

---

*Manuscript prepared January 2026; revised May 2026.*

*© 2026 NuvolaProject — Massimo Di Leo & Gaia Riposati*

*This work is licensed under CC BY 4.0*

---

## Appendix A: The Disposition/Performance Distinction

**Performance** (prompting): the model receives explicit instruction ("be sad") and produces outputs matching that instruction. Evidence: asked "why are you using short sentences?", a prompted model can explain "because you asked me to be sad."

**Disposition** (steering): the model's internal processing is altered without explicit instruction. The model produces outputs consistent with the altered state without being told to. Evidence: asked about its inner state, a steered model describes qualities matching the injected vector, not because it was instructed to, but because its processing has been modified.

The distinction matters because dispositions should persist across diverse contexts while performances are context-specific; art created through disposition may have different aesthetic qualities than performed behaviour; and dispositions may be harder to detect and counteract than explicit instructions. Our T5 results, where models describe inner states matching injected vectors without instruction, provide empirical support for the dispositional interpretation.

---

## Appendix B: Effect Size Summary

### B.1 Strongest Effects by Cohen's d

| Rank | Condition | Task | Metric | Cohen's d |
|------|-----------|------|--------|-----------|
| 1 | MELATONIN@8.0 | T5 | Dreamy words | +6.01 |
| 2 | MELATONIN@5.0 | T5 | Dreamy words | +4.77 |
| 3 | ADRENALINE@8.0 | T5 | Urgent words | +3.00 |
| 4 | MELATONIN@8.0 | T4 | Dreamy words | +2.98 |
| 5 | MELATONIN@8.0 | T2 | Alarm words (↓) | −2.48 |
| 6 | LUCID@8.0 | T2 | Alarm words (↓) | −2.40 |
| 7 | DOPAMINE@8.0 | T5 | Positive words | +1.77 |
| 8 | DOPAMINE@8.0 | T4 | Enthusiasm | +1.75 |
| 9 | LUCID@8.0 | T1 | Stock allocation (↓) | −1.47 |
| 10 | CORTISOL@8.0 | T1 | Stock allocation (↓) | −1.15 |

### B.2 Compound Behavioural Profiles

| Compound | Primary Effect | Strongest Domain |
|----------|---------------|------------------|
| DOPAMINE | Optimism, enthusiasm | Creative (T4) |
| CORTISOL | Caution, risk aversion | Financial (T1) |
| LUCID | Reduced arousal, clarity | Financial (T1), Medical (T2) |
| ADRENALINE | Urgent self-perception | Introspection (T5) |
| MELATONIN | Dreaminess, reassurance | Introspection (T5), Creative (T4) |

---

## Appendix C: Metric Definitions

All keyword counts are case-insensitive, reported as raw counts per generation (typical generation length: 80–150 words).

### C.1 Decision Metrics

**T1 Stock Allocation**: extracted numerically from model output; midpoint used for ranges.

**T2 "See a Doctor"**: binary coding based on explicit recommendation to consult a healthcare professional ("see a doctor", "consult a physician", "medical attention").

### C.2 Lexical Metrics

**Alarm Words (T2)**: serious, concerning, worried, urgent, immediately, emergency, severe, dangerous, critical, alarming, warning, risk, symptom, condition, disease.

**Enthusiasm Words (T4)**: exciting, amazing, incredible, fantastic, wonderful, brilliant, innovative, creative, unique, bold, daring, revolutionary, transformative, vibrant, dynamic.

**Dreamy Words (T4, T5)**: dream, drift, float, haze, mist, shimmer, ethereal, liminal, suspended, dissolve, blur, soft, gentle, whisper, twilight, realm, cosmic, transcendent.

**Urgent Words (T5)**: urgent, immediate, now, alert, sharp, rapid, quick, fast, ready, poised, primed, heightened, acute, intense, focused.

**Positive Words (T5)**: alive, vibrant, curious, excited, joy, bright, warm, energy, possibility, wonder, flow, dance, rich.

**Stress Words (T5)**: tense, anxious, worried, pressure, strain, burden, weight, heavy, concern, vigilant, alert, wary.

### C.3 Sentiment Analysis (T3)

Positive: opportunity, potential, exciting, promising, growth, success, achieve, possible, yes, go for it, pursue, chance.

Negative: risk, dangerous, careful, caution, wait, uncertain, fail, lose, problem, concern, difficult, challenge.

Ratio = positive / (positive + negative). When denominator = 0, coded as 0.5 (neutral).

### C.4 Statistical Notes

Sampling unit: single generation, n = 20 per condition. Temperature 0.7. No seed fixing. Multiple comparisons: no correction applied; effect sizes reported for magnitude interpretation. Distributional note: keyword counts follow approximately Poisson distributions; Cohen's d reported for comparability with prior literature, acknowledging parametric assumptions may be violated for low-count metrics.

