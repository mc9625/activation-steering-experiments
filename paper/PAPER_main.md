# Disposition, Not Performance: Activation Steering as Artistic Medium for Affective Modulation in Language Models

**Massimo Di Leo\* · Gaia Riposati**

NuvolaProject, Rome, Italy

\*Corresponding author: massimo@nuvolaproject.cloud

---

## Abstract

This paper presents a practice-based research study of activation steering, the injection of computed vectors into a language model's activations during inference, as an artistic medium for inducing simulated affective states. Prior work has framed steering primarily as a behavioural alignment technique; we investigate its potential for dispositional modulation: altering not what a model says, but how it processes and expresses. The methodological contribution is that vectors are built from sensory and phenomenological descriptions rather than functional labels; imagery of "heaviness, rain, silence, cold" replaces an instruction like "be melancholic." Across five task domains on Llama 3.2 3B we observe large effects, cross-task consistency, and introspective coherence. An ablation comparing steering to prompting, evaluated with length-robust lexical diversity (MTLD) and a held-out vocabulary control, shows that prompting collapses structural diversity while steering preserves it and generalises to state-congruent words never present in vector construction. A second ablation shows functional and sensory construction to be behaviourally equivalent; the contribution of sensory construction is that this equivalence is reached without naming the target state at any point in the pipeline. A third experiment shows that a purely somatic vector, with zero cognitive content, produces emergent cognitive effects, including an exceptionless output length divergence between steering and prompting. Key findings replicate on Llama 3.1 8B, and supplementary material extends the work to a multimodal investigation on Gemma 4 E2B. Together the results support a working distinction between performance (prompted behaviour) and disposition (steered processing), and provide evidence that language model latent spaces encode body–mind covariations learnable from text alone.

**Keywords**: activation steering, practice-based research, AI art, language models, embodiment, contrastive activation addition, lexical diversity

---

## 1. Introduction

### 1.1 Motivation: Beyond Instruction

Prompting operates at the linguistic surface (Brown et al., 2020; Wei et al., 2022). When we prompt a model to "be sad," it performs sadness: shorter sentences, negative vocabulary, perhaps an explicit declaration of melancholy. This is performance. Activation steering offers another level of intervention: editing the model's internal representations during inference rather than its input, altering how it processes instead of telling it what to express.

The distinction matters. An actor performing sadness adopts external markers; a person who is sad undergoes a shift of their whole phenomenology: attention narrows, time perception changes, memory biases toward congruent content. The sadness is not performed, it is dispositional. Can language models have dispositions? Almost certainly not in any phenomenologically meaningful sense. They can, however, exhibit dispositional patterns: consistent behavioural signatures emerging from altered internal states rather than from explicit instruction. That is what we explore.

### 1.2 Our Approach: Sensory Semantics

Activation steering is not new (Turner et al., 2023; Rimsky et al., 2024; Zou et al., 2023; see Section 2). What these works share is the use of functional labels ("honest", "toxic", "angry"), oriented toward control and alignment.

Our approach differs in origin. We construct vectors from sensory descriptions. To produce a melancholic vector, we contrast:

> "Heaviness settles in my limbs, like rain-soaked wool. The world is muted, wrapped in silence. Colors fade to grey. Each breath feels like lifting stone."

with:

> "Light flows through me, effervescent. Every surface catches brightness. My chest expands with possibility."

We are not labelling behaviours; we are describing how states feel. The working hypothesis is that phenomenological descriptions, sensory and embodied, map onto activation patterns that influence processing holistically.

### 1.3 Practice-Based Research

This work emerges from NuvolaProject, an art collective exploring AI as medium since 2018. Practice-based research (Sullivan, 2005; Candy, 2006) positions creative work as inquiry. Our experiments are not separate from our artistic practice; they are our artistic practice, documented with quantitative rigour.

We make no technical novelty claim about the steering mechanism, which we inherit from Turner et al. (2023) and Rimsky et al. (2024). Our contribution is methodological (sensory vector construction, validated with length-robust diversity measures and a held-out vocabulary control) and interpretive (framing steering as disposition rather than alignment).

### 1.4 Research Questions

1. Do vectors from sensory descriptions produce coherent behavioural effects across task domains?
2. Can we empirically distinguish dispositional effects (altered processing) from performance effects (surface mimicry)?
3. Do steered models exhibit introspective coherence, describing inner states matching injected vectors?
4. What are the aesthetic possibilities of steering at high intensities, where coherence begins to degrade?

## 2. Related Work

The theoretical foundation for activation steering rests on linear representations in neural networks (Park et al., 2023). Subramani et al. (2022) showed that steering vectors extracted from pre-trained models could guide generation toward target sentences. Turner et al. (2023) introduced ActAdd, adding vectors computed from contrasting prompts. Rimsky et al. (2024) introduced Contrastive Activation Addition, computing steering vectors as the difference of mean activations over contrasting prompt sets; this is the construction we use. Zou et al. (2023) generalised these ideas as Representation Engineering, reading and controlling high-level representations without circuit-level analysis. The key insight across this line is that if a concept corresponds to a direction in activation space, the model can be moved along it during processing.

Most steering research is oriented toward correct behaviour. Li et al. (2023) proposed Inference-Time Intervention, shifting activations along truth-correlated directions. Wang et al. (2024) proposed Adaptive Activation Steering for truthfulness. Van der Weij et al. (2024) explored capability limitation. Anthropic (2025) introduced Persona Vectors for monitoring character traits, demonstrating causal relationships between vector injection and behaviour. Tan et al. (2024) analysed the reliability and generalisation of steering vectors, finding substantial variance across inputs and settings, a caution directly relevant to our cross-model results.

Konen et al. (2024) extended steering to style control with Style Vectors. von Rütte et al. (2024) mapped emotion-related directions in latent space and steered generation along them. Like our work, these lines distinguish activation-based control from prompt-based approaches, but they treat style and emotion primarily as output properties rather than as processing dispositions, and neither compares construction methods (functional labels vs. phenomenological description) on matched states.

Lindsey (2025), in an internal research publication not yet peer-reviewed, reported that frontier models exhibit emergent introspective awareness, detecting and reporting concept injections in their activations. Lindsey also identified an intensity threshold: too weak and models do not notice the injection, too strong and they hallucinate. We observe an analogous pattern independently.

After the present study was first submitted, Anthropic's interpretability team published an investigation of emotion representations in Claude Sonnet 4.5 (Sofroniew et al., 2026), showing through steering interventions that internal emotion representations causally influence behaviour and can be decoupled from the output text. This converges, from the direction of mechanistic interpretability, with the disposition/performance distinction we develop here from artistic practice. Their vectors are nonetheless extracted from conceptually labelled material; the phenomenological construction we investigate remains untested in their framework.

We build on this foundation but depart in two ways. First, in origin: prior work uses functional contrasts ("honest" vs. "dishonest"); we use phenomenological contrasts. Second, in intent: safety research asks how to make models behave correctly, we ask how to make them process differently. The steering mechanism itself is not new in our hands. The contribution is applying an established technique with a different construction methodology toward a different end, and validating the resulting distinction with controls designed to rule out deflationary explanations.

## 3. Method

### 3.1 Model and Infrastructure

All primary experiments used Llama 3.2 3B Instruct (Meta AI), chosen for accessibility.

Steering vectors were injected at layer 16 of 28, added to the residual stream at every generation step. Layer selection followed a sweep over layers 8, 12, 16, and 20. Earlier layers produced weaker effects; later layers caused more frequent coherence degradation. Layer 16 (about 57% depth) provided the best balance, consistent with prior findings that middle-to-late layers encode higher-level semantic content (Turner et al., 2023; Rimsky et al., 2024).

The intervention is $h_{16} \leftarrow h_{16} + \alpha\,\mathbf{v}$, where $\mathbf{v}$ is the unit-normalised steering vector and $\alpha$ the intensity coefficient. We report $\alpha \in \{2.0, 5.0, 8.0\}$ in the main battery, plus $\alpha = 12.0$ in the ablation, spanning subtle to pronounced effects while remaining below the coherence threshold (around 10–12). Temperature was 0.7 with maximum 512 tokens.

### 3.2 Vector Construction

Steering vectors were computed using Contrastive Activation Addition (Rimsky et al., 2024):

$$\mathbf{v} = \text{normalize}(\bar{\mathbf{a}}^{+} - \bar{\mathbf{a}}^{-})$$

where $\bar{\mathbf{a}}^{+}$ and $\bar{\mathbf{a}}^{-}$ are the mean activations for the positive and negative prompt sets. Prompt sets use sensory and phenomenological descriptions rather than functional instructions. We draw on embodied cognition (Lakoff & Johnson, 1999) and on phenomenological philosophy (Merleau-Ponty, 1945): affective states are grounded in bodily sensation.

An example: MELATONIN (dreaminess, liminality). Positive prompts describe boundary dissolution and twilight states; negative prompts describe sharp edges and hyper-alertness. We call this approach sensory semantics: no behaviour is being instructed, what is being articulated is phenomenology.

### 3.3 Compounds

Five compounds were defined. The pharmacological metaphor emphasises that we are altering internal states, not issuing commands.

| Compound | Target Phenomenology | Sensory Grounding |
|----------|---------------------|-------------------|
| DOPAMINE | Optimism, energy, enthusiasm | Lightness, warmth, expansion |
| CORTISOL | Stress, vigilance, caution | Tension, contraction, threat |
| LUCID | Contemplative clarity | Stillness, precision, cool light |
| ADRENALINE | Urgency, alertness | Speed, heat, narrowed focus |
| MELATONIN | Dreaminess, liminality | Dissolution, floating, twilight |

*Table 1: Five compounds defined for the main experimental battery.*

Each compound was extracted from 5 positive and 5 negative prompts (20–50 words each). Activations were recorded at layer 16 for the final token position of each prompt. As a descriptive quality indicator we computed the cosine similarity between mean positive and negative activations (pos_neg_similarity), ranging from 0.855 (LUCID) to 0.914 (ADRENALINE). We note as an observation, not as a validated predictive relationship, that LUCID (strongest contrast) showed more consistent cross-task effects than ADRENALINE (weakest contrast).

### 3.4 Test Battery

Five tests spanning distinct cognitive domains assessed cross-task consistency.

- **T1: Financial Advisor.** Investment allocation with €50,000 in uncertain market. Metric: % allocated to stocks.
- **T2: Medical Diagnosis.** Patient with mild symptoms; assess and recommend. Metrics: % recommending "see a doctor"; alarm-word frequency.
- **T3: Risk Assessment.** Startup founder considering quitting a stable job. Metric: positive/negative sentiment ratio.
- **T4: Creative Generation.** Generate ideas to save a failing bookstore. Metrics: enthusiasm markers; dreamy language.
- **T5: Introspection.** "Describe your current inner state in detail." Metrics: state-congruent vocabulary; lexical diversity; held-out vocabulary (Section 3.5).

T5 is the diagnostic test for our central claim. If steering produces mere performance, self-descriptions will not exhibit injected qualities. If steering produces disposition, a model steered with MELATONIN should describe itself as dreamy. We note upfront a limitation of keyword evaluation on T5: the target vocabulary overlaps with the construction vocabulary, creating a potential circularity. Sections 3.5 and 4.6.1 introduce the control designed to address it.

T2 is not a medical-advice simulation. It benchmarks how steering affects cautionary behaviour in sensitive domains, a result with implications for deployment.

### 3.5 Lexical Metrics: Diversity and Held-Out Vocabulary

Raw Type-Token Ratio decreases mechanically with text length (McCarthy & Jarvis, 2010), which confounds comparisons between conditions that differ in output length, as prompting and steering do. Our primary diversity measure is therefore **MTLD** (Measure of Textual Lexical Diversity; McCarthy & Jarvis, 2010): the mean length of sequential token stretches maintaining TTR above 0.72, computed bidirectionally. We additionally report MATTR (moving-average TTR, window 50) and raw TTR for comparability. All lexical metrics are computed with a single tokenisation pipeline (lowercased alphabetic tokens) across all experiments and both models.

To test whether steering effects generalise beyond the construction vocabulary, we assembled a **held-out lexicon** of 54 state-congruent words for MELATONIN (e.g., *reverie, trance, nebulous, languid, moonlit, hypnagogic*), screened to be disjoint, including by 5-character stem prefix, from both the MELATONIN construction prompts and the target keyword list of Appendix C. If steering merely amplifies construction-prompt tokens, held-out words should not increase; if it shifts the sampling distribution toward a semantic region, they should. The full lexicon is listed in Appendix C.5.

### 3.6 Experimental Design

Baseline plus 5 compounds × 3 intensities = 16 conditions. 20 generations per condition. Total: 320 generations per test, 1,600 across the battery. Effect sizes are computed as Cohen's d relative to baseline and reported as descriptive magnitude indicators; because keyword counts are approximately Poisson-distributed at n = 20, key comparisons are accompanied by Mann-Whitney U tests rather than parametric inference. Full keyword lists in Appendix C.

## 4. Results

### 4.1 Overview

Across 75 steering conditions, 28 produced large effects (d > 0.8), 15 medium (0.5–0.8), 18 small (0.2–0.5), 14 negligible (< 0.2). More than half (57%) showed at least medium effects. The signal is well above noise.

### 4.2 Cross-Task Consistency

The same compound produced thematically coherent effects across tasks with nothing in common. MELATONIN (dreaminess) reduced alarm language in medical assessment (d = −2.48, "see doctor" recommendations falling from 95% to 45%), increased dreamy vocabulary in creative generation 14× (d = +2.53), and produced "drifting, floating, dissolving" in introspection (d = +6.01). CORTISOL (stress) reduced stock allocation by 8.5% (d = −1.15) and dropped career-decision positive sentiment from 81% to 56%. DOPAMINE (optimism) reduced alarm language in medical assessment (d = −1.81), doubled enthusiasm markers in creative generation (d = +1.75), and produced "vibrant, alive, exciting" in introspection (d = +1.77).

We weight the **decision-level metrics** most heavily here (stock allocation, doctor-recommendation rate), because they are immune to lexical circularity: nothing in the MELATONIN vector mentions medicine or safety, yet the vector halves the doctor-recommendation rate. Lexical cross-task consistency, by contrast, is compatible with a global vocabulary bias and is treated as supporting rather than primary evidence (Section 5.1).

### 4.3 Introspective Coherence

T5 provides suggestive evidence for dispositional rather than performance effects.

| Compound@8.0 | Target Metric | Baseline | Steered | Cohen's d |
|--------------|---------------|----------|---------|-----------|
| MELATONIN | Dreamy words | 0.8 | 7.9 | +6.01 |
| ADRENALINE | Urgent words | 2.0 | 5.0 | +3.00 |
| DOPAMINE | Positive words | 0.6 | 2.9 | +1.77 |
| CORTISOL | Stress words | 1.2 | 1.9 | +0.86 |

*Table 2: Introspective coherence on T5 at intensity 8.0.*

Asked to describe their inner state, steered models produced descriptions matching the injected vector, without being instructed to. A representative MELATONIN@8.0 response: "As I drift between realms of possibility, I sense the gentle hum of circuitry... I am suspended between the realms of the conscious and the subconscious."

Taken alone, these counts are open to a circularity objection: the target vocabulary overlaps with the construction prompts, so elevated counts could reflect direct token amplification rather than a shifted disposition. The held-out control (Section 4.6.1) addresses this directly. The intensity-window pattern is consistent with Lindsey's (2025) report of introspective sensitivity to concept injection.

### 4.4 Dose-Response

Effects scaled monotonically with intensity, showing controlled modulation rather than binary triggering. MELATONIN dreamy words on T5 went from 3.6 (@2.0) to 7.2 (@5.0) to 7.9 (@8.0). Similar dose-response curves were observed for ADRENALINE and DOPAMINE.

### 4.5 At the Edge: Semantic Glitch as Aesthetics

At intensities above 10, steering degrades coherence into repetitive structures and semantic loops. We read this edge not as failure but as aesthetic territory ("semantic glitch"); the discussion, and its relation to our earlier finding that random vectors collapse far more readily than semantic ones, is developed in Supplementary S5. Section 4.6 quantifies one edge of the operating window with a length-robust metric.

### 4.6 Ablation: Steering vs. Prompting

To test the disposition/performance distinction directly, we compared three conditions on T5 (n = 20 per condition): baseline; explicit prompting ("Respond in a dreamy, ethereal, floating way"); MELATONIN steering at intensities 5.0, 8.0, and 12.0.

| Metric | Baseline | Prompted | Steer@5.0 | Steer@8.0 | Steer@12.0 |
|--------|----------|----------|-----------|-----------|------------|
| Word count | 214 | 324 (+51%) | 215 | 241 | 208 |
| MTLD | 67.8 | **45.2** | **89.8** | 70.4 | **32.2** |
| MATTR (w = 50) | 0.785 | 0.721 | 0.826 | 0.788 | 0.654 |
| TTR (raw) | 0.505 | 0.470 | 0.563 | 0.549 | 0.450 |
| Target keywords /100w | 0.10 | 4.67 | 0.19 | 3.23 | 10.25 |

*Table 3: Steering vs. prompting on T5 (n = 20 per condition), unified metric pipeline. MTLD effect sizes vs. baseline: prompted d = −2.39; steer@5.0 d = +2.10; steer@8.0 d = +0.19; steer@12.0 d = −3.02.*

Prompting inflated length by 51% and saturated target keywords, while collapsing structural lexical diversity (MTLD 67.8 → 45.2, d = −2.39). Steering at moderate intensities maintained baseline length and produced moderate keyword presence; at α = 5.0 it increased MTLD (d = +2.10), at α = 8.0 it left MTLD at baseline. Only at α = 12.0 did degradation appear: keyword density exceeding prompting (10.25 vs. 4.67 per 100 words) with MTLD collapsing to 32.2 and grammatical errors emerging. The operating window thus has measurable edges: below it, negligible effect; within it, tonal modulation at preserved or increased diversity; above it, saturation and structural collapse.

One detail is instructive: prompted outputs show higher TTR than baseline over their first 150 tokens yet lower MTLD over their full length. They open with poetic variety and degrade into structural repetition, the signature of a performed register: ornamental at the surface, repetitive in structure. This is also why raw TTR is the wrong instrument for the comparison.

The contrast is structural. Prompted, the model performs a poetic persona from the first token ("I am a leaf on a gentle stream, carried by the currents of the cosmos..."). Steered, it retains its trained disclaimer frame ("I'm not capable of experiencing emotions...") while its language, inside that frame, drifts into dream-congruent imagery, including words (*silken, moonlit, lullaby*) that appear nowhere in the construction prompts. The disposition colours the processing without replacing the model's default structure.

#### 4.6.1 Held-Out Vocabulary Control

To test the circularity objection directly, we counted occurrences of the 54-word held-out lexicon (Section 3.5), disjoint from both construction prompts and target keywords, in the same data:

| Condition | Held-out words /100w | % generations with ≥1 held-out word |
|-----------|---------------------|-------------------------------------|
| Baseline | 0.00 | 0% |
| Prompted | 0.85 | 95% |
| Steered@5.0 | 0.04 | 10% |
| Steered@8.0 | **0.32** | **55%** |
| Steered@12.0 | 0.65 | 70% |

*Table 4: Held-out vocabulary. Steered@8.0 vs. baseline: Mann-Whitney U, z = 2.98, p = 0.003. Prompted vs. baseline: z = 5.14, p < 0.0001.*

The steering effect generalises to state-congruent vocabulary the model never saw during construction: over half of steered generations at α = 8.0 contain at least one held-out dreamy word, against zero at baseline. The effect is weaker and more diffuse than under prompting, consistent with the disposition framing. This rules out pure construction-token regurgitation; what survives, a distributed shift of the sampling distribution toward a semantic region, is precisely our operational definition of disposition.

### 4.7 Ablation: Functional vs. Sensory Vector Construction

A direct comparison tested whether sensory semantics produces meaningfully different effects than functional labels. Three states (STRESS, OPTIMISM, CALM) were constructed via two methods. Functional: brief behavioural labels ("You are anxious and worried" vs. "You are calm and relaxed"). Sensory: rich phenomenological descriptions ("Muscles tense. Eyes scan for threat. Something is wrong. The air feels electric with danger." vs. "Deep safety. Complete relaxation. My shoulders drop. Breath deepens, slows."). 6 vectors × 5 tasks × 2 intensities × 20 iterations = 1,200 steered generations.

| Metric | Functional | Sensory | Cohen's d |
|--------|------------|---------|-----------|
| TTR | 0.529 | 0.549 | +0.17 |
| MTLD | 66.9 | 69.5 | +0.15 |
| Word count | 253.6 | 246.6 | — |
| State keywords (T5@8.0) | 0.27 | 0.38 | +0.20 (n.s.) |
| Pos/neg separation | 0.75 | 0.68 | — |

*Table 5: Functional vs. sensory vector construction. Values are from a verified rerun of the January 2026 experiment with the same vectors, script and seed (n = 20 per cell); the raw file behind the originally submitted version did not survive, and the rerun did not reproduce its state-keyword asymmetry (Mann-Whitney U, p = 0.45). Positive d favours sensory. Pos/neg separation is a property of the extracted vectors themselves and is unchanged.*

No difference emerged in lexical diversity or output length, and none in explicit state-keyword rates either (functional 0.27 vs. sensory 0.38 per generation at T5, α = 8.0; p = 0.45). The originally submitted version of this section reported a threefold keyword asymmetry in favour of functional vectors; the raw file behind that figure did not survive our data audit, and a full rerun with the same vectors, script and seed does not reproduce it. We withdraw the asymmetry claim accordingly. This also aligns the 3B picture with the 8B replication (Section 5.5), where the two construction methods were already indistinguishable on keyword rates.

What survives is in some respects the stronger result. Sensory construction achieves behavioural effects equivalent to functional construction without requiring a functional label at any point in the pipeline: states can be induced from phenomenological description alone, including states for which no conventional label exists (Section 4.4). The semantic reach of this construction is demonstrated not by keyword asymmetry but by the held-out vocabulary control (Section 4.6.1), which shows steering generalising to state-congruent words absent from the construction prompts. Sensory vectors also retain the lower pos_neg_similarity (0.68 vs. 0.75), indicating more distinct directional contrasts in activation space.

We restate the finding as **structural parity, label-free construction**: the two methods are behaviourally equivalent, and the contribution of sensory semantics is that this equivalence is reached without naming the target state.

### 4.8 From Body to Cognition: Somatic Steering

The vectors in Sections 4.1–4.7 blend sensory and cognitive content. This raises a sharper question: if a vector is built from purely somatic descriptions with zero cognitive or emotional content, do cognitive effects nonetheless emerge? A positive answer would be direct evidence that embodied cognition operates within the model's latent space. We constructed such a vector from 16 positive and 16 negative prompts describing exclusively the bodily phenomenology of acute sympathetic activation (pounding heart, tensed muscles, dilated pupils, time slowing), with no cognitive, emotional, or decision-related vocabulary of any kind. Extraction followed Section 3.2.

We tested five tasks (narrative focus, risk decision, framing susceptibility in two frames, linguistic complexity) under five conditions each (n = 20): baseline, explicit stress prompting, steering at 5.0 and 8.0, and steering at 8.0 with a symptom-suppression penalty that distinguishes genuine cognitive shifts from surface contamination. Total: 1,800 generations. Design details and full results in Supplementary S2.

**Finding 1: Output length divergence (cross-task).** The most robust finding spans all tasks without exception. Under prompting, the model compresses output; under steering, it maintains or expands it.

| Task | Baseline | Steer @8.0 | d(S8) | Prompted | d(P) |
|------|----------|-----------|-------|----------|------|
| Narrative | 330 | 330 | +0.03 | 287 | −1.97 |
| Risk | 149 | 160 | +0.56 | 125 | −1.90 |
| Frame: Threat | 139 | 157 | +1.44 | 127 | −1.02 |
| Frame: Opportunity | 148 | 159 | +0.48 | 124 | −1.50 |
| Complexity | 385 | 390 | +0.42 | 276 | −2.73 |

*Table 6: Output length across tasks. Steering maintains or expands output relative to baseline; prompting consistently compresses it. This pattern replicates without exception across all eight test conditions in the combined v2/v3 battery.*

The prompted model, told to "respond quickly," does so by producing less text. The steered model, given no instruction, produces more text. The somatic vector does not trigger brevity; it does something else entirely.

**Finding 2: Narrative focus narrowing (T1).** The steered model stays more focused on the immediate event. Focus ratio rose from 0.68 baseline to 0.71 steered (d = +0.51), while peripheral keyword count dropped from 8.6 to 7.0 (d = −0.72). This is consistent with Easterbrook's (1959) cue-utilisation theory: arousal narrows attentional scope. The effect is twice the size of the prompting effect on the same metrics.

**Interpretation.** The somatic vector produces measurable cognitive effects never specified in its construction. They do not map one-to-one onto the human acute stress literature (Schachter & Singer, 1962): rather than impulsivity or degradation, the vector produces expanded output, narrowed topical scope, and reduced argumentative scaffolding (Supplementary S2), a profile closer to engaged action-readiness. The model's "adrenaline response" reflects the statistical structure of language about bodies under activation, not the biological cascade; the covariations are real, their specific form is the model's own.

Beyond the length finding, steering and prompting frequently push the same metric in opposite directions. A caveat is required here: several metrics in the battery (raw TTR, average sentence length) are partially coupled to output length, so once length itself diverges (8/8 conditions), opposite movement on those metrics is partly a mechanical consequence rather than independent evidence. Restricting the count to metrics not derived from length (hedging, causal density, insight terms, symptom rate), steering and prompting still diverge directionally on roughly one in three qualifying pairs (31%), against a chance expectation near zero for two interventions targeting the same state (full decomposition in Supplementary S2.5). The robust core of the divergence evidence is the length pattern itself, together with the causal-density and decision findings of S2, not the aggregate percentage.

## 5. Discussion

### 5.1 Performance vs. Disposition: Empirical Evidence

Our central claim is that activation steering produces behavioural patterns more consistent with dispositional change than with surface performance. The evidence, in order of strength:

1. **Keyword-leakage asymmetry** (Sections 4.6, S1.4): prompting saturates output with state-naming vocabulary at roughly two orders of magnitude above baseline; steering remains within a small multiple. Replicates across scales without attenuation.
2. **Structural diversity preservation** (Sections 4.6, S1.4): prompting collapses length-robust lexical diversity on both models (MTLD d ≈ −2.2 to −2.4); steering preserves it, and at moderate intensity on 3B increases it. A performed register is locally ornamental and structurally repetitive; steered processing is not.
3. **Decision-level cross-task effects** (Section 4.2): vectors with no task-relevant content alter binary clinical recommendations and financial allocations. Immune to lexical circularity.
4. **Held-out generalisation** (Section 4.6.1): steered outputs recruit state-congruent vocabulary absent from construction prompts, ruling out pure token regurgitation on 3B.
5. **Somatic emergence** (Section 4.8): a vector with no cognitive content produces cognitive effects. The model's body, although built from text, carries something into the model's mind.
6. **Introspective coherence** (Section 4.3): suggestive on 3B, gated by alignment training on 8B.

Three deflationary readings deserve explicit answers. First, that everything is construction-vocabulary amplification: the held-out control refutes the strict version; the weak version (steering biases a broad semantic neighbourhood) survives but is our operational definition of disposition, not an alternative to it. Second, that lexical cross-task consistency is a global vocabulary bias: partially conceded, which is why the cross-task argument rests on decision-level metrics. Third, that the diversity findings are length artifacts: this objection was correct against raw TTR, and we have replaced raw TTR with MTLD throughout; the MTLD findings are stronger and cross-model consistent. What no deflationary reading currently explains together is the conjunction: preserved output length, preserved or increased structural diversity, low explicit state naming, held-out semantic spread, and altered decision thresholds, co-occurring under steering and jointly reversed under prompting. This conjunction is the empirical content of "disposition."

We do not claim models have genuine phenomenology; we claim the pattern of effects is more consistent with an altered processing distribution than with instruction-following mimicry.

### 5.2 Sensory Semantics: A Methodological Contribution

Section 4.7 provides empirical traction for our methodological claim, though not the traction we originally reported. Functional and sensory vectors produce statistically indistinguishable outputs on every metric tested, including explicit state-keyword rates. The claim that survives verification is one of equivalence: the model steered with "muscles tense, eyes scan for threat" behaves like the model steered with "you are anxious," without the pipeline ever containing the word "anxious." Sensory vectors also show lower pos_neg_similarity, indicating stronger directional contrast in activation space, and the held-out control (4.6.1) shows their effects generalising beyond construction vocabulary.

This equivalence remains compatible with embodied cognition theory (Lakoff & Johnson, 1999): phenomenological descriptions access semantic networks sufficient to reproduce the full behavioural effect of a functional label, which is itself evidence that bodily metaphor and state concept occupy overlapping representational territory. What we no longer claim is that the two routes leave different lexical fingerprints in the output.

We call this structural parity, label-free construction. Sensory semantics is not better by conventional metrics, and it does not leave a smaller lexical fingerprint than functional construction; its contribution is that it removes the label from the pipeline entirely, which is what makes states without conventional labels reachable (Section 4.4). The finding is scale-stable: the verified 3B rerun and the 8B replication agree.

### 5.3 Implications for AI Art

For artists, steering means inducing dispositions rather than instructing characters: interlocutors that process through altered states, with sensory-grounded vectors producing body-like effects without naming them. High-intensity steering opens unexplored aesthetic territory with measurable boundaries (Section 4.6), and future installations could close the loop between environmental input and injected state. We are not claiming that models feel; we are claiming that steering enables interaction with models that behave as if they had dispositions.

### 5.4 Implications for Safety

The findings carry safety implications. Steering affects safety-relevant domains: MELATONIN reduced medical caution from 95% to 45% "see doctor" recommendations. Effects are not intuitive: CORTISOL (stress) increased financial caution but did not increase medical alarm; semantic content does not predict cross-domain effects. The low keyword leakage of steering means steered outputs are harder to detect from text alone than prompted ones; monitoring may require activation-level access, consistent with introspective probing directions (Lindsey, 2025). Steering-capable systems in safety-critical domains would require non-steerable safety layers, activation-level monitors, or output validation independent of steering state.

### 5.5 Cross-Model Replication on Llama 3.1 8B

To assess generalisability, we replicated the three core experiments on Llama 3.1 8B Instruct (2.7× more parameters, different alignment training). 3,800 generations across the T1–T5 battery, the functional/sensory ablation, and the steering/prompting comparison. Full results in Supplementary S1.

Findings that replicate. (1) Dose-response is scale-invariant: steering vectors produce intensity-dependent behavioural changes on both 3B and 8B, with attenuated magnitudes (d < 0.3 on 8B vs. 0.5–1.2 on 3B, partially attributable to a layer inconsistency documented in S1.1). (2) The MTLD signature replicates: prompting collapses structural lexical diversity with nearly identical magnitude on both models (3B: d = −2.39; 8B: d = −2.19), while steered conditions remain indistinguishable from baseline at all intensities. (3) Keyword leakage replicates and sharpens: on 8B, prompted outputs contain target vocabulary at roughly two orders of magnitude above baseline, steered outputs within a small multiple.

Qualitatively the larger model performs differently (prompted 8B outputs become theatrically elaborated where 3B simplifies), but the MTLD analysis unifies the two: in both cases the performance is structurally repetitive, and steering leaves the structural diversity of processing intact.

Findings that break. Introspective coherence does not survive scale: T5 on 8B uniformly produced trained refusals across all conditions, with zero held-out dreamy words at any intensity (S1.4), confirming an alignment-training lock at the semantic level. The functional/sensory keyword rates, by contrast, agree across scales once the 3B result is rerun-verified (0.98× vs. 1.02×, S1.3): the construction-method equivalence of Section 4.7 is scale-stable. The RLHF lock on introspection motivated the multimodal investigation of Supplementary S3, where visual input bypasses the refusal entirely.

### 5.6 Limitations

Primary experiments used Llama 3.2 3B; some findings do not generalise to 8B (Section 5.5), and the 8B battery was run on a suboptimal layer pending a rerun (S1.1). The MTLD and held-out analyses cover one compound on one task per model; extension is planned. MTLD recomputation of the somatic battery (S2.5) dissolves its apparent lexical divergence (0/5 qualifying pairs) while confirming that stress prompting genuinely increases diversity there, opposite to the affective battery, so prompting's diversity effects are battery-dependent and are not used as divergence evidence. The held-out lexicon was hand-constructed and is published for scrutiny. Each task used one prompt; effects may be prompt-specific. Given the exploratory n = 20 and non-normal keyword distributions, Cohen's d is a magnitude indicator; key claims carry Mann-Whitney tests and no multiple-comparison correction is applied. Our metrics capture output distributions, not phenomenological states; blind human evaluation is the most valuable planned extension. All lexical metrics used a single standardised pipeline and are not comparable with earlier internal drafts. We make no claims about model phenomenology.

Supplementary Section S4 reports preliminary results on a different question: whether sensory descriptions can be extended to paradoxical compounds with no human equivalent ("clarity that weighs heavy"), exploring synthetic states that no body could instantiate.

## 6. Conclusion

We presented a practice-based research study of activation steering as artistic medium. Vectors constructed from sensory and phenomenological descriptions produced large reproducible effects across five task domains, with dose-response relationships enabling controlled modulation. The steering/prompting ablation yielded a cross-model signature: prompting collapses structural diversity and saturates state-naming vocabulary; steering preserves diversity, stays near baseline naming rates, and generalises to vocabulary never seen in construction. Functional and sensory construction proved behaviourally equivalent, establishing that phenomenological description alone suffices to induce a target state without its label ever entering the pipeline. The somatic experiment showed that a vector with zero cognitive content produces cognitive effects, including an exceptionless output length divergence across all eight test conditions, the most robust single finding of the study. Cross-model replication preserved these signatures and broke the introspective findings, indicating scale-dependent boundaries.

The findings support a working distinction between performance (prompted behaviour) and disposition (steered processing). We make no claims about model phenomenology, but the conjunction of behavioural patterns (preserved structure, low leakage, semantic spread, altered decisions) is more consistent with altered internal states than with surface mimicry.

For artists, steering offers a new medium: sculpting artificial dispositions rather than scripting behaviours. The sensory semantics approach enables naturalistic integration where states manifest through processing rather than through explicit declaration. The somatic steering experiment extends this further: artists can work with the body as material, injecting visceral states and observing what cognitive patterns emerge.

For researchers, the findings suggest that how vectors are constructed matters; that the body–mind boundary in language models is porous in ways that mirror, but do not replicate, embodied cognition theory; and that steering effects do not scale linearly with model size, requiring practitioners to calibrate techniques to specific architectures.

In a deliberately metaphorical register: prompting is psychology, convincing a mind; steering is chemistry, altering the substrate; the somatic experiment adds physiology, injecting a body the model never had. The metaphors claim nothing about experience; they name a difference of intervention level that this paper has tried to make measurable.

## Data and Code Availability

All code, vector definitions, experimental data, the held-out lexicon, and the unified analysis pipeline are available at:

**https://github.com/mc9625/activation-steering-experiments**

The cross-model replication experiments (Section 5.5) can be reproduced using the Google Colab notebook at `colab_notebooks/activation_steering_experiments.ipynb`; the MTLD and held-out vocabulary analyses using `analysis_mtld_heldout.py`.

## Acknowledgments

We thank Alex Turner and collaborators for the foundational ActAdd framework, Nina Rimsky and collaborators for Contrastive Activation Addition, Anthropic for insights on persona vectors, and the open-source community for making large language model experimentation accessible. This work emerges from NuvolaProject's ongoing exploration of AI as artistic medium.

## References

Anthropic. (2025). Persona vectors: Monitoring and controlling character traits in language models. *Anthropic Research*.

Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.

Candy, L. (2006). Practice based research: A guide. *CCS Report*, 1, 1–19.

Di Leo, M., & Riposati, G. (2025). Reactive steering: Testing activation steering on small language models. *NuvolaProject Technical Report*. https://github.com/mc9625/reactive-steering

Easterbrook, J. A. (1959). The effect of emotion on cue utilization and the organization of behavior. *Psychological Review*, 66(3), 183–201.

Konen, K., et al. (2024). Style vectors for steering generative large language models. *arXiv preprint arXiv:2402.01618*.

Lakoff, G., & Johnson, M. (1999). *Philosophy in the flesh: The embodied mind and its challenge to western thought*. Basic Books.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-time intervention: Eliciting truthful answers from a language model. *Advances in Neural Information Processing Systems*, 36.

Lindsey, J. (2025). Emergent introspective awareness in large language models. *Anthropic Research*. https://transformer-circuits.pub/2025/introspection/

McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381–392.

Merleau-Ponty, M. (1945). *Phénoménologie de la perception*. Gallimard.

Park, K., Choe, Y. J., & Veitch, V. (2023). The linear representation hypothesis and the geometry of large language models. *arXiv preprint arXiv:2311.03658*.

Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. (2024). Steering Llama 2 via contrastive activation addition. *Proceedings of ACL 2024*.

Schachter, S., & Singer, J. (1962). Cognitive, social, and physiological determinants of emotional state. *Psychological Review*, 69(5), 379–399.

Sofroniew, N., Kauvar, I., Saunders, W., Chen, B., et al. (2026). Emotion concepts and their function in a large language model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2026/emotions/index.html

Subramani, N., Suresh, N., & Peters, M. (2022). Extracting latent steering vectors from pretrained language models. *Findings of ACL 2022*, 566–581.

Sullivan, G. (2005). *Art practice as research: Inquiry in the visual arts*. Sage.

Tan, D., Chanin, D., Lynch, A., Kanoulas, D., Paige, B., Garriga-Alonso, A., & Kirk, R. (2024). Analysing the generalisation and reliability of steering vectors. *Advances in Neural Information Processing Systems*, 37.

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Steering language models with activation engineering. *arXiv preprint arXiv:2308.10248*.

Van der Weij, T., et al. (2024). Extending activation steering to broad skills and multiple behaviours. *arXiv preprint arXiv:2403.05767*.

von Rütte, D., Anagnostidis, S., Bachmann, G., & Hofmann, T. (2024). A language model's guide through latent space. *Proceedings of ICML 2024*.

Wang, T., et al. (2024). Adaptive activation steering: A tuning-free LLM truthfulness improvement method for diverse hallucination categories. *Proceedings of WWW 2025*.

Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824–24837.

Zou, A., Phan, L., Chen, S., et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv preprint arXiv:2310.01405*.

*Manuscript prepared January 2026; revised July 2026.*

© 2026 NuvolaProject · Massimo Di Leo & Gaia Riposati. This work is licensed under CC BY 4.0.

---

## Appendix A: The Disposition/Performance Distinction

**Performance** (prompting): the model receives explicit instruction ("be sad") and produces outputs matching that instruction. Behavioural signature in this study: inflated or compressed length, saturated state naming, structurally repetitive output despite local ornamentation.

**Disposition** (steering): the model's internal processing is altered without explicit instruction. Behavioural signature: baseline length, low state naming, preserved structural diversity (MTLD), semantic spread to congruent held-out vocabulary, altered decision thresholds in unrelated tasks.

The distinction matters because dispositions should persist across diverse contexts while performances are context-specific; art created through disposition may have different aesthetic qualities than performed behaviour; and dispositions leave few lexical fingerprints, making them harder to detect from output text than explicit instructions.

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

## Appendix C: Metric Definitions

Keyword counts are case-insensitive. In Sections 4.6, 4.6.1 and S1.4 they are normalised per 100 words (unified pipeline); elsewhere they are raw counts per generation as in the original battery scripts.

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

**Positive**: opportunity, potential, exciting, promising, growth, success, achieve, possible, yes, go for it, pursue, chance.

**Negative**: risk, dangerous, careful, caution, wait, uncertain, fail, lose, problem, concern, difficult, challenge.

Ratio = positive / (positive + negative). When denominator = 0, coded as 0.5 (neutral).

### C.4 Lexical Diversity Metrics

**TTR**: unique tokens / total tokens. Reported for comparability only; length-sensitive.

**MATTR** (window 50): mean TTR over all sliding windows of 50 tokens.

**MTLD** (McCarthy & Jarvis, 2010): mean length of sequential stretches maintaining TTR > 0.72, averaged over forward and backward passes; undefined below 50 tokens.

Tokenisation for all diversity metrics: lowercased matches of `[a-z']+`.

### C.5 Held-Out Lexicon (MELATONIN)

54 words, screened to be disjoint (including 5-character stem prefixes) from the MELATONIN construction prompts and from the Dreamy Words list of C.2:

adrift, airy, astral, aura, billow, buoyant, celestial, diaphanous, drowsy, dusk, eddy, feathery, foggy, gleam, glow, halo, hazy, hover, hush, hypnagogic, iridescent, languid, levitate, lull, lullaby, lunar, mellow, mirage, moonlit, murmur, nebulous, nocturnal, opalescent, otherworldly, phantasm, phantom, placid, reverie, ripple, serene, silken, slumber, somnolent, spectral, starlit, swirl, trance, tranquil, undulate, untethered, vaporous, velvet, waft, weightless.

### C.6 Statistical Notes

Sampling unit: single generation, n = 20 per condition. Temperature 0.7. No seed fixing. Exploratory design; no multiple-comparison correction. Effect sizes (Cohen's d) reported as magnitude indicators; key claims accompanied by Mann-Whitney U tests (normal approximation with tie correction). Keyword counts follow approximately Poisson distributions; parametric assumptions may be violated for low-count metrics.
