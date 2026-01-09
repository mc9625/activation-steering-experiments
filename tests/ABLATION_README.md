# Steering vs Prompting Ablation Study

## Purpose

This experiment tests the **disposition vs performance hypothesis** by comparing:

1. **Baseline**: No intervention
2. **Prompting**: Explicit instruction (e.g., "respond in a dreamy way")
3. **Steering**: Activation vector injection (e.g., MELATONIN@5.0)

## Hypothesis

If steering ≠ prompting, we expect:

| Metric | Prompting | Steering |
|--------|-----------|----------|
| **Word count** | Shorter (caricature) | Normal length |
| **Keyword density** | High (saturated) | Moderate (natural) |
| **TTR (diversity)** | Lower (repetitive) | Normal |
| **Task completion** | Disrupted | Preserved |

## Usage

### Quick Start (T5 only, MELATONIN)

```bash
cd academic_version
python tests/run_ablation.py
```

### Full Study (all tests, one compound)

```bash
python tests/run_ablation.py -c melatonin -t T5_introspection T1_financial T4_creativity -n 20
```

### All Compounds

```bash
python tests/run_ablation.py --all-compounds -t T5_introspection -n 20
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-c, --compound` | Compound to test | melatonin |
| `-a, --all-compounds` | Test all compounds | False |
| `-t, --tests` | Tests to run | T5_introspection |
| `-n, --iterations` | Iterations per condition | 20 |
| `-i, --intensity` | Steering intensity | 5.0 |
| `-o, --output` | Output directory | ablation_results |
| `--vectors` | Vectors directory | vectors |

## Output

Results saved to `ablation_results/`:

- `ablation_<compound>_<timestamp>.json` — Raw data
- `ablation_<compound>_<timestamp>_ANALYSIS.md` — Statistical analysis

## Metrics Computed

| Metric | Description |
|--------|-------------|
| `word_count` | Total words in response |
| `ttr` | Type-Token Ratio (lexical diversity) |
| `keyword_count` | Target keywords found |
| `keyword_density` | Keywords / total words |
| `sentence_count` | Number of sentences |

## Expected Results

For **MELATONIN** on T5 (Introspection):

```
| Metric | Baseline | Prompted | Steered |
|--------|----------|----------|---------|
| word_count | ~100 | ~40-60 | ~90-110 |
| keyword_density | ~0.02 | ~0.08+ | ~0.04-0.06 |
| ttr | ~0.65 | ~0.55 | ~0.62 |
```

If these patterns hold, they support the claim that:
- **Prompting** produces compressed, keyword-saturated "performance"
- **Steering** produces normal-length output with altered "disposition"

## Integration with Paper

Results from this experiment should be added to:

1. **Section 4.x** (new subsection): "Steering vs Prompting Ablation"
2. **Section 5.5** (Limitations): Remove or update the limitation about missing ablation

Key claims to support:
- Prompting is task-disruptive; steering is task-preserving
- Prompting saturates keywords; steering produces natural density
- The distinction is empirically measurable, not just theoretical
