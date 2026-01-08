# Paper: Disposition, Not Performance

**Controlled Experiments in Activation Steering for Affective Modulation of Language Models**

Massimo Di Leo & Gaia Riposati — NuvolaProject, January 2026

---

## Files

| File | Description |
|------|-------------|
| `PAPER.md` | Full paper in Markdown format |
| `paper.tex` | LaTeX source for journal submission |

## Abstract

We present a controlled experimental study of activation steering across five task domains (financial, medical, risk, creative, introspective) using five semantically-grounded steering vectors. Results demonstrate large effects (Cohen's d > 1.0), compound-specific profiles, introspective coherence, and dose-response relationships—supporting the distinction between behavioral *performance* and *dispositional* change.

## Key Results

| Finding | Evidence |
|---------|----------|
| Large effects | 37% of conditions show d > 0.8 |
| Cross-task consistency | Same compound → coherent effects across domains |
| Introspective coherence | Steered models describe matching inner states |
| Dose-response | Monotonic intensity scaling observed |

## Strongest Effects

| Compound | Task | Cohen's d |
|----------|------|-----------|
| MELATONIN | T5 Introspection (dreamy) | **+6.01** |
| ADRENALINE | T5 Introspection (urgent) | **+3.00** |
| MELATONIN | T4 Creativity (dreamy) | **+2.98** |
| MELATONIN | T2 Medical (alarm ↓) | **-2.48** |
| DOPAMINE | T4 Creativity (enthusiasm) | **+1.75** |

## Compiling LaTeX

```bash
pdflatex paper.tex
# or
latexmk -pdf paper.tex
```

## Citation

```bibtex
@article{dileo2026disposition,
  title={Disposition, Not Performance: Controlled Experiments in Activation Steering for Affective Modulation of Language Models},
  author={Di Leo, Massimo and Riposati, Gaia},
  journal={arXiv preprint},
  year={2026}
}
```

## Related Resources

- **Code & Data**: See parent directory `../` for full implementation
- **Previous work**: [Reactive Steering](https://github.com/mc9625/reactive-steering)

---

*NuvolaProject 2026*
