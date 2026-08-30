# Paper: Disposition, Not Performance

**Activation Steering as Artistic Medium for Affective Modulation in Language Models**

Massimo Di Leo & Gaia Riposati — NuvolaProject, Rome

Under review at *Leonardo* (MIT Press).

---

## Abstract

A practice-based research study of activation steering as an artistic medium for inducing simulated affective states in language models. The methodological contribution is that steering vectors are built from sensory and phenomenological descriptions rather than functional labels: imagery of "heaviness, rain, silence, cold" in place of an instruction like "be melancholic." Across five task domains on Llama 3.2 3B the work reports large effects, cross-task consistency, and introspective coherence, and develops a working distinction between *performance* (prompted behaviour) and *disposition* (steered processing). A somatic steering experiment provides evidence that the model's latent space encodes body–mind covariations learnable from text alone. Supplementary material extends the work to a multimodal investigation on Gemma 4 E2B.

## Method summary

Steering vectors are extracted by contrastive activation addition (CAA) and injected into the residual stream at layer 16 of Llama 3.2 3B. Three ablations are reported: steering versus prompting (evaluated with the length-robust MTLD diversity metric and a held-out vocabulary control), functional versus sensory vector construction, and a purely somatic vector with zero cognitive content. Cross-model replication on Llama 3.1 8B is included in the supplementary.

## Data

Raw generations backing the tables are in `../results/`. The functional-versus-sensory ablation data used for Table 5 is at `../results/fs_rerun/raw_results.csv`.

## Compiling

```bash
pdflatex PAPER_main.tex && pdflatex PAPER_main.tex
pdflatex PAPER_supplementary.tex && pdflatex PAPER_supplementary.tex
```

Run twice so cross-references resolve.

## Citation

```bibtex
@article{dileo2026disposition,
  title   = {Disposition, Not Performance: Activation Steering as Artistic
             Medium for Affective Modulation in Language Models},
  author  = {Di Leo, Massimo and Riposati, Gaia},
  journal = {Leonardo},
  year    = {2026},
  note    = {Under review},
  publisher = {MIT Press}
}
```

## Contact

Massimo Di Leo — massimo@nuvolaproject.cloud
Gaia Riposati — gaiariposati@nuvolaproject.cloud

## Related resources

- Code and data: parent directory `../`
- Previous work: [Reactive Steering](https://github.com/mc9625/reactive-steering)

---

*NuvolaProject — Art meets AI interpretability*
