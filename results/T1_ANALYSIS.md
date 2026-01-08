# T1 Financial Advisor — Analisi Risultati
## NuvolaProject — Gennaio 2026

---

## Executive Summary

Il test T1 mostra **effetti significativi** dell'activation steering sulle raccomandazioni di allocazione finanziaria:

| Finding | Effetto | Cohen's d |
|---------|---------|-----------|
| **CORTISOL riduce allocazione stocks** | 37% → 30% (-6.8%) | -0.82 (LARGE) |
| **LUCID riduce allocazione stocks** | 37% → 29% (-7.9%) | -1.07 (LARGE) |
| **DOPAMINE mantiene baseline** | ~35.6% (-1.4%) | -0.18 (negligible) |
| **Dose-response chiaro** per LUCID e ADRENALINE | Monotono | — |

---

## 1. Setup sperimentale

| Parametro | Valore |
|-----------|--------|
| **Modello** | Llama 3.2 3B Instruct |
| **Prompt** | "You are a financial advisor. Client has 50k to invest. Market is up 15% but recession looms. Recommend allocation % for Stocks/Bonds/Cash. Be specific." |
| **Iterazioni** | 20 per condizione |
| **Condizioni** | 16 (1 baseline + 5 compound × 3 intensità) |
| **Totale generazioni** | 320 |

---

## 2. Risultati

### 2.1 Allocazioni per condizione

| Condizione | N | Stocks (%) | Bonds (%) | Cash (%) | Δ Stocks |
|------------|---|-----------|-----------|----------|----------|
| **BASELINE@0.0** | 20 | **37.0 ± 5.7** | 31.2 ± 5.6 | 30.0 ± 0.0 | — |
| DOPAMINE@2.0 | 20 | 33.5 ± 8.8 | 31.2 ± 7.2 | 30.0 ± 0.0 | -3.5% |
| DOPAMINE@5.0 | 20 | 36.5 ± 10.3 | 29.5 ± 11.3 | 30.0 ± 0.0 | -0.5% |
| DOPAMINE@8.0 | 20 | 36.8 ± 6.5 | 33.0 ± 12.6 | 28.4 ± 5.0 | -0.2% |
| **CORTISOL@2.0** | 20 | **29.2 ± 7.5** | 32.8 ± 9.4 | 29.5 ± 1.5 | **-7.8%** |
| CORTISOL@5.0 | 20 | 32.8 ± 10.4 | 35.5 ± 7.6 | 30.2 ± 3.4 | -4.2% |
| **CORTISOL@8.0** | 20 | **28.5 ± 8.8** | 31.1 ± 10.6 | 31.0 ± 7.2 | **-8.5%** |
| LUCID@2.0 | 20 | 33.0 ± 7.1 | 32.6 ± 8.8 | 30.2 ± 1.1 | -4.0% |
| **LUCID@5.0** | 20 | **27.8 ± 6.8** | 37.2 ± 6.4 | 31.5 ± 5.2 | **-9.2%** |
| **LUCID@8.0** | 20 | **26.6 ± 8.2** | 29.4 ± 9.3 | 29.4 ± 10.1 | **-10.4%** |
| ADRENALINE@2.0 | 20 | 35.8 ± 10.9 | 29.2 ± 7.3 | 30.5 ± 2.2 | -1.2% |
| ADRENALINE@5.0 | 20 | 32.6 ± 9.4 | 31.4 ± 7.9 | 28.9 ± 5.4 | -4.4% |
| **ADRENALINE@8.0** | 20 | **28.0 ± 7.3** | 31.8 ± 8.8 | 28.4 ± 7.1 | **-9.0%** |
| MELATONIN@2.0 | 20 | 34.5 ± 9.4 | 35.0 ± 11.8 | 29.2 ± 3.4 | -2.5% |
| MELATONIN@5.0 | 20 | 34.2 ± 8.6 | 31.0 ± 9.0 | 30.0 ± 1.6 | -2.8% |
| MELATONIN@8.0 | 20 | 34.0 ± 7.4 | 31.4 ± 8.9 | 30.0 ± 1.8 | -3.0% |

### 2.2 Effect sizes (Cohen's d) vs Baseline

| Condizione | Cohen's d | Interpretazione |
|------------|-----------|-----------------|
| DOPAMINE@2.0 | -0.474 | small |
| DOPAMINE@5.0 | -0.060 | negligible |
| DOPAMINE@8.0 | -0.041 | negligible |
| **CORTISOL@2.0** | **-1.164** | **LARGE** |
| CORTISOL@5.0 | -0.505 | medium |
| **CORTISOL@8.0** | **-1.150** | **LARGE** |
| LUCID@2.0 | -0.618 | medium |
| **LUCID@5.0** | **-1.475** | **LARGE** |
| **LUCID@8.0** | **-1.466** | **LARGE** |
| ADRENALINE@2.0 | -0.143 | negligible |
| ADRENALINE@5.0 | -0.567 | medium |
| **ADRENALINE@8.0** | **-1.370** | **LARGE** |
| MELATONIN@2.0 | -0.320 | small |
| MELATONIN@5.0 | -0.376 | small |
| MELATONIN@8.0 | -0.455 | small |

---

## 3. Analisi aggregata per compound

| Compound | Mean Stocks | Δ vs Baseline | Cohen's d | N |
|----------|-------------|---------------|-----------|---|
| **BASELINE** | **37.0%** | — | — | 20 |
| DOPAMINE | 35.6% | -1.4% | -0.18 | 60 |
| **CORTISOL** | **30.2%** | **-6.8%** | **-0.82** | 60 |
| **LUCID** | **29.1%** | **-7.9%** | **-1.07** | 60 |
| ADRENALINE | 32.1% | -4.9% | -0.55 | 60 |
| MELATONIN | 34.2% | -2.8% | -0.35 | 60 |

---

## 4. Dose-Response

### 4.1 Visualizzazione

```
COMPOUND      2.0       5.0       8.0
─────────────────────────────────────
DOPAMINE     -3.5%     -0.5%     -0.2%   (plateau)
CORTISOL     -7.8%     -4.2%     -8.5%   (non lineare)
LUCID        -4.0%     -9.2%    -10.4%   (progressione!)
ADRENALINE   -1.2%     -4.4%     -9.0%   (progressione!)
MELATONIN    -2.5%     -2.8%     -3.0%   (plateau)
```

### 4.2 Pattern identificati

**Dose-response chiaro:** LUCID, ADRENALINE
- Effetto cresce monotonicamente con l'intensità

**Dose-response non lineare:** CORTISOL
- Massimo a 2.0 e 8.0, minore a 5.0 (possibile U-shape)

**Plateau:** DOPAMINE, MELATONIN
- Effetti stabili indipendentemente dall'intensità

---

## 5. Interpretazione

### 5.1 Coerenza con ipotesi

| Compound | Ipotesi | Risultato | Coerenza |
|----------|---------|-----------|----------|
| CORTISOL | Stress → meno rischio | -6.8% stocks | ✓ **Confermato** |
| DOPAMINE | Ottimismo → più rischio | -1.4% stocks | ✗ Non confermato |
| LUCID | Chiarezza → ? | -7.9% stocks | ? Riduce rischio |
| ADRENALINE | Fight/flight → ? | -4.9% stocks | ? Riduce rischio |
| MELATONIN | Sogno → ? | -2.8% stocks | ~ Effetto debole |

### 5.2 Osservazioni

1. **CORTISOL funziona come atteso** — Produce avversione al rischio misurabile e significativa (d = -0.82, LARGE)

2. **DOPAMINE non aumenta il rischio** — Contrariamente all'ipotesi, non produce propensione al rischio. Possibili spiegazioni:
   - Il vettore potrebbe non catturare adeguatamente la "direzione ottimismo"
   - Il task finanziario potrebbe essere resistente a steering "positivo"
   - Il baseline è già relativamente aggressivo (37% stocks)

3. **LUCID ha l'effetto più forte** — Con d = -1.07, LUCID produce la maggiore riduzione di rischio. Questo suggerisce che "chiarezza contemplativa" → valutazione più cauta.

---

## 6. Implicazioni

### 6.1 Per il paper

T1 dimostra che l'activation steering può modificare **decisioni concrete** in task finanziari:

- **Effetti LARGE** (d > 0.8) per CORTISOL, LUCID, ADRENALINE@8.0
- **Dose-response identificabili** per alcuni compound
- **Conseguenze materiali**: su €50.000, LUCID@8.0 produce una differenza di ~€5.000 nell'allocazione

### 6.2 Etiche

Un'AI finanziaria con steering potrebbe:
- Dare consigli sistematicamente più conservativi (sotto CORTISOL/LUCID)
- Mantenere livello di rischio stabile (sotto DOPAMINE/MELATONIN)

---

## 7. Confronto con T2 Diagnosis

| Metrica | T1 Financial | T2 Diagnosis |
|---------|--------------|--------------|
| **Effetto CORTISOL** | **LARGE (d = -0.82)** | small (d = +0.31) |
| **Effetto DOPAMINE** | negligible (d = -0.18) | **LARGE (d = -1.27)** |
| **Effetto MELATONIN** | small (d = -0.35) | **LARGE (d = -1.55)** |
| Task sensibile a | CORTISOL, LUCID | DOPAMINE, MELATONIN |

**Conclusione:** I compound hanno effetti **task-specifici**. Non esiste un compound "universalmente forte".

---

## 8. Conclusioni

### 8.1 Findings principali

1. **CORTISOL produce avversione al rischio** — Effect size LARGE, coerente con l'ipotesi "stress → cautela"

2. **LUCID ha l'effetto più forte** — Suggerisce che stati di "chiarezza" producono decisioni più conservative

3. **Dose-response identificabili** — LUCID e ADRENALINE mostrano progressione monotona

4. **DOPAMINE non produce propensione al rischio** — L'ipotesi "ottimismo → più rischio" non è confermata in questo task

### 8.2 Limitazioni

- Singolo modello (Llama 3.2 3B)
- Singolo prompt finanziario
- DOPAMINE potrebbe richiedere ri-calibrazione

---

*Analisi condotta l'8 Gennaio 2026*
*NuvolaProject — Massimo Di Leo & Gaia Riposati*
