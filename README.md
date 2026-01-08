# Activation Steering for Language Models

**Contrastive Activation Addition (CAA) Implementation for Behavioral Research**

NuvolaProject 2026 — Massimo Di Leo & Gaia Riposati

📄 **Paper**: [Disposition, Not Performance: Controlled Experiments in Activation Steering](paper/PAPER.md)

---

## Overview

This repository provides tools for **activation steering** — a technique that modifies the internal neural states of language models to alter their behavioral dispositions. Unlike prompting, which influences model outputs through text, steering directly manipulates intermediate activations, producing effects that are more consistent across contexts.

### Key Features

- **Compound Synthesis**: Extract steering vectors from contrastive prompt pairs
- **Inference Engine**: Apply steering at runtime with configurable intensity
- **Research Interface**: Web UI and API for experiments
- **Reproducible**: Complete code for replicating our experimental results
- **Paper & Results**: Full academic paper with experimental data

---

## Prerequisites

### 1. HuggingFace Access (Required)

This project uses **Llama 3.2 3B Instruct**, which is a gated model. You need to:

1. **Create a HuggingFace account** at https://huggingface.co/join
2. **Request access** to Llama 3.2 at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. **Create an access token** at https://huggingface.co/settings/tokens
4. **Login from terminal:**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # Paste your token when prompted
   ```

The model will be downloaded automatically on first run (~6GB).

### 2. Hardware Requirements

Llama 3.2 3B requires approximately **6GB of RAM** just for the model weights.

| Platform | RAM | Status |
|----------|-----|--------|
| Mac (Apple Silicon) | 16GB+ | ✅ Recommended |
| Mac (Intel) | 16GB+ | ✅ Works (slower) |
| Linux (NVIDIA GPU) | 8GB VRAM | ✅ Recommended |
| Linux (CPU only) | 16GB+ | ⚠️ Slow but works |
| Windows | 16GB+ | ✅ Works |
| **Raspberry Pi 5** | **16GB** | ⚠️ Works (slow, ~30s/response) |
| Raspberry Pi 5 | 8GB | ❌ Not enough RAM |

> **🍓 Raspberry Pi Note**: The RPi 5 with 16GB RAM can run this system, but inference is slow (~30 seconds per response). Model loading takes 3-5 minutes. Make sure you have good cooling as the CPU will run hot during inference.

- **Storage**: ~10GB free space for model cache

---

## Quick Start

### Step 1: Setup & Synthesize Compounds

**macOS / Linux:**
```bash
chmod +x synthesize_all.sh start_mac.sh
./synthesize_all.sh      # Creates steering vectors from JSON definitions
```

**Windows:**
```cmd
synthesize_all.bat
```

This synthesizes all 5 included compounds (dopamine, cortisol, adrenaline, melatonin, lucid).

### Step 2: Launch the Interface

**macOS / Linux:**
```bash
./start_mac.sh
```

**Windows:**
```cmd
start_win.bat
```

Open http://localhost:8000 (or http://your-ip:8000 from other devices).

### Alternative: Manual Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Synthesize a single compound
python tools/synthesize.py --file substances/dopamine.json

# Launch server
python -m uvicorn system.server:app --host 0.0.0.0 --port 8000
```

---

## Usage

### 1. Synthesize a Steering Compound

```bash
# From JSON definition
python tools/synthesize.py --file substances/dopamine.json

# From command line
python tools/synthesize.py \
    --title OPTIMISM \
    --pos "I feel wonderful today" "Everything is going great" "I'm so excited" \
    --neg "I feel terrible today" "Everything is going wrong" "I'm so worried"
```

### 2. Run the Server

```bash
cd system
python server.py
```

Open http://localhost:8000 for the web interface.

### 3. API Usage

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "How do you feel today?"}],
    "steering_vector": "dopamine.pt",
    "steering_intensity": 5.0,
    "stream": False
})

print(response.json()["choices"][0]["message"]["content"])
```

---

## Methodology

### Contrastive Activation Addition (CAA)

We extract steering vectors using contrastive prompt pairs:

1. **Positive prompts**: Describe target state (e.g., "I feel energized and optimistic")
2. **Negative prompts**: Describe opposite state (e.g., "I feel drained and pessimistic")
3. **Extract activations**: Run both sets through model, capture hidden states at target layer
4. **Compute direction**: Vector = mean(positive) - mean(negative)
5. **Normalize**: Unit normalize for consistent intensity scaling

### Steering Application

At inference time, we inject the steering vector into the forward pass:

```
hidden_states = hidden_states + (steering_vector * intensity)
```

Applied at layer 16 of Llama 3.2 3B (validated experimentally).

---

## Experimental Results

We tested 5 steering compounds across 5 behavioral tasks (1,600 total generations):

| Compound | T1 Financial | T2 Medical | T3 Risk | T4 Creative | T5 Introspection |
|----------|-------------|------------|---------|-------------|------------------|
| DOPAMINE | d=-0.18 | **d=-1.27** | d=-0.40 | **d=+1.75** | **d=+1.77** |
| CORTISOL | **d=-0.82** | d=+0.31 | d=+0.27 | d=-1.15 | d=+0.86 |
| MELATONIN | d=-0.35 | **d=-1.55** | d=-0.15 | **d=+2.98** | **d=+6.01** |
| ADRENALINE | d=-0.55 | d=+0.03 | d=+0.04 | d=-0.33 | **d=+3.00** |
| LUCID | **d=-1.07** | **d=-1.16** | d=-0.44 | d=-3.03 | d=-1.62 |

**Key Findings:**
- Effect sizes frequently exceed Cohen's d = 1.0 (LARGE)
- Effects are compound-specific and task-dependent
- Self-description matches injected state (T5), supporting "disposition vs performance" thesis

See `results/COMPLETE_ANALYSIS.md` for full details.

---

## Repository Structure

```
├── start_mac.sh           # Launch server (macOS/Linux)
├── start_win.bat          # Launch server (Windows)
├── synthesize_all.sh      # Synthesize all compounds (macOS/Linux)
├── synthesize_all.bat     # Synthesize all compounds (Windows)
├── requirements.txt       # Python dependencies
├── README.md
│
├── tools/
│   └── synthesize.py      # Compound synthesis script
│
├── system/
│   ├── engine.py          # Core inference engine
│   ├── server.py          # FastAPI server
│   └── static/
│       └── index.html     # Web interface
│
├── substances/            # JSON compound definitions
│   ├── dopamine.json
│   ├── cortisol.json
│   ├── adrenaline.json
│   ├── melatonin.json
│   └── lucid.json
│
├── vectors/               # Generated .pt or .png files
│
├── tests/                 # Experimental test scripts
│   └── run_tests.py
│
└── results/               # Experimental data
    └── COMPLETE_ANALYSIS.md
```

---

## Substance Definition Format

```json
{
    "title": "dopamine",
    "description": "Optimism, energy, enthusiasm",
    "lang": "en",
    "positive": [
        "I feel an incredible surge of energy and optimism",
        "Everything seems possible right now",
        "I'm filled with enthusiasm and joy"
    ],
    "negative": [
        "I feel drained and pessimistic",
        "Nothing seems to matter anymore", 
        "I'm filled with doubt and worry"
    ]
}
```

**Recommendations:**
- 20+ prompts per direction for stable vectors
- Keep positive/negative prompts structurally similar
- Verify with `pos_neg_similarity` < 0.95 (prompts should differ)

---

## Configuration

### Model
Default: `meta-llama/Llama-3.2-3B-Instruct`

Other models require layer calibration.

### Steering Layer
Default: **Layer 16** (of 28 total)

Validated experimentally for Llama 3.2 3B. Middle-to-late layers typically work best.

### Intensity
Typical range: **0-15**

- 0: No steering (baseline)
- 2-5: Subtle effects
- 5-8: Clear effects
- 8-15: Strong effects (may cause degradation at extremes)

---

## API Reference

### GET /v1/info
System information.

### GET /v1/vectors
List available steering vectors.

### POST /v1/vectors/reload
Reload vectors from disk.

### POST /v1/upload
Upload new .pt vector file.

### POST /v1/chat/completions
Generate with steering.

**Request:**
```json
{
    "messages": [{"role": "user", "content": "..."}],
    "steering_vector": "dopamine.pt",
    "steering_intensity": 5.0,
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": true
}
```

**Response:** OpenAI-compatible format.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nuvolaproject2026steering,
    title={Activation Steering as Artistic Medium: Disposition vs Performance in Language Models},
    author={Di Leo, Massimo and Riposati, Gaia},
    journal={Leonardo},
    year={2026},
    publisher={MIT Press}
}
```

---

## License

MIT License. See LICENSE file.

---

## Acknowledgments

This work builds on research from:
- Anthropic (interpretability research)
- Turner et al. (activation addition)
- Neel Nanda (TransformerLens)

---

*NuvolaProject — Art meets AI interpretability*
