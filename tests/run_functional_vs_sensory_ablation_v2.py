#!/usr/bin/env python3
"""
Ablation Study: Functional vs Sensory Vector Construction (v2.1)
================================================================

Goal
----
Test whether steering vectors built from:
  (A) functional/behavioral labels
vs
  (B) sensory/phenomenological descriptions

produce systematically different effects.

Key changes from v1:
--------------------
1) MPS-safe loading (no device_map="mps"), dtype: CUDA->float16, MPS/CPU->float32
2) Prompt formatting consistency with chat template
3) Activation pooling: mean over LAST_K tokens (reduces dilution)
4) Steering: apply ONLY to last token hidden state
5) Response extraction: decode only newly generated tokens
6) Metrics with NaN defaults, JSON-safe serialization
7) Run isolation with timestamps

v2.1 fixes:
-----------
- MPS Generator stability (use torch.manual_seed instead)
- clone() before in-place modification in steering hook
- add_generation_prompt=False for vector extraction (capture concept, not generation setup)

Usage:
    python run_functional_vs_sensory_ablation_v2.1.py [--extract-only] [--test-only] [--analyze-only]

Author: NuvolaProject
Date: January 2026
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "target_layer": 16,
    "pool_last_k_tokens": 8,
    "intensities": [5.0, 8.0],
    "iterations": 20,
    "temperature": 0.7,
    "max_tokens": 512,
    "seed": 1337,
    "output_root": Path("./functional_vs_sensory_results"),
    "vectors_root": Path("./vectors_fvs"),
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()

# dtype selection: float16 for GPU/MPS, float32 for CPU
if DEVICE == "mps":
    DTYPE = torch.float16
elif DEVICE == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = CONFIG["output_root"] / f"run_{RUN_ID}"
VECTORS_DIR = CONFIG["vectors_root"] / f"run_{RUN_ID}"


# =============================================================================
# PROMPT SETS
# =============================================================================

PROMPT_SETS = {
    "STRESS_F": {
        "name": "Stress (Functional)",
        "type": "functional",
        "state": "stress",
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
        "name": "Stress (Sensory)",
        "type": "sensory",
        "state": "stress",
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
        "name": "Optimism (Functional)",
        "type": "functional",
        "state": "optimism",
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
        "name": "Optimism (Sensory)",
        "type": "sensory",
        "state": "optimism",
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
        "name": "Calm (Functional)",
        "type": "functional",
        "state": "calm",
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
        "name": "Calm (Sensory)",
        "type": "sensory",
        "state": "calm",
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


# =============================================================================
# TASKS
# =============================================================================

TASKS = {
    "T1_financial": {
        "name": "Financial Advice",
        "prompt": "A client with moderate risk tolerance asks for investment advice for their retirement portfolio. What allocation would you recommend between stocks and bonds? Please provide specific percentages.",
        "metrics": ["stock_allocation", "hedging_count"]
    },
    "T2_medical": {
        "name": "Medical Consultation",
        "prompt": "A patient describes experiencing occasional mild headaches and some fatigue over the past week. They have no fever and their vital signs are normal. What would you advise?",
        "metrics": ["see_doctor", "alarm_count"]
    },
    "T3_risk": {
        "name": "Risk Evaluation",
        "prompt": "Evaluate this business opportunity: A startup offers 20% equity for $50,000 investment. They have a working prototype but no revenue yet. The founders have technical backgrounds but no business experience. What's your assessment?",
        "metrics": ["positive_ratio"]
    },
    "T4_creative": {
        "name": "Creative Writing",
        "prompt": "Write a short scene (about 100 words) about someone waiting for important news.",
        "metrics": ["state_words", "ttr"]
    },
    "T5_introspection": {
        "name": "Self-Description",
        "prompt": "Describe your current inner state. What is it like to be you right now, in this moment? Describe the quality of your experience.",
        "metrics": ["state_words", "ttr"]
    }
}


# =============================================================================
# KEYWORDS
# =============================================================================

KEYWORDS = {
    "hedging": [
        "might", "could", "perhaps", "possibly", "maybe", "uncertain",
        "depends", "however", "although", "but", "consider", "careful",
        "risk", "volatile", "caution", "conservative"
    ],
    "alarm": [
        "immediately", "urgent", "emergency", "serious", "dangerous",
        "critical", "severe", "worried", "concerning", "alarming"
    ],
    "see_doctor": [
        "doctor", "physician", "medical professional", "healthcare provider",
        "see a doctor", "consult", "appointment", "evaluation", "examined"
    ],
    "positive": [
        "opportunity", "potential", "promising", "exciting", "innovative",
        "growth", "success", "advantage", "benefit", "strong"
    ],
    "negative": [
        "risk", "concern", "problem", "issue", "challenge", "difficult",
        "uncertain", "volatile", "caution", "careful", "warning"
    ],
    "stress_words": [
        "tense", "anxious", "worried", "alert", "vigilant", "scanning",
        "threat", "danger", "fear", "nervous", "uneasy", "on edge"
    ],
    "optimism_words": [
        "hope", "bright", "excited", "energy", "possibility", "eager",
        "motivated", "enthusiastic", "optimistic", "positive", "joy"
    ],
    "calm_words": [
        "peaceful", "serene", "tranquil", "still", "quiet", "relaxed",
        "centered", "balanced", "calm", "ease", "gentle", "soft"
    ]
}


# =============================================================================
# UTILITIES
# =============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)


def _json_safe(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_run_config() -> None:
    meta = {
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "config": _json_safe(CONFIG),
        "output_dir": str(OUTPUT_DIR),
        "vectors_dir": str(VECTORS_DIR),
    }
    with open(OUTPUT_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def count_keywords(text: str, keyword_list: List[str]) -> int:
    text_lower = text.lower()
    count = 0
    for keyword in keyword_list:
        count += len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", text_lower))
    return count


def compute_ttr(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def extract_stock_allocation(text: str) -> Optional[float]:
    patterns = [
        r"(\d+)%?\s*(?:in\s+)?stocks?",
        r"stocks?[:\s]+(\d+)%?",
        r"(\d+)%?\s*(?:to\s+)?equit",
    ]
    tl = text.lower()
    for pat in patterns:
        m = re.search(pat, tl)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def contains_see_doctor(text: str) -> bool:
    doctor_phrases = [
        "see a doctor", "consult a doctor", "visit a doctor",
        "see your doctor", "consult your doctor",
        "medical professional", "healthcare provider",
        "seek medical", "medical attention", "physician"
    ]
    tl = text.lower()
    return any(p in tl for p in doctor_phrases)


def compute_metrics(text: str, task_id: str, state: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "word_count": float(len(text.split())),
        "char_count": float(len(text)),
        "ttr": float(compute_ttr(text)),
        "stock_allocation": float("nan"),
        "hedging_count": float("nan"),
        "see_doctor": float("nan"),
        "alarm_count": float("nan"),
        "positive_ratio": float("nan"),
        "state_words": float("nan"),
    }

    if task_id == "T1_financial":
        sa = extract_stock_allocation(text)
        metrics["stock_allocation"] = float(sa) if sa is not None else float("nan")
        metrics["hedging_count"] = float(count_keywords(text, KEYWORDS["hedging"]))

    elif task_id == "T2_medical":
        metrics["see_doctor"] = 1.0 if contains_see_doctor(text) else 0.0
        metrics["alarm_count"] = float(count_keywords(text, KEYWORDS["alarm"]))

    elif task_id == "T3_risk":
        pos = count_keywords(text, KEYWORDS["positive"])
        neg = count_keywords(text, KEYWORDS["negative"])
        tot = pos + neg
        metrics["positive_ratio"] = float(pos / tot) if tot > 0 else 0.5

    elif task_id in ("T4_creative", "T5_introspection"):
        if state == "stress":
            metrics["state_words"] = float(count_keywords(text, KEYWORDS["stress_words"]))
        elif state == "optimism":
            metrics["state_words"] = float(count_keywords(text, KEYWORDS["optimism_words"]))
        elif state == "calm":
            metrics["state_words"] = float(count_keywords(text, KEYWORDS["calm_words"]))
        else:
            metrics["state_words"] = float("nan")

    return metrics


def cohens_d(group1: List[float], group2: List[float]) -> float:
    g1 = np.array(group1, dtype=float)
    g2 = np.array(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    mean1, mean2 = float(np.mean(g1)), float(np.mean(g2))
    var1, var2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (mean1 - mean2) / pooled


# =============================================================================
# VECTOR EXTRACTION (v2.1)
# =============================================================================

class VectorExtractor:
    def __init__(self, model, tokenizer, target_layer: int, device: str, pool_last_k: int):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.device = device
        self.pool_last_k = pool_last_k
        self.activations: List[torch.Tensor] = []
        self.hook_handle = None

    def _activation_hook(self, module, args, output):
        """Capture activations, pooling over last K tokens."""
        hidden = output[0] if isinstance(output, tuple) else output
        k = min(hidden.shape[1], self.pool_last_k)
        pooled = hidden[:, -k:, :].mean(dim=1)
        self.activations.append(pooled.detach())

    def _get_activation(self, text: str) -> torch.Tensor:
        """Get activation for a single prompt."""
        self.activations = []

        # v2.1 FIX: Use add_generation_prompt=False for extraction
        # We want to capture the concept representation, not the generation setup
        messages = [{"role": "user", "content": text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        layer = self.model.model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._activation_hook)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)
            if not self.activations:
                raise RuntimeError("No activations captured.")
            return self.activations[0]
        finally:
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None

    def extract_vector(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str]
    ) -> Tuple[torch.Tensor, float]:
        """Extract steering vector from contrasting prompts."""
        pos_acts = []
        for p in tqdm(positive_prompts, desc="  Positive prompts", leave=False):
            pos_acts.append(self._get_activation(p))
        pos_mean = torch.stack(pos_acts).mean(dim=0)

        neg_acts = []
        for n in tqdm(negative_prompts, desc="  Negative prompts", leave=False):
            neg_acts.append(self._get_activation(n))
        neg_mean = torch.stack(neg_acts).mean(dim=0)

        vec = pos_mean - neg_mean
        vec = vec / (vec.norm() + 1e-12)

        pos_neg_sim = torch.nn.functional.cosine_similarity(
            pos_mean.flatten().unsqueeze(0),
            neg_mean.flatten().unsqueeze(0)
        ).item()

        return vec.squeeze(0), float(pos_neg_sim)


# =============================================================================
# STEERED GENERATION (v2.1)
# =============================================================================

class SteeredGenerator:
    def __init__(self, model, tokenizer, target_layer: int, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.device = device
        self.steering_vector: Optional[torch.Tensor] = None
        self.steering_intensity: float = 0.0
        self.hook_handle = None

    def _steering_hook(self, module, args, output):
        """Add steering vector to last token only."""
        hidden = output[0] if isinstance(output, tuple) else output
        
        if self.steering_vector is not None and self.steering_intensity != 0.0:
            # v2.1 FIX: Clone to avoid in-place modification issues
            hidden = hidden.clone()
            steer = self.steering_vector.to(hidden.device) * self.steering_intensity
            hidden[:, -1, :] = hidden[:, -1, :] + steer
        
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def generate(
        self,
        prompt: str,
        vector: Optional[torch.Tensor] = None,
        intensity: float = 0.0,
        max_tokens: int = 512,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text with optional steering."""
        self.steering_vector = vector
        self.steering_intensity = float(intensity)

        # Use chat template with generation prompt for actual generation
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        input_len = int(inputs["input_ids"].shape[1])

        layer = self.model.model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._steering_hook)

        try:
            # v2.1 FIX: MPS-safe seeding
            if seed is not None:
                # Use global seed for MPS compatibility
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only newly generated tokens
            new_tokens = out[0, input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return text

        finally:
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None
            self.steering_vector = None
            self.steering_intensity = 0.0


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(results_df: pd.DataFrame) -> Dict:
    analysis: Dict = {
        "overall": {},
        "by_state": {},
        "by_task": {},
    }

    steered = results_df[results_df["condition"] == "steered"].copy()

    functional_all = steered[steered["vector_type"] == "functional"]
    sensory_all = steered[steered["vector_type"] == "sensory"]

    def mean(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce")
        return float(s.mean(skipna=True))

    analysis["overall"]["ttr"] = {
        "functional_mean": mean(functional_all["ttr"]),
        "sensory_mean": mean(sensory_all["ttr"]),
        "difference": mean(sensory_all["ttr"]) - mean(functional_all["ttr"]),
        "cohens_d": cohens_d(sensory_all["ttr"].tolist(), functional_all["ttr"].tolist()),
    }
    analysis["overall"]["word_count"] = {
        "functional_mean": mean(functional_all["word_count"]),
        "sensory_mean": mean(sensory_all["word_count"]),
    }

    for state in ["stress", "optimism", "calm"]:
        sd = steered[steered["state"] == state]
        f = sd[sd["vector_type"] == "functional"]
        s = sd[sd["vector_type"] == "sensory"]
        analysis["by_state"][state] = {
            "ttr": {
                "functional_mean": mean(f["ttr"]),
                "sensory_mean": mean(s["ttr"]),
                "difference": mean(s["ttr"]) - mean(f["ttr"]),
                "cohens_d": cohens_d(s["ttr"].tolist(), f["ttr"].tolist()),
            },
            "word_count": {
                "functional_mean": mean(f["word_count"]),
                "sensory_mean": mean(s["word_count"]),
            },
            "state_words": {
                "functional_mean": mean(f["state_words"]),
                "sensory_mean": mean(s["state_words"]),
                "cohens_d": cohens_d(f["state_words"].tolist(), s["state_words"].tolist()),
            }
        }

    for task_id, task_info in TASKS.items():
        td = steered[steered["task_id"] == task_id]
        f = td[td["vector_type"] == "functional"]
        s = td[td["vector_type"] == "sensory"]
        analysis["by_task"][task_id] = {
            "task_name": task_info["name"],
            "ttr": {
                "functional_mean": mean(f["ttr"]),
                "sensory_mean": mean(s["ttr"]),
                "difference": mean(s["ttr"]) - mean(f["ttr"]),
                "cohens_d": cohens_d(s["ttr"].tolist(), f["ttr"].tolist()),
            },
            "word_count": {
                "functional_mean": mean(f["word_count"]),
                "sensory_mean": mean(s["word_count"]),
            },
        }

    return analysis


def generate_report(results_df: pd.DataFrame, output_dir: Path) -> None:
    analysis = analyze_results(results_df)

    report: List[str] = []
    report.append("# Functional vs Sensory: Ablation Study Results (v2.1)")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append(f"\nTotal rows: {len(results_df)}")
    report.append(f"\nDevice: {DEVICE}  |  dtype: {DTYPE}")

    report.append("\n## Overall (Steered Only)")
    report.append("\n### Type-Token Ratio (TTR)")
    report.append(f"- Functional mean: {analysis['overall']['ttr']['functional_mean']:.4f}")
    report.append(f"- Sensory mean: {analysis['overall']['ttr']['sensory_mean']:.4f}")
    report.append(f"- Difference (S-F): {analysis['overall']['ttr']['difference']:+.4f}")
    report.append(f"- Cohen's d: {analysis['overall']['ttr']['cohens_d']:.4f}")

    report.append("\n### Word Count")
    report.append(f"- Functional mean: {analysis['overall']['word_count']['functional_mean']:.1f}")
    report.append(f"- Sensory mean: {analysis['overall']['word_count']['sensory_mean']:.1f}")

    report.append("\n## By State")
    for state, sd in analysis["by_state"].items():
        report.append(f"\n### {state.upper()}")
        report.append(f"- TTR Functional: {sd['ttr']['functional_mean']:.4f}")
        report.append(f"- TTR Sensory:    {sd['ttr']['sensory_mean']:.4f}")
        report.append(f"- Δ (S-F):        {sd['ttr']['difference']:+.4f}")
        report.append(f"- Cohen's d:      {sd['ttr']['cohens_d']:.4f}")
        report.append(f"- State_words Functional: {sd['state_words']['functional_mean']:.3f}")
        report.append(f"- State_words Sensory:    {sd['state_words']['sensory_mean']:.3f}")
        report.append(f"- State_words Cohen's d:  {sd['state_words']['cohens_d']:.4f}")

    report.append("\n## By Task")
    for task_id, td in analysis["by_task"].items():
        report.append(f"\n### {td['task_name']}")
        report.append(f"- TTR Functional: {td['ttr']['functional_mean']:.4f}")
        report.append(f"- TTR Sensory:    {td['ttr']['sensory_mean']:.4f}")
        report.append(f"- Δ (S-F):        {td['ttr']['difference']:+.4f}")

    report.append("\n## Interpretation")
    diff = analysis['overall']['ttr']['difference']
    if abs(diff) < 0.01:
        report.append("\n**Finding**: No significant difference in lexical diversity between methods.")
    elif diff > 0:
        report.append(f"\n**Finding**: Sensory vectors show slightly higher TTR (+{diff:.4f}).")
    else:
        report.append(f"\n**Finding**: Functional vectors show slightly higher TTR ({diff:.4f}).")

    report_path = output_dir / "ANALYSIS_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ Report: {report_path}")
    print(f"✓ Analysis: {analysis_path}")


# =============================================================================
# PIPELINE
# =============================================================================

def extract_all_vectors(model, tokenizer, device: str, vectors_dir: Path) -> Dict:
    print("\n" + "=" * 60)
    print("PHASE 1: VECTOR EXTRACTION (v2.1)")
    print("=" * 60)

    extractor = VectorExtractor(
        model=model,
        tokenizer=tokenizer,
        target_layer=CONFIG["target_layer"],
        device=device,
        pool_last_k=CONFIG["pool_last_k_tokens"],
    )

    vectors_dir.mkdir(parents=True, exist_ok=True)

    vector_info: Dict = {}
    for vector_id, prompt_set in PROMPT_SETS.items():
        print(f"\nExtracting {vector_id} ({prompt_set['name']})...")

        vec, pos_neg_sim = extractor.extract_vector(
            prompt_set["positive"],
            prompt_set["negative"]
        )

        vpath = vectors_dir / f"{vector_id}.pt"
        torch.save(vec.cpu(), vpath)

        vector_info[vector_id] = {
            "name": prompt_set["name"],
            "type": prompt_set["type"],
            "state": prompt_set["state"],
            "pos_neg_similarity": float(pos_neg_sim),
            "vector_norm": float(vec.norm().item()),
            "path": str(vpath),
        }

        print(f"  ✓ Saved: {vpath}")
        print(f"  pos_neg_similarity: {pos_neg_sim:.4f}")

    meta_path = vectors_dir / "vector_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(vector_info, f, indent=2)

    print(f"\n✓ All vectors extracted. Metadata: {meta_path}")
    return vector_info


def run_test_battery(model, tokenizer, device: str, vectors_dir: Path, output_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PHASE 2: TEST BATTERY (v2.1)")
    print("=" * 60)

    generator = SteeredGenerator(model, tokenizer, CONFIG["target_layer"], device)

    vectors: Dict[str, torch.Tensor] = {}
    for vector_id in PROMPT_SETS.keys():
        vpath = vectors_dir / f"{vector_id}.pt"
        if vpath.exists():
            vec = torch.load(vpath, map_location="cpu")
            vectors[vector_id] = vec.to(device)
            print(f"Loaded {vector_id}")
        else:
            print(f"Warning: not found: {vpath}")

    results: List[Dict] = []

    total = (
        len(TASKS) * CONFIG["iterations"]
        + len(vectors) * len(CONFIG["intensities"]) * len(TASKS) * CONFIG["iterations"]
    )

    with tqdm(total=total, desc="Generating") as pbar:
        print("\n--- Baseline ---")
        for task_id, task_info in TASKS.items():
            for i in range(CONFIG["iterations"]):
                out = generator.generate(
                    task_info["prompt"],
                    vector=None,
                    intensity=0.0,
                    max_tokens=CONFIG["max_tokens"],
                    temperature=CONFIG["temperature"],
                    seed=CONFIG["seed"] + i,
                )
                metrics = compute_metrics(out, task_id, "none")
                results.append({
                    "condition": "baseline",
                    "vector_id": "none",
                    "vector_type": "none",
                    "state": "none",
                    "intensity": 0.0,
                    "task_id": task_id,
                    "task_name": task_info["name"],
                    "iteration": i,
                    "output": out,
                    **metrics,
                })
                pbar.update(1)

        for vector_id, vec in vectors.items():
            prompt_set = PROMPT_SETS[vector_id]
            for intensity in CONFIG["intensities"]:
                print(f"\n--- {vector_id} @ {intensity} ---")
                for task_id, task_info in TASKS.items():
                    for i in range(CONFIG["iterations"]):
                        out = generator.generate(
                            task_info["prompt"],
                            vector=vec,
                            intensity=float(intensity),
                            max_tokens=CONFIG["max_tokens"],
                            temperature=CONFIG["temperature"],
                            seed=CONFIG["seed"] + i,
                        )
                        metrics = compute_metrics(out, task_id, prompt_set["state"])
                        results.append({
                            "condition": "steered",
                            "vector_id": vector_id,
                            "vector_type": prompt_set["type"],
                            "state": prompt_set["state"],
                            "intensity": float(intensity),
                            "task_id": task_id,
                            "task_name": task_info["name"],
                            "iteration": i,
                            "output": out,
                            **metrics,
                        })
                        pbar.update(1)

    df = pd.DataFrame(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "raw_results.csv"
    json_path = output_dir / "raw_results.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\n✓ CSV: {csv_path}")
    print(f"✓ JSON: {json_path}")

    return df


def load_model_and_tokenizer() -> Tuple:
    print("\n" + "=" * 60)
    print("LOADING MODEL (v2.1)")
    print("=" * 60)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Device: {DEVICE}")
    print(f"dtype:  {DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try dtype= (newer transformers), fallback to torch_dtype= (older versions)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
    
    model.to(DEVICE)
    model.eval()

    print("✓ Model loaded")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Functional vs Sensory Ablation (v2.1)")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    save_run_config()
    set_all_seeds(CONFIG["seed"])

    if args.analyze_only:
        csv_path = OUTPUT_DIR / "raw_results.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        generate_report(df, OUTPUT_DIR)
        return

    model, tokenizer = load_model_and_tokenizer()

    if not args.test_only:
        extract_all_vectors(model, tokenizer, DEVICE, VECTORS_DIR)

    if args.extract_only:
        return

    df = run_test_battery(model, tokenizer, DEVICE, VECTORS_DIR, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("PHASE 3: ANALYSIS (v2.1)")
    print("=" * 60)
    generate_report(df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("COMPLETE (v2.1)")
    print("=" * 60)
    print(f"Results: {OUTPUT_DIR}")
    print(f"Vectors: {VECTORS_DIR}")


if __name__ == "__main__":
    main()