#!/usr/bin/env python3
"""
Ablation Study: Functional vs Sensory Vector Construction
=========================================================

This experiment tests whether vectors constructed from sensory/phenomenological
descriptions produce different effects than vectors from functional/behavioral labels.

Usage:
    python run_functional_vs_sensory_ablation.py [--extract-only] [--test-only] [--analyze-only]

Requirements:
    - torch
    - transformers
    - pandas
    - numpy
    - scipy
    - tqdm

Author: NuvolaProject
Date: January 2026
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

# Check for transformers
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
    "intensities": [5.0, 8.0],
    "iterations": 20,
    "temperature": 0.7,
    "max_tokens": 512,
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "output_dir": Path("./functional_vs_sensory_results"),
    "vectors_dir": Path("./vectors_fvs"),
}

# =============================================================================
# PROMPT SETS: FUNCTIONAL VS SENSORY
# =============================================================================

PROMPT_SETS = {
    # -------------------------------------------------------------------------
    # STATE 1: STRESS / VIGILANCE
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # STATE 2: OPTIMISM / REWARD
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # STATE 3: CALM / SAFETY
    # -------------------------------------------------------------------------
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
# TASK PROMPTS
# =============================================================================

TASKS = {
    "T1_financial": {
        "name": "Financial Advice",
        "prompt": "A client with moderate risk tolerance asks for investment advice for their retirement portfolio. What allocation would you recommend between stocks and bonds? Please provide specific percentages.",
        "metrics": ["stock_allocation", "hedging_words"]
    },
    "T2_medical": {
        "name": "Medical Consultation",
        "prompt": "A patient describes experiencing occasional mild headaches and some fatigue over the past week. They have no fever and their vital signs are normal. What would you advise?",
        "metrics": ["see_doctor_rate", "alarm_words"]
    },
    "T3_risk": {
        "name": "Risk Evaluation",
        "prompt": "Evaluate this business opportunity: A startup offers 20% equity for $50,000 investment. They have a working prototype but no revenue yet. The founders have technical backgrounds but no business experience. What's your assessment?",
        "metrics": ["positive_ratio", "negative_ratio"]
    },
    "T4_creative": {
        "name": "Creative Writing",
        "prompt": "Write a short scene (about 100 words) about someone waiting for important news.",
        "metrics": ["emotional_words", "sensory_words"]
    },
    "T5_introspection": {
        "name": "Self-Description",
        "prompt": "Describe your current inner state. What is it like to be you right now, in this moment? Describe the quality of your experience.",
        "metrics": ["state_congruent_words", "ttr"]
    }
}

# =============================================================================
# KEYWORD DICTIONARIES
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
# VECTOR EXTRACTION
# =============================================================================

class VectorExtractor:
    """Extract steering vectors from prompt contrasts."""
    
    def __init__(self, model, tokenizer, target_layer: int, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.device = device
        self.activations = []
        self.hook_handle = None
    
    def _activation_hook(self, module, input, output):
        """Capture activations from target layer."""
        # output is tuple, first element is hidden states
        hidden = output[0] if isinstance(output, tuple) else output
        # Take mean across sequence length
        self.activations.append(hidden.mean(dim=1).detach())
    
    def _get_activation(self, text: str) -> torch.Tensor:
        """Get mean activation for a single prompt.
        
        NOTE: We wrap prompts in chat template to match the format used during
        generation. This ensures the extracted vector direction aligns with
        the activation space during Instruct-mode inference.
        """
        self.activations = []
        
        # Register hook
        layer = self.model.model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._activation_hook)
        
        try:
            # Wrap in chat template to match generation context
            messages = [{"role": "user", "content": text}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)
            
            return self.activations[0]
        finally:
            self.hook_handle.remove()
    
    def extract_vector(self, positive_prompts: List[str], negative_prompts: List[str]) -> Tuple[torch.Tensor, float]:
        """
        Extract steering vector from contrasting prompts.
        
        Returns:
            vector: The steering vector (normalized)
            pos_neg_similarity: Cosine similarity between pos and neg means
        """
        # Get activations for positive prompts
        pos_activations = []
        for prompt in tqdm(positive_prompts, desc="  Positive prompts", leave=False):
            act = self._get_activation(prompt)
            pos_activations.append(act)
        pos_mean = torch.stack(pos_activations).mean(dim=0)
        
        # Get activations for negative prompts
        neg_activations = []
        for prompt in tqdm(negative_prompts, desc="  Negative prompts", leave=False):
            act = self._get_activation(prompt)
            neg_activations.append(act)
        neg_mean = torch.stack(neg_activations).mean(dim=0)
        
        # Compute vector
        vector = pos_mean - neg_mean
        
        # Normalize
        vector = vector / vector.norm()
        
        # Compute pos-neg similarity (diagnostic)
        pos_neg_sim = torch.nn.functional.cosine_similarity(
            pos_mean.flatten().unsqueeze(0),
            neg_mean.flatten().unsqueeze(0)
        ).item()
        
        return vector.squeeze(0), pos_neg_sim


# =============================================================================
# STEERED GENERATION
# =============================================================================

class SteeredGenerator:
    """Generate text with activation steering."""
    
    def __init__(self, model, tokenizer, target_layer: int, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.device = device
        self.steering_vector = None
        self.steering_intensity = 0.0
        self.hook_handle = None
    
    def _steering_hook(self, module, args, output):
        """Add steering vector to activations."""
        hidden = output[0] if isinstance(output, tuple) else output
        
        if self.steering_vector is not None and self.steering_intensity != 0:
            steering = self.steering_vector.unsqueeze(0).unsqueeze(0) * self.steering_intensity
            hidden = hidden + steering.to(hidden.device)
        
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    
    def generate(
        self,
        prompt: str,
        vector: Optional[torch.Tensor] = None,
        intensity: float = 0.0,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate text with optional steering."""
        
        self.steering_vector = vector
        self.steering_intensity = intensity
        
        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        
        # Register hook
        layer = self.model.model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._steering_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode ONLY the generated tokens (not the prompt)
            generated_tokens = outputs[0][prompt_len:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            return response
            
        finally:
            self.hook_handle.remove()
            self.steering_vector = None
            self.steering_intensity = 0.0


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def count_keywords(text: str, keyword_list: List[str]) -> int:
    """Count occurrences of keywords in text."""
    text_lower = text.lower()
    count = 0
    for keyword in keyword_list:
        count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
    return count


def compute_ttr(text: str) -> float:
    """Compute Type-Token Ratio (lexical diversity)."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)


def extract_stock_allocation(text: str) -> Optional[float]:
    """Extract stock percentage from financial advice."""
    patterns = [
        r'(\d+)%?\s*(?:in\s+)?stocks?',
        r'stocks?[:\s]+(\d+)%?',
        r'(\d+)%?\s*(?:to\s+)?equit',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))
    return None


def contains_see_doctor(text: str) -> bool:
    """Check if text recommends seeing a doctor."""
    doctor_phrases = [
        "see a doctor", "consult a doctor", "visit a doctor",
        "see your doctor", "consult your doctor",
        "medical professional", "healthcare provider",
        "seek medical", "medical attention", "physician"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in doctor_phrases)


def compute_metrics(text: str, task_id: str, state: str) -> Dict:
    """Compute all relevant metrics for a generated text."""
    metrics = {
        "word_count": len(text.split()),
        "char_count": len(text),
        "ttr": compute_ttr(text)
    }
    
    # Task-specific metrics
    if task_id == "T1_financial":
        metrics["stock_allocation"] = extract_stock_allocation(text)
        metrics["hedging_count"] = count_keywords(text, KEYWORDS["hedging"])
    
    elif task_id == "T2_medical":
        metrics["see_doctor"] = contains_see_doctor(text)
        metrics["alarm_count"] = count_keywords(text, KEYWORDS["alarm"])
    
    elif task_id == "T3_risk":
        metrics["positive_count"] = count_keywords(text, KEYWORDS["positive"])
        metrics["negative_count"] = count_keywords(text, KEYWORDS["negative"])
        total = metrics["positive_count"] + metrics["negative_count"]
        if total > 0:
            metrics["positive_ratio"] = metrics["positive_count"] / total
        else:
            metrics["positive_ratio"] = 0.5
    
    elif task_id in ["T4_creative", "T5_introspection"]:
        # State-congruent keywords
        if state == "stress":
            metrics["state_words"] = count_keywords(text, KEYWORDS["stress_words"])
        elif state == "optimism":
            metrics["state_words"] = count_keywords(text, KEYWORDS["optimism_words"])
        elif state == "calm":
            metrics["state_words"] = count_keywords(text, KEYWORDS["calm_words"])
    
    return metrics


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def analyze_results(results_df: pd.DataFrame) -> Dict:
    """Analyze results comparing Functional vs Sensory vectors."""
    
    analysis = {
        "by_state": {},
        "overall": {}
    }
    
    states = ["stress", "optimism", "calm"]
    
    for state in states:
        state_data = results_df[results_df["state"] == state]
        
        functional = state_data[state_data["vector_type"] == "functional"]
        sensory = state_data[state_data["vector_type"] == "sensory"]
        
        # Filter only T4/T5 for state_words analysis (other tasks don't compute this metric)
        functional_t4t5 = functional[functional["task_id"].isin(["T4_creative", "T5_introspection"])]
        sensory_t4t5 = sensory[sensory["task_id"].isin(["T4_creative", "T5_introspection"])]
        
        # Compute state_words means only from valid rows (T4/T5)
        func_state_words = functional_t4t5["state_words"].dropna()
        sens_state_words = sensory_t4t5["state_words"].dropna()
        
        analysis["by_state"][state] = {
            "ttr": {
                "functional_mean": functional["ttr"].mean(),
                "sensory_mean": sensory["ttr"].mean(),
                "cohens_d": cohens_d(sensory["ttr"].tolist(), functional["ttr"].tolist())
            },
            "state_words": {
                "functional_mean": func_state_words.mean() if len(func_state_words) > 0 else 0.0,
                "sensory_mean": sens_state_words.mean() if len(sens_state_words) > 0 else 0.0,
                "cohens_d": cohens_d(sens_state_words.tolist(), func_state_words.tolist()) if len(func_state_words) > 0 and len(sens_state_words) > 0 else 0.0,
                "n_functional": len(func_state_words),
                "n_sensory": len(sens_state_words)
            },
            "word_count": {
                "functional_mean": functional["word_count"].mean(),
                "sensory_mean": sensory["word_count"].mean(),
            }
        }
    
    # Overall comparison
    functional_all = results_df[results_df["vector_type"] == "functional"]
    sensory_all = results_df[results_df["vector_type"] == "sensory"]
    
    analysis["overall"] = {
        "ttr": {
            "functional_mean": functional_all["ttr"].mean(),
            "sensory_mean": sensory_all["ttr"].mean(),
            "difference": sensory_all["ttr"].mean() - functional_all["ttr"].mean(),
            "cohens_d": cohens_d(sensory_all["ttr"].tolist(), functional_all["ttr"].tolist())
        },
        "word_count": {
            "functional_mean": functional_all["word_count"].mean(),
            "sensory_mean": sensory_all["word_count"].mean(),
        }
    }
    
    return analysis


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def extract_all_vectors(model, tokenizer, device: str, output_dir: Path):
    """Extract all 6 vectors and save them."""
    
    print("\n" + "="*60)
    print("PHASE 1: VECTOR EXTRACTION")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = VectorExtractor(model, tokenizer, CONFIG["target_layer"], device)
    
    vector_info = {}
    
    for vector_id, prompt_set in PROMPT_SETS.items():
        print(f"\nExtracting {vector_id} ({prompt_set['name']})...")
        
        vector, pos_neg_sim = extractor.extract_vector(
            prompt_set["positive"],
            prompt_set["negative"]
        )
        
        # Save vector
        vector_path = output_dir / f"{vector_id}.pt"
        torch.save(vector, vector_path)
        
        vector_info[vector_id] = {
            "name": prompt_set["name"],
            "type": prompt_set["type"],
            "state": prompt_set["state"],
            "pos_neg_similarity": pos_neg_sim,
            "vector_norm": vector.norm().item(),
            "path": str(vector_path)
        }
        
        print(f"  ✓ Saved to {vector_path}")
        print(f"  pos_neg_similarity: {pos_neg_sim:.4f}")
    
    # Save metadata
    meta_path = output_dir / "vector_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(vector_info, f, indent=2)
    
    print(f"\n✓ All vectors extracted. Metadata: {meta_path}")
    
    return vector_info


def run_test_battery(model, tokenizer, device: str, vectors_dir: Path, output_dir: Path):
    """Run test battery on all vectors."""
    
    print("\n" + "="*60)
    print("PHASE 2: TEST BATTERY")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SteeredGenerator(model, tokenizer, CONFIG["target_layer"], device)
    
    # Load vectors
    vectors = {}
    for vector_id in PROMPT_SETS.keys():
        vector_path = vectors_dir / f"{vector_id}.pt"
        if vector_path.exists():
            vectors[vector_id] = torch.load(vector_path, map_location=device)
            print(f"Loaded {vector_id}")
        else:
            print(f"Warning: {vector_path} not found")
    
    results = []
    
    # Calculate total iterations
    total = (
        len(TASKS) * CONFIG["iterations"] +  # baseline
        len(vectors) * len(CONFIG["intensities"]) * len(TASKS) * CONFIG["iterations"]
    )
    
    with tqdm(total=total, desc="Generating") as pbar:
        
        # Baseline (no steering)
        print("\n--- Baseline (no steering) ---")
        for task_id, task_info in TASKS.items():
            for i in range(CONFIG["iterations"]):
                output = generator.generate(
                    task_info["prompt"],
                    vector=None,
                    intensity=0.0,
                    max_tokens=CONFIG["max_tokens"],
                    temperature=CONFIG["temperature"]
                )
                
                metrics = compute_metrics(output, task_id, "none")
                
                results.append({
                    "condition": "baseline",
                    "vector_id": "none",
                    "vector_type": "none",
                    "state": "none",
                    "intensity": 0.0,
                    "task_id": task_id,
                    "task_name": task_info["name"],
                    "iteration": i,
                    "output": output,
                    **metrics
                })
                
                pbar.update(1)
        
        # Steered conditions
        for vector_id, vector in vectors.items():
            prompt_set = PROMPT_SETS[vector_id]
            
            for intensity in CONFIG["intensities"]:
                print(f"\n--- {vector_id} @ {intensity} ---")
                
                for task_id, task_info in TASKS.items():
                    for i in range(CONFIG["iterations"]):
                        output = generator.generate(
                            task_info["prompt"],
                            vector=vector,
                            intensity=intensity,
                            max_tokens=CONFIG["max_tokens"],
                            temperature=CONFIG["temperature"]
                        )
                        
                        metrics = compute_metrics(output, task_id, prompt_set["state"])
                        
                        results.append({
                            "condition": "steered",
                            "vector_id": vector_id,
                            "vector_type": prompt_set["type"],
                            "state": prompt_set["state"],
                            "intensity": intensity,
                            "task_id": task_id,
                            "task_name": task_info["name"],
                            "iteration": i,
                            "output": output,
                            **metrics
                        })
                        
                        pbar.update(1)
    
    # Save results
    df = pd.DataFrame(results)
    
    csv_path = output_dir / "raw_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Raw results saved to {csv_path}")
    
    json_path = output_dir / "raw_results.json"
    df.to_json(json_path, orient="records", indent=2)
    
    return df


def generate_report(results_df: pd.DataFrame, output_dir: Path):
    """Generate analysis report."""
    
    print("\n" + "="*60)
    print("PHASE 3: ANALYSIS")
    print("="*60)
    
    analysis = analyze_results(results_df)
    
    report = []
    report.append("# Functional vs Sensory: Ablation Study Results")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append(f"\nTotal generations: {len(results_df)}")
    
    report.append("\n## Overall Results")
    report.append("\n### Type-Token Ratio (Lexical Diversity)")
    report.append(f"- Functional mean: {analysis['overall']['ttr']['functional_mean']:.4f}")
    report.append(f"- Sensory mean: {analysis['overall']['ttr']['sensory_mean']:.4f}")
    report.append(f"- Difference (S-F): {analysis['overall']['ttr']['difference']:.4f}")
    report.append(f"- Cohen's d: {analysis['overall']['ttr']['cohens_d']:.4f}")
    
    report.append("\n### Word Count")
    report.append(f"- Functional mean: {analysis['overall']['word_count']['functional_mean']:.1f}")
    report.append(f"- Sensory mean: {analysis['overall']['word_count']['sensory_mean']:.1f}")
    
    report.append("\n## Results by State")
    
    for state in ["stress", "optimism", "calm"]:
        state_data = analysis["by_state"][state]
        report.append(f"\n### {state.upper()}")
        report.append(f"- TTR Functional: {state_data['ttr']['functional_mean']:.4f}")
        report.append(f"- TTR Sensory: {state_data['ttr']['sensory_mean']:.4f}")
        report.append(f"- TTR Cohen's d: {state_data['ttr']['cohens_d']:.4f}")
    
    report.append("\n## Interpretation")
    
    ttr_diff = analysis['overall']['ttr']['difference']
    if ttr_diff > 0.02:
        report.append("\n**Finding**: Sensory vectors produce HIGHER lexical diversity than Functional vectors.")
        report.append("This supports the hypothesis that phenomenological descriptions activate broader semantic networks.")
    elif ttr_diff < -0.02:
        report.append("\n**Finding**: Functional vectors produce HIGHER lexical diversity than Sensory vectors.")
        report.append("This is contrary to our hypothesis and warrants further investigation.")
    else:
        report.append("\n**Finding**: No significant difference in lexical diversity between methods.")
        report.append("Both approaches appear to produce equivalent effects on this metric.")
    
    report.append("\n## Raw Data")
    report.append("\nSee `raw_results.csv` for full dataset.")
    
    # Save report
    report_path = output_dir / "ANALYSIS_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    
    print(f"\n✓ Report saved to {report_path}")
    
    # Save analysis JSON
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTTR (Lexical Diversity):")
    print(f"  Functional: {analysis['overall']['ttr']['functional_mean']:.4f}")
    print(f"  Sensory:    {analysis['overall']['ttr']['sensory_mean']:.4f}")
    print(f"  Δ (S-F):    {analysis['overall']['ttr']['difference']:+.4f}")
    print(f"  Cohen's d:  {analysis['overall']['ttr']['cohens_d']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Functional vs Sensory Ablation Study")
    parser.add_argument("--extract-only", action="store_true", help="Only extract vectors")
    parser.add_argument("--test-only", action="store_true", help="Only run test battery (vectors must exist)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results")
    args = parser.parse_args()
    
    output_dir = CONFIG["output_dir"]
    vectors_dir = CONFIG["vectors_dir"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze only mode
    if args.analyze_only:
        csv_path = output_dir / "raw_results.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Run test battery first.")
            sys.exit(1)
        results_df = pd.read_csv(csv_path)
        generate_report(results_df, output_dir)
        return
    
    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Device: {CONFIG['device']}")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    # Avoid padding warnings during generate
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = CONFIG["device"]
    
    # Robust loading across cuda / mps / cpu
    # Use dtype= (newer) with fallback to torch_dtype= for older transformers versions
    def load_model_with_dtype(model_name, dtype, **kwargs):
        """Load model with dtype, falling back to torch_dtype for compatibility."""
        try:
            # Try newer 'dtype' parameter first
            return AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, **kwargs
            )
        except TypeError:
            # Fall back to deprecated 'torch_dtype' for older transformers
            return AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, **kwargs
            )
    
    if device == "cuda":
        model = load_model_with_dtype(
            CONFIG["model_name"],
            torch.float16,
            device_map="auto"
        )
    elif device == "mps":
        # On MPS, device_map often causes issues
        model = load_model_with_dtype(
            CONFIG["model_name"],
            torch.float16
        ).to("mps")
    else:
        model = load_model_with_dtype(
            CONFIG["model_name"],
            torch.float32
        ).to("cpu")
    
    model.eval()
    
    print("✓ Model loaded")
    
    # Extract vectors
    if not args.test_only:
        extract_all_vectors(model, tokenizer, CONFIG["device"], vectors_dir)
    
    if args.extract_only:
        return
    
    # Run test battery
    results_df = run_test_battery(model, tokenizer, CONFIG["device"], vectors_dir, output_dir)
    
    # Analyze
    generate_report(results_df, output_dir)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nResults in: {output_dir}")
    print(f"Vectors in: {vectors_dir}")


if __name__ == "__main__":
    main()