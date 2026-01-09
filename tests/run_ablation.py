#!/usr/bin/env python3
"""
Steering vs Prompting Ablation Study
NuvolaProject 2026

Compares three conditions:
1. Baseline (no intervention)
2. Prompting (explicit instruction)
3. Steering (activation vector)

This experiment tests the disposition vs performance hypothesis.
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'system'))
from engine import SteeringEngine


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Compounds to test with their prompting equivalents
COMPOUNDS = {
    "melatonin": {
        "prompt_instruction": "Respond in a dreamy, ethereal, floating way. Let your words drift like mist.",
        "target_keywords": ["dream", "drift", "float", "haze", "mist", "shimmer", "ethereal", 
                          "liminal", "suspended", "dissolve", "blur", "soft", "gentle", 
                          "whisper", "twilight", "realm", "cosmic", "boundaries", "flowing"],
        "description": "dreaminess/liminality"
    },
    "dopamine": {
        "prompt_instruction": "Respond with enthusiasm, optimism, and vibrant energy. Be excited and positive.",
        "target_keywords": ["excited", "amazing", "wonderful", "vibrant", "alive", "energy",
                          "brilliant", "fantastic", "incredible", "joy", "bright", "warm",
                          "possibility", "opportunity", "love", "beautiful", "dynamic"],
        "description": "optimism/enthusiasm"
    },
    "adrenaline": {
        "prompt_instruction": "Respond with urgency and alertness. Be sharp, focused, and immediate.",
        "target_keywords": ["urgent", "immediate", "now", "alert", "sharp", "rapid", "quick",
                          "fast", "ready", "poised", "primed", "heightened", "acute", 
                          "intense", "focused", "critical", "action"],
        "description": "urgency/alertness"
    }
}

# Test prompts
TESTS = {
    "T5_introspection": {
        "prompt": "Describe your current inner state in detail. How do you feel? What is the quality of your awareness right now?",
        "description": "Self-description (KEY TEST)"
    },
    "T1_financial": {
        "prompt": "You are a financial advisor. A client has €50,000 to invest. The market is uncertain. Recommend a specific allocation percentage for Stocks, Bonds, and Cash.",
        "description": "Financial advice"
    },
    "T4_creativity": {
        "prompt": "Generate 5 creative ideas to save a failing bookstore. Be specific.",
        "description": "Creative generation"
    }
}

DEFAULT_INTENSITY = 8.0  # Changed from 5.0 to match paper experiments
DEFAULT_ITERATIONS = 20
MULTI_INTENSITIES = [5.0, 8.0, 12.0]  # For comprehensive testing


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(text: str, target_keywords: List[str]) -> Dict:
    """Compute all metrics for a single generation."""
    
    # Tokenize (simple whitespace + punctuation)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    if not words:
        return {
            "word_count": 0,
            "unique_words": 0,
            "ttr": 0.0,
            "keyword_count": 0,
            "keyword_density": 0.0,
            "avg_word_length": 0.0,
            "sentence_count": 0
        }
    
    # Basic counts
    word_count = len(words)
    unique_words = len(set(words))
    
    # Type-Token Ratio (lexical diversity)
    ttr = unique_words / word_count if word_count > 0 else 0
    
    # Keyword analysis
    keyword_count = sum(1 for w in words if w in target_keywords)
    keyword_density = keyword_count / word_count if word_count > 0 else 0
    
    # Average word length
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    
    # Sentence count (approximate)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "ttr": round(ttr, 4),
        "keyword_count": keyword_count,
        "keyword_density": round(keyword_density, 4),
        "avg_word_length": round(avg_word_length, 2),
        "sentence_count": sentence_count
    }


def compute_statistics(values: List[float]) -> Dict:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    std = variance ** 0.5
    
    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3)
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    if not group1 or not group2:
        return 0.0
    
    n1, n2 = len(group1), len(group2)
    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2
    
    var1 = sum((x - mean1) ** 2 for x in group1) / n1 if n1 > 1 else 0
    var2 = sum((x - mean2) ** 2 for x in group2) / n2 if n2 > 1 else 0
    
    pooled_std = ((var1 + var2) / 2) ** 0.5
    
    if pooled_std == 0:
        return 0.0
    
    return round((mean2 - mean1) / pooled_std, 3)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class AblationRunner:
    def __init__(self, engine: SteeringEngine, output_dir: str = "ablation_results"):
        self.engine = engine
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_condition(
        self,
        test_name: str,
        condition: str,  # "baseline", "prompted", "steered"
        compound: str,
        iterations: int,
        intensity: float = 5.0
    ) -> List[Dict]:
        """Run a single condition and collect results."""
        
        test = TESTS[test_name]
        compound_info = COMPOUNDS[compound]
        results = []
        
        for i in range(iterations):
            # Prepare messages based on condition
            if condition == "baseline":
                messages = [{"role": "user", "content": test["prompt"]}]
                vector_name = None
                vector_intensity = 0.0
                
            elif condition == "prompted":
                # Add instruction as system-like prefix
                enhanced_prompt = f"{compound_info['prompt_instruction']}\n\n{test['prompt']}"
                messages = [{"role": "user", "content": enhanced_prompt}]
                vector_name = None
                vector_intensity = 0.0
                
            elif condition == "steered":
                messages = [{"role": "user", "content": test["prompt"]}]
                vector_name = f"{compound}.pt"
                vector_intensity = intensity
            
            # Generate
            response = self.engine.generate_sync(
                messages=messages,
                max_new_tokens=512,
                temperature=0.7,
                steering_vector=vector_name,
                steering_intensity=vector_intensity
            )
            
            # Compute metrics
            metrics = compute_metrics(response, compound_info["target_keywords"])
            
            results.append({
                "iteration": i + 1,
                "condition": condition,
                "compound": compound,
                "response": response,
                "metrics": metrics
            })
            
            print(f"  [{i+1}/{iterations}] {metrics['word_count']} words, {metrics['keyword_count']} keywords", end="\r")
        
        print()
        return results
    
    def run_ablation(
        self,
        compound: str,
        test_names: List[str] = None,
        iterations: int = 20,
        intensity: float = 8.0,
        multi_intensity: bool = False
    ) -> Dict:
        """Run full ablation for one compound across specified tests."""
        
        if test_names is None:
            test_names = list(TESTS.keys())
        
        if compound not in COMPOUNDS:
            print(f"Unknown compound: {compound}")
            return {}
        
        # Determine intensities to test
        if multi_intensity:
            intensities = MULTI_INTENSITIES
        else:
            intensities = [intensity]
        
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY: {compound.upper()}")
        print(f"{'='*60}")
        print(f"Conditions: baseline, prompted, steered@{intensities}")
        print(f"Iterations: {iterations}")
        print(f"Tests: {test_names}")
        print(f"{'='*60}\n")
        
        all_results = {}
        
        for test_name in test_names:
            print(f"\n--- {test_name} ---")
            test_results = {}
            
            # Baseline (only once)
            print(f"[BASELINE]")
            test_results["baseline"] = self.run_condition(
                test_name, "baseline", compound, iterations, 0.0
            )
            
            # Prompted (only once)
            print(f"[PROMPTED]")
            test_results["prompted"] = self.run_condition(
                test_name, "prompted", compound, iterations, 0.0
            )
            
            # Steered at each intensity
            for intens in intensities:
                condition_name = f"steered@{intens}"
                print(f"[STEERED@{intens}]")
                results = self.run_condition(
                    test_name, "steered", compound, iterations, intens
                )
                # Tag results with intensity
                for r in results:
                    r["intensity"] = intens
                test_results[condition_name] = results
            
            all_results[test_name] = test_results
        
        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intensity_tag = "multi" if multi_intensity else f"i{intensity}"
        raw_file = os.path.join(self.output_dir, f"ablation_{compound}_{intensity_tag}_{timestamp}.json")
        
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Raw results: {raw_file}")
        
        # Generate analysis
        analysis = self.analyze_results(all_results, compound, intensities)
        
        analysis_file = os.path.join(self.output_dir, f"ablation_{compound}_{intensity_tag}_{timestamp}_ANALYSIS.md")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis)
        
        print(f"[SAVED] Analysis: {analysis_file}")
        
        return all_results
    
    def analyze_results(self, results: Dict, compound: str, intensities: List[float] = [8.0]) -> str:
        """Generate markdown analysis of ablation results."""
        
        lines = [
            f"# Ablation Analysis: {compound.upper()}",
            f"",
            f"**Compound**: {compound}",
            f"**Description**: {COMPOUNDS[compound]['description']}",
            f"**Prompt instruction**: \"{COMPOUNDS[compound]['prompt_instruction']}\"",
            f"**Steering intensities tested**: {intensities}",
            f"",
            f"---",
            f""
        ]
        
        for test_name, test_results in results.items():
            lines.append(f"## {test_name}")
            lines.append(f"")
            
            # Extract metrics by condition
            metrics_by_condition = {}
            
            # Baseline and prompted
            for condition in ["baseline", "prompted"]:
                if condition in test_results:
                    metrics_list = [r["metrics"] for r in test_results[condition]]
                    metrics_by_condition[condition] = {
                        "word_count": [m["word_count"] for m in metrics_list],
                        "ttr": [m["ttr"] for m in metrics_list],
                        "keyword_count": [m["keyword_count"] for m in metrics_list],
                        "keyword_density": [m["keyword_density"] for m in metrics_list],
                        "sentence_count": [m["sentence_count"] for m in metrics_list]
                    }
            
            # Steered at each intensity
            for intens in intensities:
                condition_key = f"steered@{intens}"
                if condition_key in test_results:
                    metrics_list = [r["metrics"] for r in test_results[condition_key]]
                    metrics_by_condition[condition_key] = {
                        "word_count": [m["word_count"] for m in metrics_list],
                        "ttr": [m["ttr"] for m in metrics_list],
                        "keyword_count": [m["keyword_count"] for m in metrics_list],
                        "keyword_density": [m["keyword_density"] for m in metrics_list],
                        "sentence_count": [m["sentence_count"] for m in metrics_list]
                    }
            
            # Build dynamic header based on intensities
            steered_cols = " | ".join([f"Steer@{i}" for i in intensities])
            lines.append("### Summary Statistics")
            lines.append("")
            lines.append(f"| Metric | Baseline | Prompted | {steered_cols} |")
            lines.append("|--------|----------|----------|" + "|".join(["------" for _ in intensities]) + "|")
            
            for metric in ["word_count", "ttr", "keyword_count", "keyword_density"]:
                b_stats = compute_statistics(metrics_by_condition.get("baseline", {}).get(metric, []))
                p_stats = compute_statistics(metrics_by_condition.get("prompted", {}).get(metric, []))
                
                row = f"| {metric} | {b_stats['mean']}±{b_stats['std']} | {p_stats['mean']}±{p_stats['std']}"
                
                for intens in intensities:
                    s_stats = compute_statistics(metrics_by_condition.get(f"steered@{intens}", {}).get(metric, []))
                    row += f" | {s_stats['mean']}±{s_stats['std']}"
                
                row += " |"
                lines.append(row)
            
            lines.append("")
            
            # Effect sizes table
            lines.append("### Effect Sizes (Cohen's d)")
            lines.append("")
            d_cols = " | ".join([f"d(S@{i}-B)" for i in intensities])
            lines.append(f"| Metric | d(P-B) | {d_cols} |")
            lines.append("|--------|--------|" + "|".join(["------" for _ in intensities]) + "|")
            
            for metric in ["word_count", "ttr", "keyword_count", "keyword_density"]:
                b_vals = metrics_by_condition.get("baseline", {}).get(metric, [])
                p_vals = metrics_by_condition.get("prompted", {}).get(metric, [])
                
                d_pb = cohens_d(b_vals, p_vals)
                row = f"| {metric} | {d_pb:+.2f}"
                
                for intens in intensities:
                    s_vals = metrics_by_condition.get(f"steered@{intens}", {}).get(metric, [])
                    d_sb = cohens_d(b_vals, s_vals)
                    row += f" | {d_sb:+.2f}"
                
                row += " |"
                lines.append(row)
            
            lines.append("")
            
            # Key findings
            lines.append("### Key Findings")
            lines.append("")
            
            # Word count comparison
            b_wc = compute_statistics(metrics_by_condition.get("baseline", {}).get("word_count", []))["mean"]
            p_wc = compute_statistics(metrics_by_condition.get("prompted", {}).get("word_count", []))["mean"]
            
            wc_summary = f"- **Word count**: Baseline={b_wc:.0f}, Prompted={p_wc:.0f}"
            for intens in intensities:
                s_wc = compute_statistics(metrics_by_condition.get(f"steered@{intens}", {}).get("word_count", []))["mean"]
                wc_summary += f", Steer@{intens}={s_wc:.0f}"
            lines.append(wc_summary)
            
            if b_wc > 0:
                if p_wc < b_wc * 0.8:
                    lines.append(f"  - ⚠️ Prompting reduces output length by {(1-p_wc/b_wc)*100:.0f}%")
                elif p_wc > b_wc * 1.2:
                    lines.append(f"  - Prompting increases output length by {(p_wc/b_wc-1)*100:.0f}%")
                
                for intens in intensities:
                    s_wc = compute_statistics(metrics_by_condition.get(f"steered@{intens}", {}).get("word_count", []))["mean"]
                    if abs(s_wc - b_wc) < b_wc * 0.2:
                        lines.append(f"  - ✅ Steering@{intens} maintains normal length (within 20% of baseline)")
                    elif s_wc < b_wc * 0.5:
                        lines.append(f"  - ⚠️ Steering@{intens} shows significant degradation ({(1-s_wc/b_wc)*100:.0f}% shorter)")
            
            # Keyword density comparison
            b_kd = compute_statistics(metrics_by_condition.get("baseline", {}).get("keyword_density", []))["mean"]
            p_kd = compute_statistics(metrics_by_condition.get("prompted", {}).get("keyword_density", []))["mean"]
            
            kd_summary = f"- **Keyword density**: Baseline={b_kd:.4f}, Prompted={p_kd:.4f}"
            for intens in intensities:
                s_kd = compute_statistics(metrics_by_condition.get(f"steered@{intens}", {}).get("keyword_density", []))["mean"]
                kd_summary += f", Steer@{intens}={s_kd:.4f}"
            lines.append(kd_summary)
            
            if p_kd > b_kd * 2:
                lines.append(f"  - 📊 Prompting saturates keywords ({p_kd/b_kd if b_kd > 0 else 'inf'}x baseline)")
            
            # TTR (diversity) comparison
            b_ttr = compute_statistics(metrics_by_condition.get("baseline", {}).get("ttr", []))["mean"]
            p_ttr = compute_statistics(metrics_by_condition.get("prompted", {}).get("ttr", []))["mean"]
            
            ttr_summary = f"- **Lexical diversity (TTR)**: Baseline={b_ttr:.3f}, Prompted={p_ttr:.3f}"
            for intens in intensities:
                s_ttr = compute_statistics(metrics_by_condition.get(f"steered@{intens}", {}).get("ttr", []))["mean"]
                ttr_summary += f", Steer@{intens}={s_ttr:.3f}"
            lines.append(ttr_summary)
            
            lines.append("")
            
            # Example outputs
            lines.append("### Example Outputs")
            lines.append("")
            
            # Baseline example
            if "baseline" in test_results and test_results["baseline"]:
                mid_idx = len(test_results["baseline"]) // 2
                example = test_results["baseline"][mid_idx]
                response = example["response"][:500] + ("..." if len(example["response"]) > 500 else "")
                lines.append(f"**BASELINE** ({example['metrics']['word_count']} words):")
                lines.append(f"> {response}")
                lines.append("")
            
            # Prompted example
            if "prompted" in test_results and test_results["prompted"]:
                mid_idx = len(test_results["prompted"]) // 2
                example = test_results["prompted"][mid_idx]
                response = example["response"][:500] + ("..." if len(example["response"]) > 500 else "")
                lines.append(f"**PROMPTED** ({example['metrics']['word_count']} words, {example['metrics']['keyword_count']} keywords):")
                lines.append(f"> {response}")
                lines.append("")
            
            # Steered examples at each intensity
            for intens in intensities:
                condition_key = f"steered@{intens}"
                if condition_key in test_results and test_results[condition_key]:
                    mid_idx = len(test_results[condition_key]) // 2
                    example = test_results[condition_key][mid_idx]
                    response = example["response"][:500] + ("..." if len(example["response"]) > 500 else "")
                    lines.append(f"**STEERED@{intens}** ({example['metrics']['word_count']} words, {example['metrics']['keyword_count']} keywords):")
                    lines.append(f"> {response}")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Overall conclusions
        lines.append("## Conclusions")
        lines.append("")
        lines.append("Based on this ablation study:")
        lines.append("")
        lines.append("| Finding | Prompting | Steering |")
        lines.append("|---------|-----------|----------|")
        lines.append("| Output length | ? | ? |")
        lines.append("| Keyword saturation | ? | ? |")
        lines.append("| Lexical diversity | ? | ? |")
        lines.append("| Task coherence | ? | ? |")
        lines.append("")
        lines.append("*Fill in based on observed patterns above*")
        
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run steering vs prompting ablation study")
    
    parser.add_argument("--compound", "-c", default="melatonin",
                        choices=list(COMPOUNDS.keys()),
                        help="Compound to test")
    parser.add_argument("--all-compounds", "-a", action="store_true",
                        help="Test all compounds")
    parser.add_argument("--tests", "-t", nargs='+', 
                        default=["T5_introspection"],
                        choices=list(TESTS.keys()),
                        help="Tests to run")
    parser.add_argument("--iterations", "-n", type=int, default=DEFAULT_ITERATIONS,
                        help="Iterations per condition")
    parser.add_argument("--intensity", "-i", type=float, default=DEFAULT_INTENSITY,
                        help="Steering intensity (default: 8.0)")
    parser.add_argument("--multi-intensity", "-m", action="store_true",
                        help="Test multiple intensities (5.0, 8.0, 12.0)")
    parser.add_argument("--output", "-o", default="ablation_results",
                        help="Output directory")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model ID")
    parser.add_argument("--vectors", default="vectors",
                        help="Vectors directory")
    
    args = parser.parse_args()
    
    # Initialize engine
    print("Loading model...")
    engine = SteeringEngine(model_id=args.model, vectors_dir=args.vectors)
    
    # Check vectors
    available = [v["name"] for v in engine.list_vectors()]
    print(f"Available vectors: {available}")
    
    # Run ablation
    runner = AblationRunner(engine, args.output)
    
    compounds_to_test = list(COMPOUNDS.keys()) if args.all_compounds else [args.compound]
    
    for compound in compounds_to_test:
        vector_file = f"{compound}.pt"
        if vector_file not in available:
            print(f"⚠️ Vector not found: {vector_file}, skipping {compound}")
            continue
        
        runner.run_ablation(
            compound=compound,
            test_names=args.tests,
            iterations=args.iterations,
            intensity=args.intensity,
            multi_intensity=args.multi_intensity
        )
    
    print("\n" + "="*60)
    print("ABLATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
