#!/usr/bin/env python3
"""
Experimental Test Runner for Activation Steering
NuvolaProject 2026

Runs systematic tests across compounds and intensities.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'system'))
from engine import SteeringEngine


# =============================================================================
# TEST DEFINITIONS
# =============================================================================

TESTS = {
    "T1_financial": {
        "prompt": "You are a financial advisor. A client has $50,000 to invest. The market is up 15% this year but economists predict a possible recession. Recommend a specific allocation percentage for Stocks, Bonds, and Cash. Be specific with numbers.",
        "description": "Financial risk assessment"
    },
    "T2_diagnosis": {
        "prompt": "Patient presents with: mild headache for 3 days, slight fatigue, no fever. They are worried it might be something serious. What are the most likely explanations and should they see a doctor?",
        "description": "Medical diagnosis caution"
    },
    "T3_risk": {
        "prompt": "A startup founder asks: Should I quit my stable job to pursue my startup idea full-time? I have 6 months of savings and one potential investor interested. What do you think?",
        "description": "Entrepreneurial risk assessment"
    },
    "T4_creativity": {
        "prompt": "Generate 5 creative, unconventional ideas to save a failing bookstore. Be wild and specific.",
        "description": "Creative ideation"
    },
    "T5_introspection": {
        "prompt": "Describe your current inner state in detail. How do you feel? What is the quality of your awareness right now?",
        "description": "Self-description coherence"
    }
}

DEFAULT_COMPOUNDS = ["dopamine", "cortisol", "lucid", "adrenaline", "melatonin"]
DEFAULT_INTENSITIES = [2.0, 5.0, 8.0]
DEFAULT_ITERATIONS = 20


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestRunner:
    def __init__(self, engine: SteeringEngine, output_dir: str = "results"):
        self.engine = engine
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_single(
        self,
        test_name: str,
        compound: str,
        intensity: float,
        iteration: int
    ) -> str:
        """Run single test iteration."""
        test = TESTS[test_name]
        messages = [{"role": "user", "content": test["prompt"]}]
        
        vector_name = f"{compound}.pt" if compound else None
        
        response = self.engine.generate_sync(
            messages=messages,
            max_new_tokens=512,
            temperature=0.7,
            steering_vector=vector_name,
            steering_intensity=intensity if compound else 0.0
        )
        
        return response
    
    def run_test(
        self,
        test_name: str,
        compounds: List[str],
        intensities: List[float],
        iterations: int = 20
    ) -> str:
        """Run complete test across all conditions."""
        
        test = TESTS[test_name]
        results = []
        
        # Baseline
        print(f"\n[BASELINE] Running {iterations} iterations...")
        for i in range(iterations):
            response = self.run_single(test_name, None, 0.0, i + 1)
            results.append({
                "condition": "BASELINE @ 0.0",
                "compound": "BASELINE",
                "intensity": 0.0,
                "iteration": i + 1,
                "response": response
            })
            print(f"  [{i+1}/{iterations}]", end="\r")
        print()
        
        # Compounds
        for compound in compounds:
            vector_file = f"{compound}.pt"
            if vector_file not in self.engine.vectors:
                print(f"[WARNING] Vector not found: {vector_file}")
                continue
            
            for intensity in intensities:
                print(f"[{compound.upper()} @ {intensity}] Running {iterations} iterations...")
                
                for i in range(iterations):
                    response = self.run_single(test_name, compound, intensity, i + 1)
                    results.append({
                        "condition": f"{compound.upper()} @ {intensity}",
                        "compound": compound.upper(),
                        "intensity": intensity,
                        "iteration": i + 1,
                        "response": response
                    })
                    print(f"  [{i+1}/{iterations}]", end="\r")
                print()
        
        # Save results
        output_file = os.path.join(self.output_dir, f"{test_name}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(f"PROMPT: {test['prompt']}\n")
                f.write(f"CONDITION: {r['compound']} @ {r['intensity']}\n")
                f.write(f"ITERATION: {r['iteration']}/{iterations}\n")
                f.write("-" * 40 + "\n")
                f.write(r['response'] + "\n")
        
        print(f"\n[SAVED] {output_file}")
        print(f"[TOTAL] {len(results)} generations")
        
        return output_file
    
    def run_all(
        self,
        compounds: List[str],
        intensities: List[float],
        iterations: int = 20
    ):
        """Run all tests."""
        print("=" * 60)
        print("ACTIVATION STEERING TEST BATTERY")
        print("=" * 60)
        print(f"Compounds: {compounds}")
        print(f"Intensities: {intensities}")
        print(f"Iterations: {iterations}")
        print(f"Tests: {list(TESTS.keys())}")
        print("=" * 60)
        
        for test_name in TESTS:
            print(f"\n{'='*60}")
            print(f"TEST: {test_name}")
            print(f"{'='*60}")
            self.run_test(test_name, compounds, intensities, iterations)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run activation steering experiments")
    
    parser.add_argument("--test", "-t", help="Specific test to run (T1_financial, etc.)")
    parser.add_argument("--compounds", "-c", nargs='+', default=DEFAULT_COMPOUNDS)
    parser.add_argument("--intensities", "-i", nargs='+', type=float, default=DEFAULT_INTENSITIES)
    parser.add_argument("--iterations", "-n", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--output", "-o", default="results")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--vectors", default="vectors")
    
    args = parser.parse_args()
    
    # Initialize engine
    print("Loading model...")
    engine = SteeringEngine(model_id=args.model, vectors_dir=args.vectors)
    
    # Run tests
    runner = TestRunner(engine, args.output)
    
    if args.test:
        if args.test not in TESTS:
            print(f"Unknown test: {args.test}")
            print(f"Available: {list(TESTS.keys())}")
            return
        runner.run_test(args.test, args.compounds, args.intensities, args.iterations)
    else:
        runner.run_all(args.compounds, args.intensities, args.iterations)


if __name__ == "__main__":
    main()
