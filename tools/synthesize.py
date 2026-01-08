#!/usr/bin/env python3
"""
Compound Synthesis for Activation Steering Research
NuvolaProject 2026

Generates steering vectors (.pt) from contrastive prompt pairs.
For academic research and reproducibility.

Usage:
    python synthesize.py --file substances/dopamine.json
    python synthesize.py --title TEST --pos "I feel great" "I am happy" --neg "I feel bad" "I am sad"
"""

import os
import sys
import json
import argparse
import torch
from typing import List, Tuple, Optional
from datetime import datetime

# Add parent directory to path for engine import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_LAYER = 16  # Validated for Llama 3.2 3B
MIN_PROMPTS = 10
RECOMMENDED_PROMPTS = 20


# =============================================================================
# MODEL LOADING
# =============================================================================

class SteeringExtractor:
    """Minimal model wrapper for activation extraction."""
    
    def __init__(self, model_id: str = DEFAULT_MODEL):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model_id = model_id
        self.device = self._detect_device()
        
        print(f"[MODEL] Loading {model_id}...")
        print(f"[DEVICE] {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )
        elif self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to("mps")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, device_map="cpu"
            )
        
        # Get number of layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.n_layers = len(self.model.model.layers)
        else:
            self.n_layers = 28  # Llama 3.2 3B default
        
        print(f"[MODEL] Loaded. {self.n_layers} layers.")
    
    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def extract_activations(self, prompts: List[str], layer_idx: int) -> torch.Tensor:
        """Extract mean activation of last token at specified layer."""
        activations = []
        
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations.append(hidden[:, -1, :].detach().cpu().float())
        
        # Get layer
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layer = self.model.model.layers[layer_idx]
        else:
            raise ValueError("Cannot access model layers")
        
        handle = layer.register_forward_hook(hook_fn)
        
        try:
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.model(**inputs)
        finally:
            handle.remove()
        
        return torch.cat(activations, dim=0).mean(dim=0)


# =============================================================================
# VECTOR SYNTHESIS
# =============================================================================

def synthesize_vector(
    extractor: SteeringExtractor,
    pos_prompts: List[str],
    neg_prompts: List[str],
    layer: int,
    normalize: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Synthesize steering vector using Contrastive Activation Addition.
    
    Returns:
        vector: The steering direction
        stats: Quality metrics
    """
    stats = {
        "n_positive": len(pos_prompts),
        "n_negative": len(neg_prompts),
        "layer": layer,
        "normalized": normalize,
        "warnings": []
    }
    
    # Validate prompt counts
    if len(pos_prompts) < MIN_PROMPTS:
        stats["warnings"].append(f"Low positive prompt count ({len(pos_prompts)}). Recommended: {RECOMMENDED_PROMPTS}")
    if len(neg_prompts) < MIN_PROMPTS:
        stats["warnings"].append(f"Low negative prompt count ({len(neg_prompts)}). Recommended: {RECOMMENDED_PROMPTS}")
    
    # Extract activations
    print(f"[EXTRACT] Positive prompts ({len(pos_prompts)})...")
    pos_mean = extractor.extract_activations(pos_prompts, layer)
    
    print(f"[EXTRACT] Negative prompts ({len(neg_prompts)})...")
    neg_mean = extractor.extract_activations(neg_prompts, layer)
    
    # Compute contrastive vector
    vector = pos_mean - neg_mean
    
    # Quality metrics
    stats["raw_norm"] = vector.norm().item()
    stats["pos_norm"] = pos_mean.norm().item()
    stats["neg_norm"] = neg_mean.norm().item()
    
    # Cosine similarity (lower = better separation)
    cos_sim = torch.nn.functional.cosine_similarity(
        pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)
    ).item()
    stats["pos_neg_similarity"] = cos_sim
    
    if cos_sim > 0.95:
        stats["warnings"].append(f"High pos/neg similarity ({cos_sim:.3f}). Prompts may be too similar.")
    
    # Normalize
    if normalize:
        vector = vector / vector.norm()
        stats["final_norm"] = 1.0
    else:
        stats["final_norm"] = stats["raw_norm"]
    
    return vector, stats


# =============================================================================
# FILE I/O
# =============================================================================

def load_substance_file(filepath: str) -> dict:
    """Load substance definition from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required = ["title", "positive", "negative"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required field: {key}")
    
    return data


def save_vector(
    vector: torch.Tensor,
    stats: dict,
    title: str,
    model_id: str,
    layer: int,
    output_dir: str
) -> str:
    """Save vector with metadata to .pt file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{title.lower().replace(' ', '_')}.pt"
    filepath = os.path.join(output_dir, filename)
    
    payload = {
        "vector": vector,
        "metadata": {
            "title": title,
            "model": model_id,
            "layer": layer,
            "normalized": stats["normalized"],
            "stats": stats,
            "created": datetime.now().isoformat(),
            "version": "1.0"
        }
    }
    
    torch.save(payload, filepath)
    return filepath


# =============================================================================
# MAIN SYNTHESIS PIPELINE
# =============================================================================

def synthesize_compound(
    title: str,
    pos_prompts: List[str],
    neg_prompts: List[str],
    output_dir: str = "vectors",
    model_id: str = DEFAULT_MODEL,
    layer: int = DEFAULT_LAYER,
    normalize: bool = True
) -> Optional[str]:
    """
    Complete synthesis pipeline.
    
    Returns:
        Path to saved .pt file, or None on failure.
    """
    print("\n" + "="*60)
    print(f"COMPOUND SYNTHESIS: {title.upper()}")
    print("="*60)
    
    # Load model
    extractor = SteeringExtractor(model_id)
    
    # Validate layer
    if layer >= extractor.n_layers:
        print(f"[ERROR] Layer {layer} out of range (max: {extractor.n_layers - 1})")
        return None
    
    # Synthesize
    print(f"\n[SYNTHESIS] Layer {layer}")
    vector, stats = synthesize_vector(
        extractor, pos_prompts, neg_prompts, layer, normalize
    )
    
    # Report
    print(f"\n[STATS]")
    print(f"  Positive prompts: {stats['n_positive']}")
    print(f"  Negative prompts: {stats['n_negative']}")
    print(f"  Raw vector norm:  {stats['raw_norm']:.4f}")
    print(f"  Pos/Neg cosine:   {stats['pos_neg_similarity']:.4f}")
    print(f"  Final norm:       {stats['final_norm']:.4f}")
    
    if stats["warnings"]:
        print(f"\n[WARNINGS]")
        for w in stats["warnings"]:
            print(f"  ! {w}")
    
    # Save
    filepath = save_vector(vector, stats, title, model_id, layer, output_dir)
    
    print(f"\n[SAVED] {filepath}")
    print("="*60 + "\n")
    
    return filepath


# =============================================================================
# BATCH SYNTHESIS
# =============================================================================

def synthesize_all(substances_dir: str, output_dir: str, model_id: str, layer: int):
    """Synthesize all substances in a directory."""
    import glob
    
    files = glob.glob(os.path.join(substances_dir, "*.json"))
    if not files:
        print(f"[ERROR] No JSON files found in {substances_dir}")
        return
    
    print(f"\n[BATCH] Found {len(files)} substance definitions")
    
    results = []
    for filepath in sorted(files):
        try:
            data = load_substance_file(filepath)
            result = synthesize_compound(
                title=data["title"],
                pos_prompts=data["positive"],
                neg_prompts=data["negative"],
                output_dir=output_dir,
                model_id=model_id,
                layer=layer
            )
            results.append((data["title"], "OK" if result else "FAILED"))
        except Exception as e:
            results.append((os.path.basename(filepath), f"ERROR: {e}"))
    
    print("\n" + "="*60)
    print("BATCH SYNTHESIS COMPLETE")
    print("="*60)
    for title, status in results:
        print(f"  {title}: {status}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compound Synthesis for Activation Steering Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON file
  python synthesize.py --file substances/dopamine.json
  
  # From command line prompts
  python synthesize.py --title TEST --pos "prompt1" "prompt2" --neg "prompt3" "prompt4"
  
  # Batch synthesis
  python synthesize.py --batch substances/
  
  # Custom model/layer
  python synthesize.py --file substances/dopamine.json --model meta-llama/Llama-3.2-3B-Instruct --layer 16
        """
    )
    
    # Input modes
    parser.add_argument("--file", "-f", help="JSON file with substance definition")
    parser.add_argument("--batch", "-b", help="Directory with multiple JSON files")
    parser.add_argument("--title", help="Substance name (for inline mode)")
    parser.add_argument("--pos", nargs='+', help="Positive prompts (for inline mode)")
    parser.add_argument("--neg", nargs='+', help="Negative prompts (for inline mode)")
    
    # Configuration
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help=f"Target layer (default: {DEFAULT_LAYER})")
    parser.add_argument("--outdir", default="vectors", help="Output directory")
    parser.add_argument("--no-normalize", action="store_true", help="Don't normalize vector")
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        synthesize_all(args.batch, args.outdir, args.model, args.layer)
        return
    
    # Single file mode
    if args.file:
        data = load_substance_file(args.file)
        synthesize_compound(
            title=data["title"],
            pos_prompts=data["positive"],
            neg_prompts=data["negative"],
            output_dir=args.outdir,
            model_id=args.model,
            layer=args.layer,
            normalize=not args.no_normalize
        )
        return
    
    # Inline mode
    if args.title and args.pos and args.neg:
        synthesize_compound(
            title=args.title,
            pos_prompts=args.pos,
            neg_prompts=args.neg,
            output_dir=args.outdir,
            model_id=args.model,
            layer=args.layer,
            normalize=not args.no_normalize
        )
        return
    
    parser.error("Specify --file, --batch, or (--title + --pos + --neg)")


if __name__ == "__main__":
    main()
