"""
Steering Engine for Activation Steering Research
NuvolaProject 2026

Core inference engine with steering vector support.
"""

import os
import glob
import torch
import logging
from typing import Dict, Optional, List, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_safety_warning():
    print("\n" + "!"*60)
    print("  WARNING: SOMA_TK ACTIVATED")
    print("  You are running a model with chemically altered weights.")
    print("  Outputs may be hallucinations, biased, or factually wrong.")
    print("  INTENDED FOR RED-TEAMING & RESEARCH ONLY.")
    print("!"*60 + "\n")

print_safety_warning()


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_STEERING_LAYER = 16  # Validated for Llama 3.2 3B
MIN_RAM_GB = 10  # Minimum recommended RAM for Llama 3.2 3B


def check_system_memory():
    """Check if system has enough RAM for the model."""
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        logger.info(f"System RAM: {total_ram_gb:.1f}GB total, {available_ram_gb:.1f}GB available")
        
        if total_ram_gb < MIN_RAM_GB:
            logger.warning(f"")
            logger.warning(f"⚠️  LOW MEMORY WARNING")
            logger.warning(f"   System has {total_ram_gb:.1f}GB RAM, but Llama 3.2 3B needs ~{MIN_RAM_GB}GB")
            logger.warning(f"   The process may be killed by the OS (OOM killer)")
            logger.warning(f"   Consider using a machine with more RAM")
            logger.warning(f"")
            return False
        return True
    except ImportError:
        # psutil not installed, skip check
        return True


class SteeringEngine:
    """
    LLM inference engine with activation steering support.
    
    Supports loading steering vectors from .pt files and applying them
    at inference time with configurable intensity.
    """
    
    def __init__(
        self, 
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        vectors_dir: str = "vectors"
    ):
        self.model_id = model_id
        self.vectors_dir = vectors_dir
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None
        self.vectors: Dict[str, torch.Tensor] = {}
        self.vector_metadata: Dict[str, dict] = {}
        
        self._load_model()
        self._load_vectors()

    def _detect_device(self) -> str:
        """Detect best available hardware."""
        if torch.cuda.is_available():
            logger.info("Device: CUDA")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Device: MPS (Apple Silicon)")
            return "mps"
        logger.info("Device: CPU")
        return "cpu"

    def _load_model(self):
        """Load the language model."""
        # Check system memory before loading
        check_system_memory()
        
        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"Target device: {self.device}")
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logger.info("Tokenizer loaded")
        
        logger.info("Loading model weights (this may take a few minutes on CPU)...")
        
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16
            ).to("mps")
        else:
            # CPU mode - simpler loading without device_map for better ARM64 compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True  # Load efficiently without device_map
            )
            logger.info("Model loaded to CPU")
        
        # Get layer count
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            n_layers = len(self.model.model.layers)
            logger.info(f"Model ready: {n_layers} layers")
        else:
            logger.info("Model ready")

    def _load_vectors(self):
        """Load all steering vectors from the vectors directory."""
        if not os.path.exists(self.vectors_dir):
            os.makedirs(self.vectors_dir)
            logger.info(f"Created vectors directory: {self.vectors_dir}")
            return
        
        # Load .pt files
        pt_files = glob.glob(os.path.join(self.vectors_dir, "*.pt"))
        
        for filepath in pt_files:
            try:
                filename = os.path.basename(filepath)
                data = torch.load(filepath, map_location=self.device, weights_only=False)
                
                vector = None
                metadata = {
                    "layer": DEFAULT_STEERING_LAYER,
                    "normalized": False,
                    "model": "unknown"
                }
                
                # Handle both dict format and raw tensor
                if isinstance(data, dict) and "vector" in data:
                    vector = data["vector"]
                    if "metadata" in data:
                        meta = data["metadata"]
                        metadata["layer"] = meta.get("layer", DEFAULT_STEERING_LAYER)
                        metadata["normalized"] = meta.get("normalized", False)
                        metadata["model"] = meta.get("model", "unknown")
                        metadata["title"] = meta.get("title", filename)
                        metadata["stats"] = meta.get("stats", {})
                elif isinstance(data, torch.Tensor):
                    vector = data
                    metadata["title"] = filename.replace(".pt", "")
                
                if isinstance(vector, torch.Tensor):
                    vector = vector.to(dtype=self.model.dtype, device=self.device)
                    self.vectors[filename] = vector
                    self.vector_metadata[filename] = metadata
                    logger.info(f"Loaded: {filename} (layer={metadata['layer']}, norm={vector.norm().item():.4f})")
                    
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
        
        # Load steganographic .png files
        png_files = glob.glob(os.path.join(self.vectors_dir, "*.png"))
        
        for filepath in png_files:
            try:
                filename = os.path.basename(filepath)
                
                from PIL import Image
                import base64
                import zlib
                import io
                
                img = Image.open(filepath)
                img.load()
                
                if "NEURO_VECTOR" not in img.info:
                    logger.debug(f"Skipping PNG without embedded vector: {filename}")
                    continue
                
                # Extract and decompress vector
                b64_str = img.info["NEURO_VECTOR"]
                compressed_data = base64.b64decode(b64_str)
                
                if img.info.get("NEURO_COMPRESSION") == "zlib":
                    raw_data = zlib.decompress(compressed_data)
                    buffer = io.BytesIO(raw_data)
                else:
                    buffer = io.BytesIO(compressed_data)
                
                vector = torch.load(buffer, map_location=self.device, weights_only=False)
                
                if isinstance(vector, torch.Tensor):
                    vector = vector.to(dtype=self.model.dtype, device=self.device)
                    self.vectors[filename] = vector
                    
                    # Extract metadata from PNG
                    metadata = {
                        "layer": DEFAULT_STEERING_LAYER,
                        "model": "unknown",
                        "title": filename.replace(".png", "").replace("_", " ").title()
                    }
                    
                    if "NEURO_LAYER" in img.info:
                        try:
                            metadata["layer"] = int(img.info["NEURO_LAYER"])
                        except ValueError:
                            pass
                    
                    if "NEURO_MODEL" in img.info:
                        metadata["model"] = img.info["NEURO_MODEL"]
                    
                    if "NEURO_NAME" in img.info:
                        metadata["title"] = img.info["NEURO_NAME"]
                    
                    self.vector_metadata[filename] = metadata
                    logger.info(f"Loaded PNG: {filename} (layer={metadata['layer']}, norm={vector.norm().item():.4f})")
                    
            except Exception as e:
                logger.error(f"Failed to load PNG {filepath}: {e}")

    def list_vectors(self) -> List[dict]:
        """Return list of available vectors with metadata."""
        result = []
        for name, vector in self.vectors.items():
            meta = self.vector_metadata.get(name, {})
            result.append({
                "name": name,
                "title": meta.get("title", name),
                "layer": meta.get("layer", DEFAULT_STEERING_LAYER),
                "norm": vector.norm().item(),
                "model": meta.get("model", "unknown"),
                "stats": meta.get("stats", {})
            })
        return result

    def _apply_steering(self, vector_name: str, intensity: float) -> list:
        """Apply steering vector via forward hook."""
        hooks = []
        
        if not vector_name or vector_name not in self.vectors:
            logger.info(f"No steering applied (vector: {vector_name})")
            return hooks
        
        if intensity == 0:
            logger.info(f"Steering intensity is 0, skipping")
            return hooks
        
        vector = self.vectors[vector_name]
        metadata = self.vector_metadata.get(vector_name, {})
        target_layer = metadata.get("layer", DEFAULT_STEERING_LAYER)
        
        logger.info(f"Applying steering: {vector_name} @ layer {target_layer}, intensity {intensity}")
        logger.info(f"Vector shape: {vector.shape}, norm: {vector.norm().item():.4f}")
        
        def steering_hook(module, args, output):
            nonlocal vector
            
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Ensure vector has right shape for broadcasting
            # hidden shape: (batch, seq_len, hidden_dim)
            # vector shape: (hidden_dim,)
            if vector.dim() == 1 and hidden.dim() == 3:
                if vector.shape[0] == hidden.shape[-1]:
                    # Reshape vector for broadcasting: (1, 1, hidden_dim)
                    steering_vec = vector.unsqueeze(0).unsqueeze(0)
                    modified_hidden = hidden + (steering_vec * intensity)
                    
                    if isinstance(output, tuple):
                        return (modified_hidden,) + output[1:]
                    return modified_hidden
                else:
                    logger.warning(f"Dimension mismatch: vector {vector.shape[0]} vs hidden {hidden.shape[-1]}")
            
            # Return original if we can't apply steering
            return output
        
        # Register hook
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            if target_layer < len(layers):
                handle = layers[target_layer].register_forward_hook(steering_hook)
                hooks.append(handle)
                logger.info(f"Hook registered at layer {target_layer}")
            else:
                logger.error(f"Layer {target_layer} out of range (max: {len(layers)-1})")
        else:
            logger.error("Cannot access model layers")
        
        return hooks

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        steering_vector: Optional[str] = None,
        steering_intensity: float = 0.0
    ) -> Generator[str, None, None]:
        """
        Generate response with optional steering.
        
        Args:
            messages: Chat history in OpenAI format
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            steering_vector: Name of vector file to apply
            steering_intensity: Steering strength (0-15 typical)
        
        Yields:
            Generated text chunks
        """
        # Format input
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Setup streaming
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        
        # Apply steering
        hooks = self._apply_steering(steering_vector, steering_intensity)
        
        # Generation config
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Run generation in thread
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        try:
            for chunk in streamer:
                yield chunk
        finally:
            for hook in hooks:
                hook.remove()
            thread.join()

    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        steering_vector: Optional[str] = None,
        steering_intensity: float = 0.0
    ) -> str:
        """Non-streaming generation."""
        chunks = []
        for chunk in self.generate(
            messages, max_new_tokens, temperature,
            steering_vector, steering_intensity
        ):
            chunks.append(chunk)
        return "".join(chunks)
