"""
API Server for Activation Steering Research
NuvolaProject 2026

FastAPI server providing REST endpoints for steering experiments.
"""

import os
import sys
import time
import json
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Import engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import SteeringEngine


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_LAYER = 16

app = FastAPI(
    title="Activation Steering API",
    description="REST API for activation steering experiments",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine
engine: Optional[SteeringEngine] = None


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    global engine
    print("\n" + "="*50)
    print("ACTIVATION STEERING RESEARCH INTERFACE")
    print("="*50)
    print("\nInitializing Steering Engine...")
    print("(This may take several minutes on CPU/ARM devices)")
    print("")
    
    try:
        engine = SteeringEngine(model_id=DEFAULT_MODEL)
    except Exception as e:
        print(f"\n❌ FATAL: Failed to load model: {e}")
        print("\nPossible causes:")
        print("  - Not logged in to HuggingFace (run: huggingface-cli login)")
        print("  - Insufficient RAM")
        print("  - Network error downloading model")
        raise
    
    # Warmup (optional, skip on error)
    print("\nWarming up (first inference is slow)...")
    try:
        warmup = [{"role": "user", "content": "Hi"}]
        response = ""
        for chunk in engine.generate(warmup, max_new_tokens=5):
            response += chunk
        print(f"Warmup complete: '{response[:50]}...'")
    except Exception as e:
        print(f"⚠️  Warmup skipped: {e}")
        print("   (This is OK, first real request may be slow)")
    
    # Now show the URL
    print("\n" + "="*50)
    print("✅ READY! Interface available at:")
    print("   Local:   http://localhost:8000")
    print("   Network: http://<your-ip>:8000")
    print("="*50 + "\n")


# =============================================================================
# MODELS
# =============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = True
    steering_vector: Optional[str] = None
    steering_intensity: Optional[float] = 0.0
    system_prompt: Optional[str] = None

class ExtractionRequest(BaseModel):
    prompts: List[str]
    layer: int = DEFAULT_LAYER


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/v1/info")
async def get_info():
    """Get system information."""
    return {
        "model": engine.model_id if engine else "offline",
        "device": engine.device if engine else "unknown",
        "default_layer": DEFAULT_LAYER,
        "version": "1.0.0"
    }

@app.get("/v1/vectors")
async def list_vectors():
    """List available steering vectors."""
    if not engine:
        return {"vectors": []}
    return {"vectors": engine.list_vectors()}

@app.post("/v1/vectors/reload")
async def reload_vectors():
    """Reload vectors from disk."""
    if not engine:
        raise HTTPException(503, "Engine not initialized")
    
    engine.vectors.clear()
    engine.vector_metadata.clear()
    engine._load_vectors()
    
    return {"status": "reloaded", "count": len(engine.vectors)}

@app.post("/v1/upload")
async def upload_vector(file: UploadFile = File(...)):
    """Upload a new steering vector (.pt or .png with embedded vector)."""
    if not engine:
        raise HTTPException(503, "Engine not initialized")
    
    if not (file.filename.endswith('.pt') or file.filename.endswith('.png')):
        raise HTTPException(400, "Only .pt and .png files supported")
    
    filename = os.path.basename(file.filename)
    filepath = os.path.join(engine.vectors_dir, filename)
    
    try:
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Validate PNG has embedded vector
        if filename.endswith('.png'):
            from PIL import Image
            try:
                img = Image.open(filepath)
                img.load()
                if "NEURO_VECTOR" not in img.info:
                    os.remove(filepath)
                    raise HTTPException(400, "PNG does not contain embedded steering vector")
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise HTTPException(400, f"Invalid PNG: {str(e)}")
        
        # Reload to pick up new vector
        engine._load_vectors()
        
        return {"filename": filename, "status": "uploaded"}
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(500, str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Generate completion with optional steering."""
    if not engine:
        raise HTTPException(503, "Engine not initialized")
    
    # Debug logging
    print(f"[API] Request: vector={request.steering_vector}, intensity={request.steering_intensity}")
    
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Prepend system prompt if provided
    if request.system_prompt:
        messages = [{"role": "system", "content": request.system_prompt}] + messages
    
    if request.stream:
        return StreamingResponse(
            stream_response(request, messages),
            media_type="text/event-stream"
        )
    else:
        response = engine.generate_sync(
            messages=messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            steering_vector=request.steering_vector,
            steering_intensity=request.steering_intensity
        )
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop"
            }]
        }

def stream_response(request: ChatRequest, messages: list):
    """Generate streaming response."""
    generator = engine.generate(
        messages=messages,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        steering_vector=request.steering_vector,
        steering_intensity=request.steering_intensity
    )
    
    for chunk in generator:
        data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
    
    yield "data: [DONE]\n\n"


# =============================================================================
# STATIC FILES
# =============================================================================

# Static directory is inside system/
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")

# Also check parent directory (for when running from project root)
if not os.path.exists(static_dir):
    parent_static = os.path.join(os.path.dirname(base_dir), "static")
    if os.path.exists(parent_static):
        static_dir = parent_static
    else:
        os.makedirs(static_dir)
        print(f"[WARNING] Created empty static directory: {static_dir}")

print(f"[INFO] Serving static files from: {static_dir}")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
