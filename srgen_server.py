"""
TNOT (Test-time Training) Model Server with OpenAI-compatible API

This server exposes TNOT-enhanced models through an OpenAI-compatible HTTP interface.
It supports both standard generation and entropy-controlled generation methods.
"""

import os
import json
import uuid
import time
import argparse
from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

from SRGen.tnot_decorator import enable_tnot


# ============================================================================
# Server Configuration
# ============================================================================

SERVER_CONFIG = {
    # Model configuration
    "model_path": "/path/to/your/model",
    "device": "cuda:0",
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    
    # TNOT parameters
    "times": 1,  # Number of optimization iterations
    "lr": 0.1,  # Learning rate for optimization
    "entropy_weight": 0.1,  # Weight for entropy loss
    
    # Entropy control parameters
    "use_entropy_control": False,
    "entropy_threshold": 3.0,
    "max_retries": 5,
    "log_entropy_control": False,
    
    # Adaptive entropy parameters
    "adaptive_entropy": False,
    "adaptive_entropy_N": 20,
    "adaptive_entropy_K": 2.0,
    "minimal_std": 0.5,
    "minimal_threshold": 1.8,
    
    # System prompt
    "system_prompt": "You are a helpful AI assistant. Please think step by step and provide your final answer in the specified format.",
    
    # Generation defaults
    "default_max_tokens": 4096,
    "default_temperature": 0.9,
}


# ============================================================================
# Pydantic Models for OpenAI API Compatibility
# ============================================================================

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=1)  # Only support n=1 for now
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# ============================================================================
# Global Model State
# ============================================================================

class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


model_state = ModelState()


# ============================================================================
# Server Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle - load model on startup, cleanup on shutdown"""
    # Startup
    print("=" * 80)
    print("Starting TNOT Model Server...")
    print("=" * 80)
    
    load_model_from_config()
    
    print("=" * 80)
    print("Server is ready to accept requests")
    print("=" * 80)
    
    yield
    
    # Shutdown
    print("Shutting down server...")
    cleanup_model()


# ============================================================================
# Model Loading and Management
# ============================================================================

def load_model_from_config():
    """Load model and tokenizer from server configuration"""
    model_path = SERVER_CONFIG["model_path"]
    device = SERVER_CONFIG["device"]
    
    print(f"\nLoading model from: {model_path}")
    print(f"Device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    model_state.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Setup environment variables for TNOT
    setup_environment()
    
    # Create TNOT-enabled model class
    print("Creating TNOT-enabled model...")
    TNOTModelClass = enable_tnot(AutoModelForCausalLM)
    
    # Load model with TNOT functionality
    print("Loading model with TNOT functionality...")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(SERVER_CONFIG["torch_dtype"], torch.bfloat16)
    
    model_state.model = TNOTModelClass.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        _attn_implementation=SERVER_CONFIG["attn_implementation"],
        device_map=device,
        trust_remote_code=True
    )
    model_state.model.eval()
    
    # Store model name
    model_state.model_name = model_path.split("/")[-1]
    
    print(f"Model loaded successfully: {model_state.model_name}")
    print(f"Model type: {model_state.model.config.model_type}")


def setup_environment():
    """Setup environment variables for TNOT from server configuration"""
    os.environ["times"] = str(SERVER_CONFIG["times"])
    os.environ["lr"] = str(SERVER_CONFIG["lr"])
    os.environ["entropy_weight"] = str(SERVER_CONFIG["entropy_weight"])
    os.environ["tokenizer_path"] = SERVER_CONFIG["model_path"]
    
    # Entropy control settings
    os.environ["use_entropy_control"] = str(SERVER_CONFIG["use_entropy_control"])
    os.environ["entropy_threshold"] = str(SERVER_CONFIG["entropy_threshold"])
    os.environ["max_retries"] = str(SERVER_CONFIG["max_retries"])
    os.environ["log_entropy_control"] = str(SERVER_CONFIG["log_entropy_control"])
    
    # Adaptive entropy settings
    os.environ["adaptive_entropy"] = str(SERVER_CONFIG["adaptive_entropy"])
    os.environ["adaptive_entropy_N"] = str(SERVER_CONFIG["adaptive_entropy_N"])
    os.environ["adaptive_entropy_K"] = str(SERVER_CONFIG["adaptive_entropy_K"])
    os.environ["minimal_std"] = str(SERVER_CONFIG["minimal_std"])
    os.environ["minimal_threshold"] = str(SERVER_CONFIG["minimal_threshold"])
    
    # Other settings
    os.environ["record_entropy"] = "False"
    os.environ["entropy_control"] = "False"
    
    print("\nEnvironment variables configured:")
    print(f"  times: {os.environ['times']}")
    print(f"  lr: {os.environ['lr']}")
    print(f"  entropy_weight: {os.environ['entropy_weight']}")
    print(f"  use_entropy_control: {os.environ['use_entropy_control']}")
    print(f"  entropy_threshold: {os.environ['entropy_threshold']}")
    print(f"  adaptive_entropy: {os.environ['adaptive_entropy']}")


def cleanup_model():
    """Cleanup model resources"""
    if model_state.model is not None:
        del model_state.model
        model_state.model = None
    
    if model_state.tokenizer is not None:
        del model_state.tokenizer
        model_state.tokenizer = None
    
    torch.cuda.empty_cache()
    print("Model resources cleaned up")


# ============================================================================
# Generation Logic
# ============================================================================

def build_prompt_from_messages(messages: List[Message]) -> str:
    """Build prompt from OpenAI message format"""
    # Extract system prompt and user message
    system_prompt = SERVER_CONFIG["system_prompt"]
    user_content = ""
    
    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_content = msg.content
        elif msg.role == "assistant":
            # For multi-turn conversations, we could handle this
            # For now, just concatenate
            user_content += f"\nAssistant: {msg.content}\n"
    
    # Build chat template
    prompt_text = model_state.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt_text


def generate_with_entropy_control(
    inputs: Dict[str, torch.Tensor],
    generation_params: Dict[str, Any],
    max_retries: int = 5
) -> tuple[str, int, int]:
    """
    Generate text with entropy control (from base_evaluator.py)
    
    Returns:
        tuple: (completion_text, retry_count, num_tokens)
    """
    os.environ["entropy_control"] = "True"
    os.environ["log_entropy_control"] = str(SERVER_CONFIG["log_entropy_control"])
    
    full_completion = ""
    current_inputs = {k: v.clone() for k, v in inputs.items()}
    retry_count = 0
    prompt_tokens = inputs['input_ids'].shape[1]
    total_new_tokens = 0
    
    while retry_count < max_retries:
        model_state.model.reset_entropy_detection()
        model_state.model.prompt_only = True
        
        outputs = model_state.model.generate(
            **current_inputs,
            **generation_params,
        )
        
        new_tokens = outputs[0][current_inputs['input_ids'].shape[1]:]
        completion_part = model_state.tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_new_tokens += len(new_tokens)
        
        del outputs
        torch.cuda.empty_cache()
        
        if model_state.model.high_entropy_detected:
            if SERVER_CONFIG["log_entropy_control"]:
                print(f"High entropy detected at retry {retry_count}, position {model_state.model.high_entropy_position}")
                print(f"Partial completion: {completion_part}")
            
            # Add the partial completion
            full_completion += completion_part
            
            # Update inputs for next iteration
            old_inputs = current_inputs
            new_text = model_state.tokenizer.decode(
                current_inputs['input_ids'][0], 
                skip_special_tokens=True
            ) + completion_part
            current_inputs = model_state.tokenizer(
                new_text, 
                return_tensors="pt", 
                add_special_tokens=False
            ).to(model_state.model.device)
            del old_inputs
            
            retry_count += 1
        else:
            full_completion += completion_part
            if SERVER_CONFIG["log_entropy_control"]:
                print(f"Generation completed normally after {retry_count} retries")
            break
    
    if retry_count >= max_retries:
        if SERVER_CONFIG["log_entropy_control"]:
            print(f"Max retries ({max_retries}) reached, continuing with normal generation")
        
        # Continue with normal generation
        os.environ["entropy_control"] = "False"
        model_state.model.prompt_only = False
        model_state.model.reset_entropy_detection()
        
        final_outputs = model_state.model.generate(
            **current_inputs,
            **generation_params,
        )
        
        final_new_tokens = final_outputs[0][current_inputs['input_ids'].shape[1]:]
        final_completion_part = model_state.tokenizer.decode(
            final_new_tokens, 
            skip_special_tokens=True
        )
        total_new_tokens += len(final_new_tokens)
        
        full_completion += final_completion_part
    
    os.environ["entropy_control"] = "False"
    
    return full_completion, retry_count, total_new_tokens


def generate_standard(
    inputs: Dict[str, torch.Tensor],
    generation_params: Dict[str, Any]
) -> tuple[str, int]:
    """
    Standard generation without entropy control
    
    Returns:
        tuple: (completion_text, num_tokens)
    """
    model_state.model.prompt_only = True
    
    outputs = model_state.model.generate(
        **inputs,
        **generation_params,
    )
    
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    completion = model_state.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return completion, len(new_tokens)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="TNOT Model Server",
    description="OpenAI-compatible API for TNOT (Test-time Training) enhanced models",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TNOT Model Server",
        "version": "1.0.0",
        "model_loaded": model_state.is_loaded(),
        "model_name": model_state.model_name if model_state.is_loaded() else None
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    if not model_state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "object": "list",
        "data": [
            {
                "id": model_state.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tnot-server",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion (OpenAI compatible)"""
    if not model_state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate request
    if request.n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported")
    
    # Build prompt from messages
    prompt_text = build_prompt_from_messages(request.messages)
    
    # Prepare inputs
    inputs = model_state.tokenizer(
        prompt_text, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model_state.model.device)
    
    prompt_tokens = inputs['input_ids'].shape[1]
    
    # Prepare generation parameters
    # Determine temperature to use
    temperature = request.temperature if request.temperature is not None else SERVER_CONFIG["default_temperature"]
    os.environ["temperature"] = str(temperature)
    
    generation_params = {
        "max_new_tokens": request.max_tokens or SERVER_CONFIG["default_max_tokens"],
        "do_sample": True if temperature > 0 else False,
    }
    
    # Only add sampling parameters when do_sample=True
    if generation_params["do_sample"]:
        generation_params["temperature"] = temperature
        os.environ["temperature"] = "1.0"
        if request.top_p is not None:
            generation_params["top_p"] = request.top_p
    
    if request.stop is not None:
        if isinstance(request.stop, str):
            generation_params["stop_strings"] = [request.stop]
        else:
            generation_params["stop_strings"] = request.stop
    
    # Reset model state
    model_state.model.reset_entropy_detection()
    model_state.model.reset_model_parameters()
    
    # Generate response
    try:
        use_entropy_control = SERVER_CONFIG["use_entropy_control"]
        
        if use_entropy_control:
            max_retries = SERVER_CONFIG["max_retries"]
            completion, retry_count, completion_tokens = generate_with_entropy_control(
                inputs, 
                generation_params, 
                max_retries
            )
        else:
            completion, completion_tokens = generate_standard(inputs, generation_params)
            retry_count = 0
        
        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(
                    completion=completion,
                    model=request.model,
                    request_id=f"chatcmpl-{uuid.uuid4().hex[:8]}"
                ),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=completion),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        return response
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


async def stream_chat_completion(completion: str, model: str, request_id: str):
    """Stream chat completion response"""
    # Send initial chunk with role
    chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={"role": "assistant"},
                finish_reason=None
            )
        ]
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    
    # Stream content word by word
    words = completion.split()
    for i, word in enumerate(words):
        content = word if i == 0 else f" {word}"
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": content},
                    finish_reason=None
                )
            ]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
    
    # Send final chunk
    chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_state.is_loaded() else "unhealthy",
        "model_loaded": model_state.is_loaded(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/v1/config/update")
async def update_config(config: Dict[str, Any]):
    """Update server configuration (non-OpenAI, custom endpoint)"""
    global SERVER_CONFIG
    
    for key, value in config.items():
        if key in SERVER_CONFIG:
            SERVER_CONFIG[key] = value
            print(f"Updated config: {key} = {value}")
    
    # Re-setup environment with new config
    setup_environment()
    
    return {"message": "Configuration updated", "config": SERVER_CONFIG}


@app.get("/v1/config")
async def get_config():
    """Get current server configuration (non-OpenAI, custom endpoint)"""
    return {"config": SERVER_CONFIG}


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TNOT Model Server")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    
    # TNOT parameters
    parser.add_argument("--times", type=int, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--use_entropy_control", action="store_true", help="Enable entropy control")
    parser.add_argument("--entropy_threshold", type=float, help="Entropy threshold")
    parser.add_argument("--max_retries", type=int, help="Maximum retries for entropy control")
    
    return parser.parse_args()


def load_config_file(config_path: str):
    """Load configuration from JSON file"""
    global SERVER_CONFIG
    
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    SERVER_CONFIG.update(config)
    print("Configuration loaded successfully")


def update_config_from_args(args):
    """Update configuration from command line arguments"""
    global SERVER_CONFIG
    
    if args.model_path:
        SERVER_CONFIG["model_path"] = args.model_path
    if args.device:
        SERVER_CONFIG["device"] = args.device
    if args.times is not None:
        SERVER_CONFIG["times"] = args.times
    if args.lr is not None:
        SERVER_CONFIG["lr"] = args.lr
    if args.use_entropy_control:
        SERVER_CONFIG["use_entropy_control"] = True
    if args.entropy_threshold is not None:
        SERVER_CONFIG["entropy_threshold"] = args.entropy_threshold
    if args.max_retries is not None:
        SERVER_CONFIG["max_retries"] = args.max_retries


if __name__ == "__main__":
    args = parse_args()
    
    # Load config file if provided
    if args.config:
        load_config_file(args.config)
    
    # Override with command line arguments
    update_config_from_args(args)
    
    # Start server
    import uvicorn
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    )
