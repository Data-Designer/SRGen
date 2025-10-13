# 🔄 Self-Reflective Generation at Test Time (SRGen)

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2510.02919-b31b1b.svg)](https://arxiv.org/abs/2510.02919)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

*A lightweight test-time framework for proactive error prevention in LLM reasoning*

[Paper](https://arxiv.org/abs/2510.02919) | [Installation](#-installation) | [Quick Start](#-quick-start) | [Usage](#-usage)

</div>

---

## 📖 Introduction

**SRGen** (Self-Reflective Generation at Test Time) is a novel test-time framework that enables large language models to **reflect before generating** at uncertain points during token generation. Unlike traditional post-hoc refinement methods that are reactive and computationally expensive, SRGen proactively prevents errors by:

- 🎯 **Dynamic Entropy Monitoring**: Identifying high-uncertainty tokens during generation using adaptive entropy thresholds
- 🔧 **Test-Time Optimization**: Training corrective vectors at critical points to steer generation toward more reliable paths
- 🚀 **Plug-and-Play**: Zero additional training required, works with any pre-trained transformer model
- ⚡ **Bounded Overhead**: Minimal latency increase while significantly improving reasoning quality

### Key Results

**Table: Performance of SRGen on Mathematical Reasoning Benchmarks**

Reporting Avg@5, Cons@5, and Pass@5 metrics. Values in parentheses show absolute improvement vs. Base.

| Model | Benchmark | Avg@5 | | Cons@5 | | Pass@5 | |
|-------|-----------|-------|----------|--------|----------|--------|----------|
| | | **Base** | **w/SRGen** | **Base** | **w/SRGen** | **Base** | **w/SRGen** |
| **Qwen2.5-Math-7B** | AIME2024 | 14.6 | 22.0 **(+7.4)** | 6.7 | 23.3 **(+16.6)** | 40.0 | 40.0 **(→0.0)** |
| | AIME2025 | 6.0 | 9.3 **(+3.3)** | 6.7 | 6.7 **(→0.0)** | 13.0 | 26.7 **(+13.7)** |
| | HMMT2025 | 1.3 | 3.3 **(+2.0)** | 0.0 | 0.0 **(→0.0)** | 6.0 | 13.3 **(+7.3)** |
| | AMC | 34.0 | 41.2 **(+7.2)** | 34.0 | 41.0 **(+7.0)** | 49.0 | 52.0 **(+3.0)** |
| **Distill-Qwen-7B** | AIME2024 | 49.3 | 61.3 **(+12.0)** | 50.0 | 63.3 **(+13.3)** | 73.0 | 80.0 **(+7.0)** |
| | AIME2025 | 35.3 | 42.7 **(+7.4)** | 33.0 | 46.7 **(+13.7)** | 53.0 | 60.0 **(+7.0)** |
| | HMMT2025 | 15.3 | 18.0 **(+2.7)** | 16.7 | 16.7 **(→0.0)** | 23.3 | 33.0 **(+9.7)** |
| | AMC | 51.0 | 51.2 **(+0.2)** | 51.0 | 51.0 **(→0.0)** | 64.0 | 64.0 **(→0.0)** |
| **Distill-Llama-8B** | AIME2024 | 48.0 | 52.7 **(+4.7)** | 46.7 | 63.3 **(+16.6)** | 70.0 | 76.7 **(+6.7)** |
| | AIME2025 | 30.7 | 32.7 **(+2.0)** | 26.7 | 33.3 **(+6.6)** | 50.0 | 50.0 **(→0.0)** |
| | HMMT2025 | 14.0 | 16.0 **(+2.0)** | 10.0 | 13.0 **(+3.0)** | 20.0 | 33.0 **(+13.0)** |
| | AMC | 50.0 | 50.6 **(+0.6)** | 53.0 | 53.0 **(→0.0)** | 57.0 | 57.0 **(→0.0)** |
| **Qwen3-32B** | AIME2024 | 76.7 | 82.7 **(+6.0)** | 80.0 | 90.0 **(+10.0)** | 90.0 | 93.0 **(+3.0)** |
| | AIME2025 | 70.7 | 76.0 **(+5.3)** | 73.0 | 76.7 **(+3.7)** | 86.7 | 86.7 **(→0.0)** |
| | HMMT2025 | 23.3 | 28.0 **(+4.7)** | 26.7 | 26.7 **(→0.0)** | 33.0 | 43.3 **(+10.3)** |
| | AMC | 54.0 | 56.8 **(+2.8)** | 54.0 | 57.0 **(+3.0)** | 60.0 | 61.0 **(+1.0)** |

**Highlights:**
- 🏆 **Best Cons@5 improvement**: +16.6% on AIME2024 (Qwen2.5-Math-7B & Distill-Llama-8B)
- 🚀 **Best Avg@5 improvement**: +12.0% on AIME2024 (Distill-Qwen-7B)
- 📊 **Consistent gains** across multiple models and benchmarks
- 🎯 **Strongest on challenging datasets** (AIME, HMMT) where reasoning quality matters most

---

## 🌟 Features

- ✅ **Proactive Error Prevention**: Intervenes at high-uncertainty points during generation, not after errors occur
- ✅ **Test-Time Only**: No additional training or fine-tuning required
- ✅ **Model-Agnostic**: Works with any Hugging Face Transformers CausalLM model
- ✅ **Composable**: Integrates seamlessly with other techniques (RLHF, SLOT, etc.)
- ✅ **Configurable**: Flexible entropy thresholds, adaptive strategies, and optimization parameters
- ✅ **OpenAI-Compatible API**: Easy deployment with RESTful API server

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ VRAM for 7B models

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/2020-qqtcg/SRGen.git
cd SRGen

# Install required packages
pip install -r SRGen/requirements.txt

# Additional evaluation dependencies (optional)
pip install sympy timeout-decorator latex2sympy2_extended
```

### Flash Attention (Optional, Recommended)

For faster inference with Flash Attention 2:

```bash
pip install flash-attn --no-build-isolation
```

---

## 🚀 Quick Start

### 1. Using Pre-configured Evaluation Scripts

The easiest way to get started is to use our pre-configured evaluation scripts:

```bash
# Run AIME 2024 evaluation with DeepSeek-R1-Distill-Qwen-7B
bash scripts/parallel_aime_distill_qwen.sh
```

**Script Configuration:**

The script automatically configures SRGen with optimized parameters:

```bash
export model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

python -m SRGen.aime_evaluator \
    --model_path $model_path \
    --parallel \
    --max_parallel_gpus 4 \
    --average 5 \                    # Number of samples per question
    --split train \
    --version 2024 \
    --times 3 \                       # Optimization iterations
    --lr 0.01 \                       # Learning rate
    --entropy_threshold 3.0 \         # Base entropy threshold
    --entropy_weight 0.05 \           # Entropy loss weight
    --use_entropy_control \           # Enable SRGen
    --adaptive_entropy \              # Dynamic threshold adjustment
    --adaptive_entropy_N 25 \         # Window size for adaptation
    --adaptive_entropy_K 4 \          # Standard deviation multiplier
    --do_sample \
    --temperature 0.6 \
    --max_new_tokens 32768
```

### 2. Basic Python Usage with TNOT Decorator

Here's how to use SRGen in your own code:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from SRGen.tnot_decorator import enable_tnot
import os

# Step 1: Apply TNOT decorator to enable SRGen functionality
TNOTModel = enable_tnot(AutoModelForCausalLM)

# Step 2: Load model and tokenizer
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model = TNOTModel.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Step 3: Configure SRGen parameters via environment variables
os.environ["times"] = "3"              # Number of optimization steps
os.environ["lr"] = "0.01"              # Learning rate
os.environ["entropy_weight"] = "0.05"  # Entropy loss weight
os.environ["entropy_threshold"] = "3.0"
os.environ["temperature"] = "0.6"
os.environ["tokenizer_path"] = model_path

# Optional: Enable adaptive entropy thresholding
os.environ["adaptive_entropy"] = "True"
os.environ["adaptive_entropy_N"] = "25"
os.environ["adaptive_entropy_K"] = "4"
os.environ["minimal_std"] = "0.5"
os.environ["minimal_threshold"] = "1.8"

# Step 4: Generate with entropy control
prompt = "Solve this problem step by step: What is the sum of the first 10 odd numbers?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Use generate_with_entropy_control for automatic retry on high entropy
generation_params = {
    "max_new_tokens": 2048,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "pad_token_id": tokenizer.eos_token_id
}

# Helper function for entropy-controlled generation
def generate_with_entropy_control(model, inputs, generation_params, max_retries=5):
    """Generate text with entropy control and automatic retry"""
    os.environ["entropy_control"] = "True"
    
    full_completion = ""
    current_inputs = inputs.copy()
    retry_count = 0
    
    while retry_count < max_retries:
        model.reset_entropy_detection()
        model.prompt_only = True
        
        outputs = model.generate(**current_inputs, **generation_params)
        new_tokens = outputs[0][current_inputs['input_ids'].shape[1]:]
        completion_part = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if model.high_entropy_detected:
            print(f"High entropy detected at retry {retry_count}")
            full_completion += completion_part
            
            # Continue generation from current position
            new_text = tokenizer.decode(current_inputs['input_ids'][0]) + completion_part
            current_inputs = tokenizer(new_text, return_tensors="pt").to(model.device)
            retry_count += 1
        else:
            full_completion += completion_part
            break
    
    return full_completion

response = generate_with_entropy_control(model, inputs, generation_params, max_retries=5)
print(response)
```

---

## 📊 Configuration Guide

### Core SRGen Parameters

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| **Optimization Steps** | `times` | 1 | Number of gradient descent iterations |
| **Learning Rate** | `lr` | 0.1 | Learning rate for corrective vector optimization |
| **Entropy Weight** | `entropy_weight` | 0.1 | Weight λ for entropy minimization loss |
| **Temperature** | `temperature` | 1.0 | Temperature for probability distribution |

### Entropy Control Parameters

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| **Enable Control** | `entropy_control` | False | Enable entropy-based early stopping |
| **Base Threshold** | `entropy_threshold` | 3.0 | Static entropy threshold value |
| **Max Retries** | `max_retries` | 5 | Maximum retry attempts |

### Adaptive Entropy Parameters

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| **Enable Adaptive** | `adaptive_entropy` | False | Enable dynamic threshold adjustment |
| **Window Size** | `adaptive_entropy_N` | 20 | Number of recent tokens for statistics |
| **Std Dev Multiplier** | `adaptive_entropy_K` | 2.0 | Multiplier for standard deviation |
| **Minimal Std** | `minimal_std` | 0.5 | Minimum standard deviation floor |
| **Minimal Threshold** | `minimal_threshold` | 1.8 | Minimum threshold value |

**Adaptive Threshold Formula:**
```
threshold = max(mean(history[-N:]) + K * std(history[-N:]), minimal_threshold)
std = max(std(history[-N:]), minimal_std)
```

---

## 🌐 API Server Deployment

SRGen provides a ready-to-use server that deploys a TNOT-decorated model with OpenAI-compatible API endpoints. This allows you to quickly set up a service that leverages SRGen's entropy-controlled generation capabilities.

### Quick Start

**Step 1: Configure the server**

Edit `server_config.json` to set your model and SRGen parameters:

```json
{
  "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "device": "cuda:0",
  "torch_dtype": "bfloat16",
  
  "times": 3,
  "lr": 0.01,
  "entropy_weight": 0.05,
  
  "use_entropy_control": true,
  "entropy_threshold": 3.0,
  "adaptive_entropy": true,
  "adaptive_entropy_N": 25,
  "adaptive_entropy_K": 4
}
```

**Step 2: Start the server**

```bash
# Use the provided startup script
bash start_server.sh

# Or specify custom host/port
HOST=0.0.0.0 PORT=8000 bash start_server.sh

# Or run directly
python srgen_server.py --config server_config.json --host 0.0.0.0 --port 8000
```

The server will automatically:
- Apply TNOT decorator to the model
- Load the model with your configured parameters
- Expose OpenAI-compatible API endpoints

**Step 3: Make requests**

Use the OpenAI Python client or any HTTP client:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required
)

response = client.chat.completions.create(
    model="srgen",
    messages=[{"role": "user", "content": "What is 15 * 17?"}],
    temperature=0.6,
    max_tokens=2048
)

print(response.choices[0].message.content)
```

Or use curl:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "srgen",
    "messages": [{"role": "user", "content": "Solve: What is the derivative of x^2?"}],
    "temperature": 0.6,
    "max_tokens": 2048
  }'
```

> **Note**: The server internally uses the TNOT decorator and `generate_with_entropy_control` to provide self-reflective generation capabilities through a simple API interface

---

## 📈 Evaluation

### Available Evaluators

```bash
# AIME (American Invitational Mathematics Examination)
python -m SRGen.aime_evaluator --model_path MODEL --version 2024

# GSM8K (Grade School Math)
python -m SRGen.gsm8k_evaluator --model_path MODEL --split test

# MATH-500
python -m SRGen.math_evaluator --model_path MODEL

# GPQA (Graduate-Level Google-Proof Q&A)
python -m SRGen.gpqa_evaluator --model_path MODEL
```

### Common Evaluator Arguments

```bash
--model_path PATH           # Path to model or HF model ID
--use_entropy_control       # Enable SRGen
--times INT                 # Optimization iterations (default: 1)
--lr FLOAT                  # Learning rate (default: 0.01)
--entropy_threshold FLOAT   # Entropy threshold (default: 3.0)
--entropy_weight FLOAT      # Entropy loss weight (default: 0.05)
--adaptive_entropy          # Enable adaptive thresholding
--average INT               # Number of samples per question
--parallel                  # Enable multi-GPU parallel evaluation
--max_parallel_gpus INT     # Maximum GPUs to use
```

---

## 🔬 How SRGen Works

### The Joint Optimization Loss

At each high-entropy point, SRGen optimizes a corrective vector `δ` using:

```
L_total = (1 - λ) * L_CE + λ * L_entropy

where:
  L_CE = CrossEntropy(logits, labels)           # Contextual fidelity
  L_entropy = -Σ p(x) * log(p(x))               # Confidence maximization
  λ = entropy_weight                             # Balance parameter
```

### Adaptive Entropy Thresholding

```python
# Static threshold (simple)
if entropy > threshold:
    trigger_reflection()

# Adaptive threshold (dynamic)
history = recent_entropies[-N:]
mean_entropy = mean(history)
std_entropy = max(std(history), minimal_std)
dynamic_threshold = mean_entropy + K * std_entropy
dynamic_threshold = max(dynamic_threshold, minimal_threshold)

if entropy > dynamic_threshold:
    trigger_reflection()
```

---

## 📝 Citation

If you find SRGen useful in your research, please cite our paper:

```bibtex
@article{mu2025srgen,
  title={Self-Reflective Generation at Test Time},
  author={Mu, Jian and Zhang, Qixin and Wang, Zhiyong and Yang, Menglin and Qiu, Shuang and Qin, Chengwei and Dai, Zhongxiang and Shu, Yao},
  journal={arXiv preprint arXiv:2510.02919},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests.

---

## 🙏 Acknowledgments

- Built on [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Inspired by test-time adaptation and self-reflection techniques
- Thanks to all contributors and the research community

---

<div align="center">

**⭐ Star us on GitHub — it motivates us a lot!**

Made with ❤️ by the SRGen Team

</div>
