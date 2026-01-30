# DejaVu Changes Summary

## Overview
This document summarizes all modifications made to adapt the DejaVu framework for Llama 3.2 3B and fix single-GPU inference issues.

**Goal:** Prove that DejaVu's sparse prediction approach fails on Llama because token embeddings change significantly layer-to-layer.

---

## Files Created (Llama Implementation)

### 1. `Decentralized_FM_alpha/modules/hf_llama_module.py`
**Purpose:** Core Llama module for inference

**Key Components:**
- `LlamaEmbeddings`: Token embeddings (no positional - RoPE handled in attention)
- `LlamaAttention`: Attention with RoPE and GQA support
- `LlamaMLP`: Gated MLP with SiLU activation
- `LlamaBlock`: Full decoder layer
- `LlamaLMHead`: Final norm + output projection

---

### 2. `Decentralized_FM_alpha/modules/hf_llama_module_save.py`
**Purpose:** Llama module with data collection for sparse predictor training

**Data Collected Per Layer:**
- `mlp_x_{layer}.mmap`: MLP input before RMSNorm
- `mlp_label_{layer}.mmap`: MLP activation (SiLU(gate) * up)
- `att_x_{layer}.mmap`: Attention input before RMSNorm
- `att_label_{layer}.mmap`: Attention head importance scores

---

### 3. `Decentralized_FM_alpha/convert_llama_checkpoint.py`
**Purpose:** Convert HuggingFace Llama checkpoint to DejaVu format

**Output Structure:**
```
pretrained_models/llama-3.2-3b/
├── config.json
├── tokenizer files
├── pytorch_embs.pt          # embed_tokens.weight
├── pytorch_lm_head.pt       # norm.weight + lm_head.weight
├── pytorch_0.pt             # Layer 0 weights
├── pytorch_1.pt             # Layer 1 weights
...
└── pytorch_27.pt            # Layer 27 weights
```

---

### 4. `Decentralized_FM_alpha/run_infer_llama_3b_c4_1gpu.sh`
**Purpose:** Run inference with Llama 3.2 3B on C4 validation set

**Key Arguments:**
- `--model-type llama`
- `--num-layers 28`
- `--max-layers 28`

---

### 5. `Decentralized_FM_alpha/run_infer_llama_3b_collect_sp_data.sh`
**Purpose:** Collect training data for sparse predictor

**Key Arguments:**
- `--model-type llama-save`
- `DATA_PATH=./data/llama_3b_c4`

---

### 6. `sparse_predictor/main_mlp_llama.py`
**Purpose:** Train MLP sparse predictor for Llama

**Llama Configuration:**
```python
CONFIG = {
    'llama-3b': {
        'num_layer': 28,
        'd': 3072,              # hidden_size
        'intermediate': 8192,   # intermediate_size
        'h': 24,
        'kv_h': 8,
    },
}
```

---

### 7. `sparse_predictor/run_c4_mlp_llama.sh`
**Purpose:** Train predictor for all 28 layers

---

### 8. `LLAMA_IMPLEMENTATION_NOTES.md`
**Purpose:** Detailed documentation of all implementation changes

---

## Files Modified (Bug Fixes)

### 1. `pipeline_parallel/dist_pipeline_inference_mask_greedy_token_pipe_sync.py`

**Changes:**
1. Added Llama support to `_get_embedding_size()`:
```python
elif self.model_type in ["llama", "llama-save"]:
    from transformers import LlamaConfig
    config = LlamaConfig.from_pretrained(self.model_name)
    return config.hidden_size
```

2. Added Llama support to `_create_layers()`:
```python
elif self.model_type == "llama":
    from modules.hf_llama_module import (...)
elif self.model_type == "llama-save":
    from modules.hf_llama_module_save import (...)
```

3. **Single GPU support**: Skip NCCL send/recv when `pipeline_group_size=1`

4. **Fixed attention mask**: Process mask for single GPU at step 0

5. **Critical fix**: Reset `i_current_token` AFTER `super()._merge_cached_seqs_and_attentions()` call to prevent off-by-one index error

---

### 2. `pipeline_parallel/dist_pipeline_inference_greedy_token_pipe_sync.py`

**Changes:**
1. **Single GPU profiling**: Only log compute time, skip send/recv profiling when `pipeline_group_size=1`
2. Added directory creation for profiling output

---

### 3. `utils/dist_inference_utils.py`

**Changes:**
- Added single GPU case to `distributed_inference_mask_iter()`:
```python
if comm.get_world_size() == 1:
    request_processor.add_result(...)
    write_scenario_state(...)
```

---

### 4. `modules/generation_utils.py`

**Changes:**
- Added try/except for transformers import compatibility:
```python
try:
    from transformers.generation.beam_constraints import (...)
except ImportError:
    from transformers.generation_beam_constraints import (...)
```

---

### 5. `task_datasets/inference_data.py`

**Changes:**
- Handle both `"prompt"` and `"text"` keys for C4 dataset:
```python
text = item.get("prompt") or item.get("text", "")
```

---

### 6. `run_infer_opt_125m_c4_1gpu.sh`

**Changes:**
- Added `--model-name ./pretrained_models/opt-125m` checkpoint path
- Added `max_tokens` to dummy C4 data

---

## Architectural Differences: OPT vs Llama 3.2 3B

| Component | OPT | Llama 3.2 3B |
|-----------|-----|--------------|
| **Normalization** | LayerNorm (with bias) | RMSNorm (no bias) |
| **Position Encoding** | Learned absolute embeddings | Rotary Position Embeddings (RoPE) |
| **Activation** | ReLU | SiLU (Swish) |
| **MLP Structure** | `fc1 → ReLU → fc2` | `gate_proj + up_proj → SiLU → down_proj` |
| **Attention** | Standard Multi-Head Attention | Grouped Query Attention (GQA) |
| **Bias** | Has bias in linear layers | No bias in linear layers |

### Llama 3.2 3B Parameters
- Hidden size: 3072
- Intermediate size: 8192
- Number of layers: 28
- Attention heads: 24
- KV heads: 8 (GQA ratio: 3)
- Vocab size: 128256

---

## Test Results

### OPT-125M (Baseline)
- **Status:** ✅ Working
- **Perplexity:** 28.85
- **Tokens generated:** 16
- **Time:** ~30 seconds

### Llama 3.2 3B
- **Status:** ✅ Working
- **Tokens generated:** 15
- **Time:** 32.75 seconds

---

## Why DejaVu Fails on Llama (Hypothesis)

**DejaVu's Core Assumption (from OPT):**
- Token embeddings remain relatively stable layer-to-layer
- A predictor trained on layer N's input can predict which neurons/heads will be active

**Why This Fails on Llama:**
1. **RoPE:** Rotary Position Embeddings continuously modify embeddings at each layer
2. **SiLU:** Doesn't create strict sparsity like ReLU
3. **Embedding Drift:** Embeddings change significantly between layers
4. **Conclusion:** A predictor cannot accurately predict downstream layer activity

---

## Environment

- **Python:** 3.13
- **PyTorch:** 2.6.0+cu124
- **CUDA:** 12.4
- **GPU:** RTX 6000 Ada (48GB)
- **Key packages:** transformers, cupy-cuda12x, boto3, loguru, netifaces

---

## Evaluation & Analysis Phase

### 1. Comprehensive Benchmarking
- **Benchmarks Covered:**
  - **0-shot:** BoolQ, Winogrande, RTE, ARC-e, ARC-c, HellaSwag, OpenBookQA
  - **Multi-shot:** MMLU, GPQA, MedMCQA
- **Key Finding:** Documented accuracy trade-offs at varying sparsity levels.

### 2. Sparse Inference with Llama
- Successfully ran sparse inference on **Wikitext** and **C4**.
- Validated perplexity calculations for both 3B and 8B models.
