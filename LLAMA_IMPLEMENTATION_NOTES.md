# DejaVu Llama 3.2 3B Implementation Notes

## Overview
This document summarizes all changes made to adapt the DejaVu framework for Llama 3.2 3B, with the goal of proving that DejaVu's sparse prediction approach fails on Llama architecture due to embedding drift across layers.

---

## Key Hypothesis Being Tested

**DejaVu's Core Assumption (from OPT):**
- Token embeddings remain relatively stable layer-to-layer
- A predictor trained on layer N's input can predict which neurons/heads will be active in layer N

**Why This Fails on Llama:**
- Rotary Position Embeddings (RoPE) continuously modify embeddings
- SiLU activation doesn't create strict sparsity like ReLU
- Embeddings drift significantly between layers
- Therefore, a predictor cannot accurately predict downstream layer activity

---

## Architectural Differences: OPT vs Llama 3.2 3B

| Component | OPT | Llama 3.2 3B | Impact on DejaVu |
|-----------|-----|--------------|------------------|
| **Normalization** | LayerNorm (with bias) | RMSNorm (no bias) | Different normalization affects embedding magnitude |
| **Positional Encoding** | Learned absolute embeddings | Rotary Position Embeddings (RoPE) | RoPE causes embedding rotation at each layer |
| **Activation Function** | ReLU | SiLU (Swish) | SiLU doesn't create strict zeros, affects sparsity pattern |
| **MLP Structure** | `fc1 → ReLU → fc2` | `gate_proj + up_proj → SiLU → down_proj` | Gated MLP has different sparsity characteristics |
| **Attention** | Standard Multi-Head Attention | Grouped Query Attention (GQA) | 8 KV heads shared across 24 query heads |
| **Bias** | Has bias in linear layers | No bias in linear layers | Affects weight loading |

### Llama 3.2 3B Specific Parameters
- Hidden size: 3072
- Intermediate size: 8192
- Number of layers: 28
- Attention heads: 24
- KV heads: 8 (GQA ratio: 3)
- Vocab size: 128256
- RMS norm epsilon: 1e-5
- RoPE theta: 500000.0

---

## Files Created

### 1. `Decentralized_FM_alpha/modules/hf_llama_module.py`
**Purpose:** Core Llama module for inference

**Key Classes:**
- `LlamaEmbeddings`: Token embeddings (no positional - RoPE in attention)
- `LlamaAttention`: Attention with RoPE and GQA support
- `LlamaMLP`: Gated MLP with SiLU activation
- `LlamaBlock`: Full decoder layer
- `LlamaLMHead`: Final norm + output projection

**Key Differences from OPT:**
```python
# OPT: Positional embeddings added in embedding layer
hidden_states = inputs_embeds + position_embeds

# Llama: No positional in embedding, RoPE applied in attention
hidden_states = self.embed_tokens(input_ids)  # Just token embeddings
# RoPE applied later:
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

```python
# OPT MLP: Simple two-layer
hidden = self.fc1(x)
hidden = self.activation_fn(hidden)  # ReLU
output = self.fc2(hidden)

# Llama MLP: Gated architecture
gate = self.gate_proj(x)
up = self.up_proj(x)
output = self.down_proj(self.act_fn(gate) * up)  # SiLU gating
```

---

### 2. `Decentralized_FM_alpha/modules/hf_llama_module_save.py`
**Purpose:** Data collection for sparse predictor training

**Data Collected Per Layer:**
| File | Shape | Description |
|------|-------|-------------|
| `mlp_x_{layer}.mmap` | (400000, 3072) | MLP input before RMSNorm |
| `mlp_label_{layer}.mmap` | (400000, 8192) | MLP activation: SiLU(gate) * up |
| `att_x_{layer}.mmap` | (400000, 3072) | Attention input before RMSNorm |
| `att_label_{layer}.mmap` | (400000, 24) | Attention head importance (output norm) |

**Key Difference in Label Collection:**
```python
# OPT: Binary label based on ReLU
label = (fc1_output > 0).float()  # Strict binary

# Llama: Continuous activation values
activation = self.act_fn(gate) * up  # SiLU doesn't create zeros
# We save the actual values and threshold later during training
```

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

**Weight Mapping:**
```python
# HuggingFace → DejaVu format
'model.embed_tokens.weight' → 'embed_tokens.weight' (in pytorch_embs.pt)
'model.norm.weight' → 'norm.weight' (in pytorch_lm_head.pt)
'model.layers.{i}.self_attn.q_proj.weight' → 'self_attn.q_proj.weight'
'model.layers.{i}.self_attn.k_proj.weight' → 'self_attn.k_proj.weight'
'model.layers.{i}.self_attn.v_proj.weight' → 'self_attn.v_proj.weight'
'model.layers.{i}.self_attn.o_proj.weight' → 'self_attn.o_proj.weight'
'model.layers.{i}.mlp.gate_proj.weight' → 'gate_proj.weight'
'model.layers.{i}.mlp.up_proj.weight' → 'up_proj.weight'
'model.layers.{i}.mlp.down_proj.weight' → 'down_proj.weight'
'model.layers.{i}.input_layernorm.weight' → 'input_layernorm.weight'
'model.layers.{i}.post_attention_layernorm.weight' → 'post_attention_layernorm.weight'
```

---

### 4. `Decentralized_FM_alpha/run_infer_llama_3b_c4_1gpu.sh`
**Purpose:** Run basic inference on C4 validation set

**Key Arguments:**
```bash
--model-type llama           # Use Llama modules
--num-layers 28              # Llama 3.2 3B has 28 layers
--max-layers 28
```

---

### 5. `Decentralized_FM_alpha/run_infer_llama_3b_collect_sp_data.sh`
**Purpose:** Collect training data for sparse predictor

**Key Arguments:**
```bash
--model-type llama-save      # Use data collection modules
export DATA_PATH=./data/llama_3b_c4
```

---

### 6. `sparse_predictor/main_mlp_llama.py`
**Purpose:** Train MLP sparse predictor for Llama

**Key Differences from OPT:**
```python
# Llama configuration
CONFIG = {
    'llama-3b': {
        'num_layer': 28,
        'd': 3072,              # hidden_size
        'intermediate': 8192,    # intermediate_size (NOT d*4 like OPT)
        'h': 24,
        'kv_h': 8,
    },
}

# Predictor dimensions
predictor = nn.Sequential(
    nn.Linear(3072, 1000, bias=None),   # hidden_size → low_rank
    nn.Linear(1000, 8192, bias=None),   # low_rank → intermediate_size
)
```

**Label Generation Difference:**
```python
# OPT: Binary labels (ReLU creates zeros)
def generate_label(y):
    return (y > 0).float()

# Llama: Threshold-based (SiLU doesn't create strict zeros)
def generate_label_llama(y, threshold=0.0):
    return (torch.abs(y) > threshold).float()
```

---

### 7. `sparse_predictor/run_c4_mlp_llama.sh`
**Purpose:** Train predictor for all 28 layers

---

## Files Modified

### `Decentralized_FM_alpha/pipeline_parallel/dist_pipeline_inference_mask_greedy_token_pipe_sync.py`

**Changes Made:**

1. **Added Llama to `_get_embedding_size()`:**
```python
elif self.model_type in ["llama", "llama-save"]:
    from transformers import LlamaConfig
    config = LlamaConfig.from_pretrained(self.model_name)
    return config.hidden_size
```

2. **Added Llama to `_create_layers()`:**
```python
elif self.model_type == "llama":
    from modules.hf_llama_module import (
        LlamaEmbeddings as GPTEmbeddings,
        LlamaBlock as GPTBlock,
        LlamaLMHead as GPTLMHead,
    )
elif self.model_type == "llama-save":
    from modules.hf_llama_module_save import (
        LlamaEmbeddings as GPTEmbeddings,
        LlamaBlock as GPTBlock,
        LlamaLMHead as GPTLMHead,
    )
```

---

## Expected Results for Paper

### Prediction Accuracy Comparison

| Model | Expected Recall | Reason |
|-------|-----------------|--------|
| OPT-175B | ~0.95+ | Stable embeddings, ReLU creates clear sparsity |
| Llama 3.2 3B | **Significantly Lower** | RoPE causes embedding drift, SiLU doesn't create strict zeros |

### Metrics to Report
1. **Recall per layer**: Percentage of active neurons correctly predicted
2. **True Sparsity**: Actual sparsity in the model
3. **Classifier Sparsity**: Sparsity predicted by classifier
4. **Layer-wise embedding drift**: Cosine similarity between layer inputs

### Hypothesis Verification
- If Llama Recall << OPT Recall across layers, the hypothesis is verified
- Document embedding drift by measuring cosine similarity of same token across layers
- Show that prediction gets worse for deeper layers (more drift accumulated)

---

## Execution Steps

```bash
# Step 1: Install dependencies
pip3 install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install transformers

# Step 2: Convert checkpoint
cd Decentralized_FM_alpha
python convert_llama_checkpoint.py \
    --model-name meta-llama/Llama-3.2-3B \
    --save-path ./pretrained_models/llama-3.2-3b \
    --auth-token YOUR_TOKEN

# Step 3: Test inference
./run_infer_llama_3b_c4_1gpu.sh

# Step 4: Collect training data
./run_infer_llama_3b_collect_sp_data.sh

# Step 5: Train predictors
cd ../sparse_predictor
./run_c4_mlp_llama.sh

# Step 6: Compare results with OPT
# Document Recall values for each layer
```

---

## Code Comments for Verification

All created files include detailed comments explaining:
1. Architectural differences from OPT
2. Why certain design choices were made
3. What data is being collected and why
4. How to interpret results for the paper

These comments are marked with:
- `KEY DIFFERENCE FROM OPT:` - Highlights architectural changes
- `IMPORTANT FOR DEJAVU HYPOTHESIS:` - Explains relevance to research
- `NOTE:` - Implementation details

---

## Potential Issues and Solutions

| Issue | Solution |
|-------|----------|
| HuggingFace token required | Use `--auth-token` flag with valid token |
| Out of memory | Reduce batch size or use gradient checkpointing |
| Low prediction accuracy | This is expected! Document it for the paper |
| Different sparsity patterns | Llama uses SiLU, adjust threshold in training |

---

## Summary of Changes Count

- **New Python files created:** 4
  - `hf_llama_module.py`
  - `hf_llama_module_save.py`
  - `convert_llama_checkpoint.py`
  - `main_mlp_llama.py`

- **New Shell scripts created:** 3
  - `run_infer_llama_3b_c4_1gpu.sh`
  - `run_infer_llama_3b_collect_sp_data.sh`
  - `run_c4_mlp_llama.sh`

- **Files modified:** 1
  - `dist_pipeline_inference_mask_greedy_token_pipe_sync.py`

- **Total lines of code added:** ~1200+

---

## TESTING ORDER (IMPORTANT!)

### Step 1: Test OPT First (Verify Dependencies Work)

**Why?** Before testing Llama, we MUST verify the existing OPT implementation works. This ensures:
1. CUDA/PyTorch versions are compatible
2. Transformers version works with DejaVu
3. The pipeline infrastructure functions correctly

```bash
# 1. SSH to GPU server
ssh your-gpu-server

# 2. Navigate to DejaVu workspace
cd /path/to/DejaVu/Decentralized_FM_alpha

# 3. Install dependencies
pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.21.0
pip install cupy-cuda11x

# 4. Convert OPT-125M checkpoint (small test model)
python convert_opt_checkpoint.py \
  --model-name facebook/opt-125m \
  --save-path ./pretrained_models/opt-125m

# 5. Run OPT-125M inference test
bash run_infer_opt_125m_c4_1gpu.sh
```

**Expected Output:** Inference should complete without errors. Check for:
- No CUDA memory errors
- No missing module errors
- Generated text looks reasonable

### Step 2: Test Llama 3.2 3B

Only proceed if Step 1 succeeded!

```bash
# 1. Convert Llama checkpoint (requires HuggingFace token for gated model)
python convert_llama_checkpoint.py \
  --model-name meta-llama/Llama-3.2-3B \
  --save-path ./pretrained_models/llama-3.2-3b \
  --auth-token YOUR_HF_TOKEN

# 2. Run Llama inference test  
bash run_infer_llama_3b_c4_1gpu.sh

# 3. If inference works, collect sparse predictor training data
bash run_infer_llama_3b_collect_sp_data.sh

# 4. Train sparse predictors
bash run_c4_mlp_llama.sh
```

### Step 3: Analyze Results

Compare Llama predictor performance (Recall) vs OPT predictor performance.

**Expected Hypothesis Confirmation:**
- OPT predictors: High Recall (>90% at reasonable K)
- Llama predictors: Lower Recall (due to embedding drift)

Document these results for the paper!

---

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Core Llama module | ✅ Complete | `hf_llama_module.py` |
| Data collection module | ✅ Complete | `hf_llama_module_save.py` |
| Checkpoint converter | ✅ Complete | `convert_llama_checkpoint.py` |
| Pipeline integration | ✅ Complete | `dist_pipeline...sync.py` |
| Predictor training | ✅ Complete | `main_mlp_llama.py` |
| Inference scripts | ✅ Complete | Shell scripts |

**The Llama implementation is complete and ready for testing.**
