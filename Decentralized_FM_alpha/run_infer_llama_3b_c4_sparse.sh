#!/bin/bash
# ==============================================================================
# DejaVu Sparse Inference Script for Llama 3.2 3B
# ==============================================================================
#
# This script runs SPARSE inference with Llama 3.2 3B using trained predictors.
# It uses the sparse module that skips inactive neurons and attention heads.
#
# COMPARISON WITH DENSE:
# - Run run_infer_llama_3b_c4_1gpu.sh first to get dense perplexity
# - Run this script to get sparse perplexity
# - The perplexity should be similar (within ~1%) if predictors are good
# - Latency should be reduced due to sparsity
#
# Prerequisites:
#   1. Convert Llama checkpoint: python convert_llama_checkpoint.py
#   2. Collect training data: ./run_infer_llama_3b_collect_sp_data.sh
#   3. Train MLP predictors: cd ../sparse_predictor && ./run_c4_mlp_llama.sh
#   4. Train Attention predictors: cd ../sparse_predictor && ./run_c4_att_llama.sh
#
# Usage:
#   ./run_infer_llama_3b_c4_sparse.sh
#
# Environment Variables:
#   SPARSE_PATH: Path to trained predictor checkpoints
#   MLP_TOPK: Number of active MLP neurons (default: 1000)
#   ATT_TOPK: Fraction of active attention heads (default: 0.7)
#
# ==============================================================================

# Paths
file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_c4_val_llama_3b_sparse.jsonl
eval_file=./c4_val/eval_c4_val_llama_3b_sparse.txt
model_path=./pretrained_models/llama-3.2-3b

# Sparse predictor settings
export SPARSE_PATH=../checkpoint/llama-3b-sparse-predictor
export MLP_TOPK=4500           # Number of active neurons (out of 8192) - matches training
export ATT_TOPK=0.7            # Fraction of active attention heads

# Create directories if needed
mkdir -p ./c4_val

# Check if predictors exist
if [ ! -d "$SPARSE_PATH" ]; then
    echo "WARNING: Predictor checkpoint directory not found: $SPARSE_PATH"
    echo "Make sure to train predictors first!"
    echo ""
fi

echo "======================================"
echo "DejaVu SPARSE Inference - Llama 3.2 3B"
echo "======================================"
echo "Input file: ${file}"
echo "Output file: ${output_file}"
echo "Model path: ${model_path}"
echo ""
echo "Sparse Settings:"
echo "  Predictor path: ${SPARSE_PATH}"
echo "  MLP TopK: ${MLP_TOPK} neurons"
echo "  Attention TopK: ${ATT_TOPK} fraction"
echo ""

# For Llama 3.2 3B with sparse inference:
# - Uses llama-sparse model type
# - Loads trained MLP and attention predictors
# - Skips inactive neurons and heads

ARGS="--model-name ${model_path} \
--model-type llama-sparse \
--seed 42 \
--fp16 \
--num-layers 28 \
--max-layers 28 \
--budget 32800 \
--num-iters 1000000 \
--dist-url tcp://127.0.0.1:9035 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

# Ensure we are in the correct directory
cd "$(dirname "$0")"

echo "Starting sparse inference..."
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0

echo ""
echo "Computing perplexity..."
python -c "
import json
import numpy as np

logprobs = []

try:
    with open('${output_file}') as f:
        for line in f:
            if line.strip() == '':
                continue
            item = json.loads(line)
            if 'result' in item and 'choices' in item['result']:
                for choice in item['result']['choices']:
                    if 'logprobs' in choice and 'token_logprobs' in choice['logprobs']:
                        for lp in choice['logprobs']['token_logprobs']:
                            if lp is not None and not np.isnan(lp):
                                logprobs.append(lp)
    
    if logprobs:
        ppl = np.exp(-np.mean(logprobs))
        print(f'Perplexity (Sparse): {ppl:.4f}')
        print(f'Number of tokens: {len(logprobs)}')
        print(f'Average log probability: {np.mean(logprobs):.4f}')
        
        with open('${eval_file}', 'w') as f:
            f.write(f'Perplexity (Sparse): {ppl:.4f}\n')
            f.write(f'Number of tokens: {len(logprobs)}\n')
            f.write(f'Average log probability: {np.mean(logprobs):.4f}\n')
            f.write(f'MLP TopK: ${MLP_TOPK}\n')
            f.write(f'Attention TopK: ${ATT_TOPK}\n')
    else:
        print('No valid log probabilities found in output file.')
except FileNotFoundError:
    print('Output file not found. Inference may have failed.')
except json.JSONDecodeError as e:
    print(f'Error parsing output file: {e}')
"

echo ""
echo "Done! Results saved to ${eval_file}"
echo ""
echo "Compare with dense inference results:"
echo "  Dense: ./c4_val/eval_c4_val_llama_3b.txt"
echo "  Sparse: ${eval_file}"
