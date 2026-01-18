#!/bin/bash
# ==============================================================================
# DejaVu Inference Script for Llama 3.2 3B
# ==============================================================================
#
# This script runs inference with Llama 3.2 3B on the C4 validation set.
# It's designed to test whether DejaVu's sparse prediction approach works
# on Llama architecture.
#
# IMPORTANT FOR DEJAVU HYPOTHESIS:
# - OPT's embedding stability allows prediction of downstream layer activity
# - Llama's RoPE and architecture cause significant embedding drift
# - We expect prediction accuracy to be significantly worse than OPT
#
# Usage:
#   ./run_infer_llama_3b_c4_1gpu.sh
#
# Prerequisites:
#   1. Convert Llama checkpoint: python convert_llama_checkpoint.py
#   2. Download C4 validation data: python c4_val/get_data.py
#
# ==============================================================================

# Paths
file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_c4_val_llama_3b.jsonl
eval_file=./c4_val/eval_c4_val_llama_3b.txt
model_path=./pretrained_models/llama-3.2-3b

# Create directories if needed
mkdir -p ./c4_val

# Create a dummy c4_valid.jsonl if it doesn't exist for initial testing
if [ ! -f "$file" ]; then
    echo '{"text": "This is a test sentence for perplexity calculation."}' > "$file"
    echo "Created dummy test file: $file"
fi

echo "======================================"
echo "DejaVu Inference - Llama 3.2 3B"
echo "======================================"
echo "Input file: ${file}"
echo "Output file: ${output_file}"
echo "Model path: ${model_path}"
echo ""

# For Llama 3.2 3B:
# - num-layers: 28
# - hidden-size: 3072
# - num-attention-heads: 24
# - num-kv-heads: 8 (GQA)
# - intermediate-size: 8192
# - world-size: 1 (single GPU)
# - pipeline-group-size: 1

ARGS="--model-name ${model_path} \
--model-type llama \
--seed 42 \
--fp16 \
--num-layers 28 \
--max-layers 28 \
--budget 32800 \
--num-iters 1000000 \
--dist-url tcp://127.0.0.1:9033 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

# Ensure we are in the correct directory
cd "$(dirname "$0")"

echo "Starting inference..."
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
            if 'result' in item and item['result']:
                for i, r in enumerate(item['result']):
                    if 'token_logprob' in r and r['token_logprob'] is not None:
                        logprobs.append(r['token_logprob'])
    
    if logprobs:
        ppl = np.exp(-np.mean(logprobs))
        print(f'Perplexity: {ppl:.4f}')
        print(f'Number of tokens: {len(logprobs)}')
        print(f'Average log probability: {np.mean(logprobs):.4f}')
        
        with open('${eval_file}', 'w') as f:
            f.write(f'Perplexity: {ppl:.4f}\n')
            f.write(f'Number of tokens: {len(logprobs)}\n')
            f.write(f'Average log probability: {np.mean(logprobs):.4f}\n')
    else:
        print('No valid log probabilities found in output file.')
except FileNotFoundError:
    print('Output file not found. Inference may have failed.')
except json.JSONDecodeError as e:
    print(f'Error parsing output file: {e}')
"

echo ""
echo "Done! Results saved to ${eval_file}"
