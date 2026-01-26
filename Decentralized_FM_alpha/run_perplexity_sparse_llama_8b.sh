#!/bin/bash
# ==============================================================================
# Calculate Sparse Perplexity for Llama 3.1 8B
# ==============================================================================

# Paths
file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_llama_8b_sparse.jsonl
eval_file=./c4_val/eval_llama_8b_sparse.txt
model_path=./pretrained_models/llama-3.1-8b
sparse_path=../checkpoint/llama-8b-sparse-predictor

# Check input
if [ ! -f "$file" ]; then
    echo "ERROR: C4 data not found at ${file}"
    exit 1
fi

echo "======================================"
echo "Running Sparse Perplexity Evaluation"
echo "Model: Llama 3.1 8B"
echo "--------------------------------------"
# Target Sparsity:
#   MLP TopK: 7168 (50% density of 14336)
#   Att TopK: 1.0  (100% density - NO sparsity)
echo "--------------------------------------"
echo "Target Sparsity:"
echo "  MLP TopK: 7168 (50%)"
echo "  Att TopK: 1.0  (100%)"
echo "======================================"

# Export Sparsity Config
export SPARSE_PATH=${sparse_path}
export MLP_TOPK=7168
export ATT_TOPK=1.0  # 1.0 = Keep all heads (0% sparsity)
export SPARSE_ATT=1  # Still required to use the sparse module logic

# Run Inference
# model-type: llama-sparse
args="--model-name ${model_path} \
--model-type llama-sparse \
--seed 42 \
--num-layers 32 \
--max-layers 32 \
--budget 20480 \
--num-iters 200 \
--dist-url tcp://127.0.0.1:9041 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

echo "Starting inference..."
python dist_inference_runner.py $(echo ${args}) --cuda-id 0 --rank 0

# Calculate Perplexity
echo "Calculating perplexity..."
python -c "import json
import numpy as np

logprobs = []
with open('$output_file') as f:
    for line in f:
        if line.strip() == '': continue
        data = json.loads(line)
        if 'result' in data:
            logprobs += data['result']['choices'][0]['logprobs']['token_logprobs'][1:]

mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print(f'Sparse Perplexity: {perplexity}')
" > $eval_file

cat $eval_file
