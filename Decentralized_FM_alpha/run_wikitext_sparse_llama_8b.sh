#!/bin/bash
# ==============================================================================
# WikiText Sparse Perplexity Evaluation - Llama 3.1 8B
# ==============================================================================

# Paths
input_file=./wikitext_eval/wikitext_test.jsonl
output_file=./wikitext_eval/output_llama_8b_sparse.jsonl
eval_file=./wikitext_eval/eval_llama_8b_sparse.txt
model_path=./pretrained_models/llama-3.1-8b
sparse_path=../checkpoint/llama-8b-sparse-predictor

# Create output directory
mkdir -p ./wikitext_eval

# Check if WikiText data exists
if [ ! -f "$input_file" ]; then
    echo "WikiText data not found. Downloading and preparing..."
    python download_wikitext.py
fi

# Clean up previous output
if [ -f "$output_file" ]; then
    echo "Removing old output file: $output_file"
    rm -f "$output_file"
fi

echo "======================================"
echo "WikiText Sparse Perplexity Evaluation"
echo "Model: Llama 3.1 8B"
echo "--------------------------------------"
echo "Sparsity Settings:"
echo "  MLP TopK: 4301 (30% kept = 70% sparsity)"
echo "  Att TopK: 1.0 (0% sparsity - Dense Attention)"
echo "======================================"

# Sparsity config
export SPARSE_PATH=${sparse_path}
export MLP_TOPK=4301
export ATT_TOPK=1.0

# Run inference with WikiText test set
args="--model-name ${model_path} \
--model-type llama-sparse \
--seed 42 \
--num-layers 32 \
--max-layers 32 \
--budget 20480 \
--num-iters 1000000 \
--dist-url tcp://127.0.0.1:9051 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${input_file} \
--output-path ${output_file}"

echo "Starting sparse inference on WikiText..."
python dist_inference_runner.py $(echo ${args}) --cuda-id 0 --rank 0

# Calculate perplexity
echo ""
echo "Calculating perplexity..."
python -c "
import json
import numpy as np

logprobs = []
with open('${output_file}') as f:
    for line in f:
        if line.strip() == '': continue
        data = json.loads(line)
        if 'result' in data and 'choices' in data['result']:
            for choice in data['result']['choices']:
                if 'logprobs' in choice and 'token_logprobs' in choice['logprobs']:
                    for lp in choice['logprobs']['token_logprobs']:
                        if lp is not None and not np.isnan(lp):
                            logprobs.append(lp)

if logprobs:
    ppl = np.exp(-np.mean(logprobs))
    print(f'WikiText Sparse Perplexity (Llama 3.1 8B): {ppl:.4f}')
    print(f'Number of tokens: {len(logprobs)}')
    with open('${eval_file}', 'w') as f:
        f.write(f'WikiText Sparse Perplexity: {ppl:.4f}\n')
        f.write(f'Tokens: {len(logprobs)}\n')
        f.write(f'MLP TopK: ${MLP_TOPK}\n')
        f.write(f'Att TopK: ${ATT_TOPK}\n')
else:
    print('No valid log probabilities found.')
"

echo ""
echo "Results saved to ${eval_file}"
