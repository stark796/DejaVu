#!/bin/bash
# ==============================================================================
# Calculate Dense Perplexity for Llama 3.1 8B
# ==============================================================================

# Paths
file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_llama_8b_dense.jsonl
eval_file=./c4_val/eval_llama_8b_dense.txt
model_path=./pretrained_models/llama-3.1-8b

# Check input
if [ ! -f "$file" ]; then
    echo "ERROR: C4 data not found at ${file}"
    exit 1
fi

echo "======================================"
echo "Running Dense Perplexity Evaluation"
echo "Model: Llama 3.1 8B"
echo "======================================"

# Run Inference
# Using same settings as data collection but model-type is llama-save (dense)
# Reduced budget to 20480 to avoid OOM
args="--model-name ${model_path} \
--model-type llama-save \
--seed 42 \
--num-layers 32 \
--max-layers 32 \
--budget 20480 \
--num-iters 200 \
--dist-url tcp://127.0.0.1:9040 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

echo "Starting inference..."
# Use nohup pattern if desired, but running foreground for verify
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
            # Skip first token logprob (usually None or 0)
            logprobs += data['result']['choices'][0]['logprobs']['token_logprobs'][1:]

mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print(f'Dense Perplexity: {perplexity}')
" > $eval_file

cat $eval_file
