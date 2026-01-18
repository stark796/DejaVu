file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_c4_val_opt_125m.jsonl
eval_file=./c4_val/eval_c4_val_opt_125m.txt

# Create c4_val directory if it doesn't exist
mkdir -p ./c4_val

# Create a dummy c4_valid.jsonl if it doesn't exist for initial testing
if [ ! -f "$file" ]; then
    echo '{"text": "This is a test sentence for perplexity calculation."}' > "$file"
fi
    
echo "start running ${file}"

# For OPT-125M:
# num-layers: 12
# world-size: 1 (single GPU)
# pipeline-group-size: 1

ARGS="--model-name facebook/opt-125m \
--model-type opt \
--seed 42 \
--fp16 \
--num-layers 12 \
--max-layers 12 \
--budget 32800 \
--num-iters 1000000 \
--dist-url tcp://127.0.0.1:9032 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--checkpoint ./pretrained_models/opt-125m \
--infer-data ${file} \
--output-path ${output_file}"

# Ensure we are in the correct directory
cd "$(dirname "$0")"

python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0

python -c "import json
import numpy as np

logprobs = []

try:
    with open('$output_file') as f:
        for line in f:
            if line.strip() == '':
                continue
            item = json.loads(line)
            if 'result' not in item:
                continue
            logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
    if logprobs:
        mean_logprob = sum(logprobs) / len(logprobs)
        perplexity = np.exp(-mean_logprob)
        print('perplexity:', perplexity)
    else:
        print('No logprobs found in output.')
except FileNotFoundError:
    print('Output file not found.')
" > $eval_file
cat $eval_file
