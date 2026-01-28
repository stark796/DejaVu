#!/bin/bash
# ==============================================================================
# WikiText MLP Sparsity Sweep (0% to 70%)
# ==============================================================================

# Common Settings
export ATT_TOPK=1.0  # 0% Attention Sparsity
INPUT_FILE="./wikitext_eval/wikitext_test.jsonl"
SWEEP_DIR="./wikitext_eval/mlp_sweep"
mkdir -p "$SWEEP_DIR"

# Check data
if [ ! -f "$INPUT_FILE" ]; then
    echo "WikiText data not found. Downloading and preparing..."
    python download_wikitext.py
fi

# Function to run inference
run_sweep() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    SPARSE_PATH=$3
    TOTAL_NEURONS=$4
    PORT=$5

    echo "================================================================="
    echo "Starting Sweep for $MODEL_NAME"
    echo "Total Neurons: $TOTAL_NEURONS"
    echo "================================================================="

    # Ratios of neurons to KEEP (1.0 = 0% sparsity, 0.3 = 70% sparsity)
    # 0% 10% 20% 30% 40% 50% 60% 70% sparsity
    RATIOS=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3)

    for RATIO in "${RATIOS[@]}"; do
        # Calculate MLP_TOPK (integer math)
        MLP_TOPK=$(python -c "print(int($TOTAL_NEURONS * $RATIO))")
        SPARSITY=$(python -c "print(int((1.0 - $RATIO) * 100))")
        
        OUTPUT_FILE="$SWEEP_DIR/${MODEL_NAME}_sparsity_${SPARSITY}.jsonl"
        EVAL_FILE="$SWEEP_DIR/${MODEL_NAME}_sparsity_${SPARSITY}.txt"

        echo "-----------------------------------------------------------------"
        echo "Running ${SPARSITY}% Sparsity (Keep $RATIO)"
        echo "MLP_TOPK: $MLP_TOPK / $TOTAL_NEURONS"
        echo "Attention: Dense (1.0)"
        echo "-----------------------------------------------------------------"

        # Export variables for this run
        export SPARSE_PATH=$SPARSE_PATH
        export MLP_TOPK=$MLP_TOPK
        export ATT_TOPK=1.0

        # Model specific args
        if [ "$MODEL_NAME" == "llama_3b" ]; then
             ARGS="--model-name $MODEL_PATH \
            --model-type llama-sparse \
            --seed 42 \
            --num-layers 28 --max-layers 28 \
            --budget 32800 --num-iters 1000000 \
            --dist-url tcp://127.0.0.1:$PORT \
            --token-micro-batch-size 1 \
            --world-size 1 --pipeline-group-size 1 --data-group-size 1 \
            --pp-mode pipe_sync_sample_mask_token_pipe \
            --infer-data $INPUT_FILE \
            --output-path $OUTPUT_FILE"
        else
            # 8B settings
             ARGS="--model-name $MODEL_PATH \
            --model-type llama-sparse \
            --seed 42 \
            --num-layers 32 --max-layers 32 \
            --budget 20480 --num-iters 1000000 \
            --dist-url tcp://127.0.0.1:$PORT \
            --token-micro-batch-size 1 \
            --world-size 1 --pipeline-group-size 1 --data-group-size 1 \
            --pp-mode pipe_sync_sample_mask_token_pipe \
            --infer-data $INPUT_FILE \
            --output-path $OUTPUT_FILE"
        fi

        # Run Inference
        # Clean old output if exists
        rm -f "$OUTPUT_FILE"
        
        python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 > "${SWEEP_DIR}/log_${MODEL_NAME}_${SPARSITY}.log" 2>&1

        # Calculate Perplexity
        python -c "
import json
import numpy as np

logprobs = []
try:
    with open('$OUTPUT_FILE') as f:
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
        print(f'Sparsity: ${SPARSITY}% | MLP_TOPK: ${MLP_TOPK} | PPL: {ppl:.4f}')
        with open('$EVAL_FILE', 'w') as f:
            f.write(f'Sparsity: ${SPARSITY}%\n')
            f.write(f'MLP_TOPK: ${MLP_TOPK}\n')
            f.write(f'Perplexity: {ppl:.4f}\n')
    else:
        print(f'Error: No logprobs found for ${SPARSITY}%')
except Exception as e:
    print(f'Error calculating perplexity: {e}')
"
    done
}

# ==============================================================================
# Run Llama 3.2 3B Sweep
# ==============================================================================
run_sweep "llama_3b" \
          "./pretrained_models/llama-3.2-3b" \
          "../checkpoint/llama-3b-sparse-predictor" \
          8192 \
          9060

# ==============================================================================
# Run Llama 3.1 8B Sweep
# ==============================================================================
run_sweep "llama_8b" \
          "./pretrained_models/llama-3.1-8b" \
          "../checkpoint/llama-8b-sparse-predictor" \
          14336 \
          9070

echo "================================================================="
echo "Sweep Completed. Results in $SWEEP_DIR"
echo "================================================================="
