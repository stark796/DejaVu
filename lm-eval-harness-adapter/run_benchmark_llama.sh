#!/bin/bash
# Benchmark Inference Script for DejaVu Llama
# This script runs inference on benchmark task data (e.g., MMLU, GPQA, MedMCQA)
#
# Usage:
#   ./run_benchmark_llama.sh --input <input.jsonl> --output <output.jsonl> [--sparse]
#
# Arguments:
#   --input   Path to input JSONL file from generate_task_data.py
#   --output  Path to output JSONL file for evaluate_task_result.py
#   --sparse  Use sparse inference (default: dense)

set -e

# Parse arguments
INPUT_FILE=""
OUTPUT_FILE=""
SPARSE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --sparse)
            SPARSE_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 --input <input.jsonl> --output <output.jsonl> [--sparse]"
    exit 1
fi

# Model configuration
export MODEL_NAME="meta-llama/Llama-3.2-3B"
export MODEL_PATH="../checkpoint/llama-3b"

# Sparsity configuration
if [ "$SPARSE_MODE" = true ]; then
    export SPARSE_PATH="../checkpoint/llama-3b-sparse-predictor"
    export MLP_TOPK=2458   # 30% active (70% sparsity)
    export ATT_TOPK=1.0    # 100% attention
    MODEL_TYPE="llama-sparse"
    echo "Running SPARSE inference with 70% MLP sparsity"
else
    MODEL_TYPE="llama"
    echo "Running DENSE inference"
fi

# Run the benchmark inference
python benchmark_inference.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-path "$MODEL_PATH" \
    --tokenizer-name "$MODEL_NAME" \
    --model-type "$MODEL_TYPE"

echo "Inference complete. Results saved to $OUTPUT_FILE"
