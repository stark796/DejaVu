#!/bin/bash
# ==============================================================================
# Collect Sparse Predictor Training Data for Llama 3.1 8B
# ==============================================================================
#
# This script runs Llama 3.1 8B inference on C4 validation set and collects
# training data for the sparse predictor.
#
# Llama 3.1 8B Architecture:
#   - 32 layers
#   - hidden_size: 4096
#   - intermediate_size: 14336
#
# Usage:
#   ./run_infer_llama_8b_collect_sp_data.sh
#
# Output:
#   Data saved to: ./data/llama_8b_c4/
# ==============================================================================

# Paths
file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_llama_8b_collect.jsonl
model_path=./pretrained_models/llama-3.1-8b
data_path=./data/llama_8b_c4

# Create directories
mkdir -p ${data_path}

# Check if data exists
if [ ! -f "$file" ]; then
    echo "ERROR: C4 data not found at ${file}"
    echo "Please create c4_valid.jsonl first."
    exit 1
fi

echo "======================================"
echo "Collecting Sparse Predictor Training Data"
echo "Model: Llama 3.1 8B (32 layers)"
echo "======================================"
echo "Input file: ${file}"
echo "Output path: ${data_path}"
echo "Model path: ${model_path}"
echo ""

# Set environment variable for data path
export DATA_PATH=${data_path}

# For Llama 3.1 8B:
# - num-layers: 32
# - hidden-size: 4096
# - NOTE: Using fp32 (not fp16) because Llama 3.1 RoPE scaling causes NaN in fp16

ARGS="--model-name ${model_path} \
--model-type llama-save \
--seed 42 \
--num-layers 32 \
--max-layers 32 \
--budget 40960 \
--num-iters 2000 \
--dist-url tcp://127.0.0.1:9035 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

echo "Starting data collection inference..."
echo ""

cd "$(dirname "$0")"

python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0

echo ""
echo "======================================"
echo "Data Collection Complete!"
echo "======================================"
echo ""
echo "Data saved to: ${data_path}/"
echo ""
echo "Next step: Train sparse predictors with:"
echo "  cd ../sparse_predictor && ./run_c4_mlp_llama_8b.sh"
