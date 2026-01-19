#!/bin/bash
# ==============================================================================
# Collect Sparse Predictor Training Data for Llama 3.2 3B
# ==============================================================================
#
# This script runs Llama 3.2 3B inference on C4 validation set and collects
# training data for the sparse predictor.
#
# Data collected for each layer:
#   - mlp_x_{layer}.mmap: Input to MLP block (query for predictor)
#   - mlp_label_{layer}.mmap: MLP activation pattern (label for predictor)
#   - att_x_{layer}.mmap: Input to attention block (query for attention predictor)
#   - att_label_{layer}.mmap: Attention head importance (label for attention predictor)
#
# IMPORTANT: This uses llama-save model type which saves intermediate activations.
#
# Prerequisites:
#   1. Convert Llama checkpoint: python convert_llama_checkpoint.py
#   2. Download C4 data: python c4_train/get_data.py
#
# Usage:
#   ./run_infer_llama_3b_collect_sp_data.sh
#
# Output:
#   Data saved to: ./data/llama_3b_c4/
# ==============================================================================

# Paths - use c4_val data we already downloaded
file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_llama_3b_collect.jsonl
model_path=./pretrained_models/llama-3.2-3b
data_path=./data/llama_3b_c4

# Create directories
mkdir -p ${data_path}

# Check if data exists
if [ ! -f "$file" ]; then
    echo "ERROR: C4 data not found at ${file}"
    echo "Please run: python c4_val/getdata.py"
    exit 1
fi

echo "======================================"
echo "Collecting Sparse Predictor Training Data"
echo "Model: Llama 3.2 3B"
echo "======================================"
echo "Input file: ${file}"
echo "Output path: ${data_path}"
echo "Model path: ${model_path}"
echo ""

# Set environment variable for data path
export DATA_PATH=${data_path}

# For Llama 3.2 3B:
# - num-layers: 28
# - hidden-size: 3072
# - Uses llama-save model type for data collection

ARGS="--model-name ${model_path} \
--model-type llama-save \
--seed 42 \
--fp16 \
--num-layers 28 \
--max-layers 28 \
--budget 22800 \
--num-iters 2000 \
--dist-url tcp://127.0.0.1:9034 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

echo "Starting data collection inference..."
echo "This will process samples and save intermediate activations."
echo ""

# Ensure we are in the correct directory
cd "$(dirname "$0")"

python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0

echo ""
echo "======================================"
echo "Data Collection Complete!"
echo "======================================"
echo ""
echo "Data saved to: ${data_path}/"
echo ""
echo "Files created per layer:"
echo "  - mlp_x_{layer}.mmap: MLP input query"
echo "  - mlp_label_{layer}.mmap: MLP activation label"
echo "  - att_x_{layer}.mmap: Attention input query"
echo "  - att_label_{layer}.mmap: Attention head importance label"
echo ""
echo "Next step: Train sparse predictors with:"
echo "  cd ../sparse_predictor && ./run_c4_mlp_llama.sh"
