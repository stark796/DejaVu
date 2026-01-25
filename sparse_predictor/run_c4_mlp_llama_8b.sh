#!/bin/bash
# ==============================================================================
# Train MLP Sparse Predictor for Llama 3.1 8B - All Layers
# ==============================================================================
#
# This script trains sparse predictors for all 32 layers of Llama 3.1 8B.
#
# Prerequisites:
#   1. Run data collection: run_infer_llama_8b_collect_sp_data.sh
#   2. Verify data exists in: ../data/llama_8b_c4/
#
# Usage:
#   ./run_c4_mlp_llama_8b.sh
#
# Output:
#   Checkpoints saved to: ../checkpoint/llama-8b-sparse-predictor/
# ==============================================================================

# Create checkpoint directory
mkdir -p ../checkpoint/llama-8b-sparse-predictor

# Llama 3.1 8B has 32 layers (0-31)
NUM_LAYERS=32

echo "======================================"
echo "Training MLP Sparse Predictor for Llama 3.1 8B"
echo "Total layers: $NUM_LAYERS"
echo "======================================"

for L in $(seq 0 $((NUM_LAYERS - 1))); do
    echo ""
    echo "======================================"
    echo "Training Layer $L / $((NUM_LAYERS - 1))"
    echo "======================================"
    
    python main_mlp_llama.py \
        --model llama-8b \
        --dataset c4 \
        --L $L \
        --D 1000 \
        --batch_size 1024 \
        --epochs 20 \
        --lr 0.0001 \
        --threshold 0.0
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for layer $L"
        exit 1
    fi
done

echo ""
echo "======================================"
echo "Training Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Train attention predictors: ./run_c4_att_llama_8b.sh"
echo "2. Run benchmarks in lm-eval-harness-adapter"
echo ""
