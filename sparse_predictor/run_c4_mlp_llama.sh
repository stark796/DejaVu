#!/bin/bash
# ==============================================================================
# Train MLP Sparse Predictor for Llama 3.2 3B - All Layers
# ==============================================================================
#
# This script trains sparse predictors for all 28 layers of Llama 3.2 3B.
#
# IMPORTANT FOR DEJAVU HYPOTHESIS TESTING:
# - Compare the Recall values with OPT models
# - OPT typically achieves >0.95 Recall
# - Llama is expected to have significantly lower Recall due to embedding drift
#
# Prerequisites:
#   1. Run data collection: run_infer_llama_3b_collect_sp_data.sh
#   2. Verify data exists in: ../data/llama_3b_c4/
#
# Usage:
#   ./run_c4_mlp_llama.sh
#
# Output:
#   Checkpoints saved to: ../checkpoint/llama-3b-sparse-predictor/
# ==============================================================================

# Create checkpoint directory
mkdir -p ../checkpoint/llama-3b-sparse-predictor

# Llama 3.2 3B has 28 layers (0-27)
NUM_LAYERS=28

echo "======================================"
echo "Training MLP Sparse Predictor for Llama 3.2 3B"
echo "Total layers: $NUM_LAYERS"
echo "======================================"

for L in $(seq 0 $((NUM_LAYERS - 1))); do
    echo ""
    echo "======================================"
    echo "Training Layer $L / $((NUM_LAYERS - 1))"
    echo "======================================"
    
    python main_mlp_llama.py \
        --model llama-3b \
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
echo "1. Compare Recall values with OPT models"
echo "2. If Llama Recall << OPT Recall, DejaVu hypothesis is verified"
echo ""
