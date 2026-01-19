#!/bin/bash
# Run cross-layer prediction for multiple target layers
# Shows how prediction degrades with layer distance

echo "=" * 60
echo "CROSS-LAYER PREDICTION STUDY"
echo "Query: Layer 0 → Target: Layer M"
echo "=" * 60

# Test key layers
for layer in 0 5 10 14 20 27; do
    echo ""
    echo "Running: Layer 0 → Layer $layer"
    python main_mlp_llama_crosslayer.py --target-layer $layer --epochs 10
done

echo ""
echo "=========================================="
echo "SUMMARY: Compare recall as layer distance increases"
echo "If embedding drift exists, recall should drop for higher layers"
echo "=========================================="
