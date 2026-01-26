#!/bin/bash
# =============================================================================
# Full Benchmark Evaluation Script for DejaVu Llama 3.1 8B
# Runs ALL MMLU (57 tasks) + GPQA (3 tasks) + MedMCQA (1 task)
# =============================================================================

set -e

# Configuration
RESULTS_DIR="./results_8b"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# ALL MMLU TASKS (57 total)
# =============================================================================
MMLU_ALL=(
    "hendrycksTest-abstract_algebra"
    "hendrycksTest-anatomy"
    "hendrycksTest-astronomy"
    "hendrycksTest-business_ethics"
    "hendrycksTest-clinical_knowledge"
    "hendrycksTest-college_biology"
    "hendrycksTest-college_chemistry"
    "hendrycksTest-college_computer_science"
    "hendrycksTest-college_mathematics"
    "hendrycksTest-college_medicine"
    "hendrycksTest-college_physics"
    "hendrycksTest-computer_security"
    "hendrycksTest-conceptual_physics"
    "hendrycksTest-econometrics"
    "hendrycksTest-electrical_engineering"
    "hendrycksTest-elementary_mathematics"
    "hendrycksTest-formal_logic"
    "hendrycksTest-global_facts"
    "hendrycksTest-high_school_biology"
    "hendrycksTest-high_school_chemistry"
    "hendrycksTest-high_school_computer_science"
    "hendrycksTest-high_school_european_history"
    "hendrycksTest-high_school_geography"
    "hendrycksTest-high_school_government_and_politics"
    "hendrycksTest-high_school_macroeconomics"
    "hendrycksTest-high_school_mathematics"
    "hendrycksTest-high_school_microeconomics"
    "hendrycksTest-high_school_physics"
    "hendrycksTest-high_school_psychology"
    "hendrycksTest-high_school_statistics"
    "hendrycksTest-high_school_us_history"
    "hendrycksTest-high_school_world_history"
    "hendrycksTest-human_aging"
    "hendrycksTest-human_sexuality"
    "hendrycksTest-international_law"
    "hendrycksTest-jurisprudence"
    "hendrycksTest-logical_fallacies"
    "hendrycksTest-machine_learning"
    "hendrycksTest-management"
    "hendrycksTest-marketing"
    "hendrycksTest-medical_genetics"
    "hendrycksTest-miscellaneous"
    "hendrycksTest-moral_disputes"
    "hendrycksTest-moral_scenarios"
    "hendrycksTest-nutrition"
    "hendrycksTest-philosophy"
    "hendrycksTest-prehistory"
    "hendrycksTest-professional_accounting"
    "hendrycksTest-professional_law"
    "hendrycksTest-professional_medicine"
    "hendrycksTest-professional_psychology"
    "hendrycksTest-public_relations"
    "hendrycksTest-security_studies"
    "hendrycksTest-sociology"
    "hendrycksTest-us_foreign_policy"
    "hendrycksTest-virology"
    "hendrycksTest-world_religions"
)

# GPQA TASKS (3 total)
GPQA_ALL=(
    "gpqa_diamond"
    "gpqa_extended"
    "gpqa_main"
)

# MEDICAL TASKS
MEDICAL_ALL=(
    "medmcqa"
)

# =============================================================================
# FUNCTION: Run single benchmark
# =============================================================================
run_benchmark() {
    local task_name=$1
    local num_fewshot=$2
    local mode=$3  # "dense" or "sparse"
    
    local safe_name=$(echo "$task_name" | tr '-' '_')
    local input_file="${RESULTS_DIR}/${safe_name}_input.jsonl"
    local output_file="${RESULTS_DIR}/${safe_name}_${mode}.jsonl"
    local result_file="${RESULTS_DIR}/${safe_name}_${mode}_result.json"
    
    echo ""
    echo "=========================================="
    echo "Task: $task_name (${num_fewshot}-shot, $mode)"
    echo "=========================================="
    
    # Step 1: Generate task data (only needed once per task)
    if [ ! -f "$input_file" ]; then
        echo "[1/3] Generating task data..."
        python generate_task_data.py \
            --task-name "$task_name" \
            --output-file "$input_file" \
            --num-fewshot "$num_fewshot" 2>/dev/null || {
            echo "WARNING: Failed to generate data for $task_name, skipping..."
            return 1
        }
    else
        echo "[1/3] Using cached task data"
    fi
    
    # Step 2: Run inference (using 8B script)
    echo "[2/3] Running $mode inference..."
    if [ "$mode" = "sparse" ]; then
        ./run_benchmark_llama_8b.sh \
            --input "$input_file" \
            --output "$output_file" \
            --sparse
    else
        ./run_benchmark_llama_8b.sh \
            --input "$input_file" \
            --output "$output_file"
    fi
    
    # Step 3: Evaluate
    echo "[3/3] Evaluating results..."
    python evaluate_task_result.py \
        --result-file "$output_file" \
        --task-name "$task_name" \
        --model-type llama \
        --num-fewshot "$num_fewshot" > "$result_file" 2>&1
    
    # Extract accuracy
    local acc=$(grep -o '"acc":[^,}]*' "$result_file" | head -1 || echo "N/A")
    echo "Result: $acc"
    
    return 0
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

MODE="${1:-both}"  # Options: dense, sparse, both
BENCHMARK="${2:-all}"  # Options: mmlu, gpqa, medical, all

echo "=============================================="
echo "DejaVu Llama 3.1 8B Benchmark Suite"
echo "Mode: $MODE"
echo "Benchmark: $BENCHMARK"
echo "Results Dir: $RESULTS_DIR"
echo "=============================================="

# Track results
SUMMARY_FILE="${RESULTS_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"
echo "Llama 3.1 8B Benchmark Results" > "$SUMMARY_FILE"
echo "==============================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Sparsity: 50% MLP, 0% Attention" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Run MMLU
if [ "$BENCHMARK" = "mmlu" ] || [ "$BENCHMARK" = "all" ]; then
    echo ""
    echo "### Running MMLU (${#MMLU_ALL[@]} tasks) ###"
    echo "" >> "$SUMMARY_FILE"
    echo "MMLU Results:" >> "$SUMMARY_FILE"
    
    for task in "${MMLU_ALL[@]}"; do
        if [ "$MODE" = "dense" ] || [ "$MODE" = "both" ]; then
            run_benchmark "$task" 0 "dense" && \
            echo "$task (dense): $(grep -o '"acc":[^,}]*' ${RESULTS_DIR}/$(echo $task | tr '-' '_')_dense_result.json 2>/dev/null || echo 'N/A')" >> "$SUMMARY_FILE"
        fi
        if [ "$MODE" = "sparse" ] || [ "$MODE" = "both" ]; then
            run_benchmark "$task" 0 "sparse" && \
            echo "$task (sparse): $(grep -o '"acc":[^,}]*' ${RESULTS_DIR}/$(echo $task | tr '-' '_')_sparse_result.json 2>/dev/null || echo 'N/A')" >> "$SUMMARY_FILE"
        fi
    done
fi

# Run GPQA
if [ "$BENCHMARK" = "gpqa" ] || [ "$BENCHMARK" = "all" ]; then
    echo ""
    echo "### Running GPQA (${#GPQA_ALL[@]} tasks) ###"
    echo "" >> "$SUMMARY_FILE"
    echo "GPQA Results:" >> "$SUMMARY_FILE"
    
    for task in "${GPQA_ALL[@]}"; do
        fewshot=5
        if [ "$task" = "gpqa_extended" ]; then
            fewshot=0
        fi
        
        if [ "$MODE" = "dense" ] || [ "$MODE" = "both" ]; then
            run_benchmark "$task" "$fewshot" "dense" && \
            echo "$task (dense): $(grep -o '"acc":[^,}]*' ${RESULTS_DIR}/$(echo $task | tr '-' '_')_dense_result.json 2>/dev/null || echo 'N/A')" >> "$SUMMARY_FILE"
        fi
        if [ "$MODE" = "sparse" ] || [ "$MODE" = "both" ]; then
            run_benchmark "$task" "$fewshot" "sparse" && \
            echo "$task (sparse): $(grep -o '"acc":[^,}]*' ${RESULTS_DIR}/$(echo $task | tr '-' '_')_sparse_result.json 2>/dev/null || echo 'N/A')" >> "$SUMMARY_FILE"
        fi
    done
fi

# Run Medical
if [ "$BENCHMARK" = "medical" ] || [ "$BENCHMARK" = "all" ]; then
    echo ""
    echo "### Running Medical (${#MEDICAL_ALL[@]} tasks) ###"
    echo "" >> "$SUMMARY_FILE"
    echo "Medical Results:" >> "$SUMMARY_FILE"
    
    for task in "${MEDICAL_ALL[@]}"; do
        if [ "$MODE" = "dense" ] || [ "$MODE" = "both" ]; then
            run_benchmark "$task" 0 "dense" && \
            echo "$task (dense): $(grep -o '"acc":[^,}]*' ${RESULTS_DIR}/$(echo $task | tr '-' '_')_dense_result.json 2>/dev/null || echo 'N/A')" >> "$SUMMARY_FILE"
        fi
        if [ "$MODE" = "sparse" ] || [ "$MODE" = "both" ]; then
            run_benchmark "$task" 0 "sparse" && \
            echo "$task (sparse): $(grep -o '"acc":[^,}]*' ${RESULTS_DIR}/$(echo $task | tr '-' '_')_sparse_result.json 2>/dev/null || echo 'N/A')" >> "$SUMMARY_FILE"
        fi
    done
fi

echo ""
echo "=============================================="
echo "Benchmarking complete!"
echo "Summary saved to: $SUMMARY_FILE"
echo "=============================================="

cat "$SUMMARY_FILE"
