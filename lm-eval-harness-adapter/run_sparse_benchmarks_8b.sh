#!/bin/bash
# =============================================================================
# Sparse Benchmark Script for Llama 3.1 8B
# Runs SPARSE inference only with OOM tracking
# =============================================================================

set -e

DATA_DIR="./data_8b"
RESULTS_DIR="./results_8b"
OOM_LOG="${RESULTS_DIR}/oom_skipped.txt"

mkdir -p "$DATA_DIR" "$RESULTS_DIR"
echo "OOM Skipped Prompts Log - $(date)" > "$OOM_LOG"
echo "==============================" >> "$OOM_LOG"

# =============================================================================
# 5-SHOT TASKS
# =============================================================================

# MMLU (57 subjects)
MMLU_TASKS=(
    "mmlu_abstract_algebra"
    "mmlu_anatomy"
    "mmlu_astronomy"
    "mmlu_business_ethics"
    "mmlu_clinical_knowledge"
    "mmlu_college_biology"
    "mmlu_college_chemistry"
    "mmlu_college_computer_science"
    "mmlu_college_mathematics"
    "mmlu_college_medicine"
    "mmlu_college_physics"
    "mmlu_computer_security"
    "mmlu_conceptual_physics"
    "mmlu_econometrics"
    "mmlu_electrical_engineering"
    "mmlu_elementary_mathematics"
    "mmlu_formal_logic"
    "mmlu_global_facts"
    "mmlu_high_school_biology"
    "mmlu_high_school_chemistry"
    "mmlu_high_school_computer_science"
    "mmlu_high_school_european_history"
    "mmlu_high_school_geography"
    "mmlu_high_school_government_and_politics"
    "mmlu_high_school_macroeconomics"
    "mmlu_high_school_mathematics"
    "mmlu_high_school_microeconomics"
    "mmlu_high_school_physics"
    "mmlu_high_school_psychology"
    "mmlu_high_school_statistics"
    "mmlu_high_school_us_history"
    "mmlu_high_school_world_history"
    "mmlu_human_aging"
    "mmlu_human_sexuality"
    "mmlu_international_law"
    "mmlu_jurisprudence"
    "mmlu_logical_fallacies"
    "mmlu_machine_learning"
    "mmlu_management"
    "mmlu_marketing"
    "mmlu_medical_genetics"
    "mmlu_miscellaneous"
    "mmlu_moral_disputes"
    "mmlu_moral_scenarios"
    "mmlu_nutrition"
    "mmlu_philosophy"
    "mmlu_prehistory"
    "mmlu_professional_accounting"
    "mmlu_professional_law"
    "mmlu_professional_medicine"
    "mmlu_professional_psychology"
    "mmlu_public_relations"
    "mmlu_security_studies"
    "mmlu_sociology"
    "mmlu_us_foreign_policy"
    "mmlu_virology"
    "mmlu_world_religions"
)

# GPQA (all 3)
GPQA_TASKS=(
    "gpqa_diamond_n_shot"
    "gpqa_extended_n_shot"
    "gpqa_main_n_shot"
)

# MedMCQA
MEDICAL_TASKS=(
    "medmcqa"
)

# =============================================================================
# 0-SHOT TASKS
# =============================================================================

ZEROSHOT_TASKS=(
    "boolq"
    "rte"
    "hellaswag"
    "winogrande"
    "arc_easy"
    "arc_challenge"
    "openbookqa"
)

# =============================================================================
# FUNCTION: Run single benchmark
# =============================================================================
run_benchmark() {
    local task_name=$1
    local num_fewshot=$2
    
    local safe_name=$(echo "$task_name" | tr '-' '_')
    local input_file="${DATA_DIR}/${safe_name}_input.jsonl"
    local output_file="${DATA_DIR}/${safe_name}_sparse.jsonl"
    local result_file="${RESULTS_DIR}/${safe_name}_sparse_result.json"
    
    echo ""
    echo "=========================================="
    echo "Task: $task_name (${num_fewshot}-shot, SPARSE)"
    echo "=========================================="
    
    # Step 1: Generate task data
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
    
    # Step 2: Run sparse inference
    echo "[2/3] Running SPARSE inference..."
    ./run_benchmark_llama_8b.sh \
        --input "$input_file" \
        --output "$output_file" \
        --sparse 2>&1 | tee -a "${RESULTS_DIR}/${safe_name}_inference.log"
    
    # Count OOM skips from output file
    local oom_count=$(grep -c "CUDA out of memory" "${RESULTS_DIR}/${safe_name}_inference.log" 2>/dev/null || echo "0")
    if [ "$oom_count" -gt 0 ]; then
        echo "$task_name: $oom_count prompts skipped due to OOM" >> "$OOM_LOG"
    fi
    
    # Step 3: Evaluate
    echo "[3/3] Evaluating results..."
    python evaluate_task_result.py \
        --result-file "$output_file" \
        --task-name "$task_name" \
        --model-type llama \
        --num-fewshot "$num_fewshot" > "$result_file" 2>&1
    
    # Extract and display accuracy
    local acc=$(grep -o '"acc":[^,}]*' "$result_file" | head -1 || echo "N/A")
    echo "Result: $acc"
    echo "$task_name (sparse, ${num_fewshot}-shot): $acc" >> "${RESULTS_DIR}/summary.txt"
    
    return 0
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "=============================================="
echo "Llama 3.1 8B SPARSE Benchmark Suite"
echo "Config: 70% MLP sparsity, 0% Att sparsity"
echo "=============================================="
echo ""

# Initialize summary
echo "Llama 3.1 8B Sparse Benchmark Results" > "${RESULTS_DIR}/summary.txt"
echo "Config: 70% MLP, 0% Attention" >> "${RESULTS_DIR}/summary.txt"
echo "Date: $(date)" >> "${RESULTS_DIR}/summary.txt"
echo "==============================" >> "${RESULTS_DIR}/summary.txt"

# Run 5-shot tasks
echo "" >> "${RESULTS_DIR}/summary.txt"
echo "=== 5-SHOT TASKS ===" >> "${RESULTS_DIR}/summary.txt"

echo ""
echo "### Running MMLU (${#MMLU_TASKS[@]} tasks, 5-shot) ###"
for task in "${MMLU_TASKS[@]}"; do
    run_benchmark "$task" 5 || true
done

echo ""
echo "### Running GPQA (${#GPQA_TASKS[@]} tasks, 5-shot) ###"
for task in "${GPQA_TASKS[@]}"; do
    run_benchmark "$task" 5 || true
done

echo ""
echo "### Running Medical (${#MEDICAL_TASKS[@]} tasks, 5-shot) ###"
for task in "${MEDICAL_TASKS[@]}"; do
    run_benchmark "$task" 5 || true
done

# Run 0-shot tasks
echo "" >> "${RESULTS_DIR}/summary.txt"
echo "=== 0-SHOT TASKS ===" >> "${RESULTS_DIR}/summary.txt"

echo ""
echo "### Running Zero-shot Tasks (${#ZEROSHOT_TASKS[@]} tasks) ###"
for task in "${ZEROSHOT_TASKS[@]}"; do
    run_benchmark "$task" 0 || true
done

# Final summary
echo ""
echo "=============================================="
echo "Benchmarking Complete!"
echo "=============================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}/summary.txt"
echo "OOM log saved to: ${OOM_LOG}"
echo ""
cat "${RESULTS_DIR}/summary.txt"
echo ""
echo "OOM Skipped Prompts:"
cat "$OOM_LOG"
