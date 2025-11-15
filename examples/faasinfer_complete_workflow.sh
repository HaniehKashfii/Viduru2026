#!/bin/bash
# FaaSInfer Complete Workflow Example
#
# This script demonstrates the complete end-to-end workflow for profiling
# a serverless LLM inference system and generating sim_params.json for
# capacity planning with SimFaaSInfer.
#
# Steps:
# 1. Profile all components (model loading, storage I/O, scaling, compute, network)
# 2. Generate sim_params.json from profiling data
# 3. Validate the generated parameters
#
# Usage: bash examples/faasinfer_complete_workflow.sh

set -e  # Exit on error

echo "================================================================================"
echo "FaaSInfer Complete Workflow: Profiling to sim_params.json"
echo "================================================================================"
echo ""

# ============================================================================
# Configuration
# ============================================================================

MODEL="meta-llama/Meta-Llama-3-8B"
DEVICE="a100"  # or h100, a40, etc.
NUM_GPUS=4
STORAGE_TIERS="remote_blob local_ssd nvme_cache"
TP_SIZES="1 2 4"  # Tensor parallel sizes to profile

PROFILING_DIR="data/profiling"
OUTPUT_DIR="sim_params_output"

echo "Configuration:"
echo "  Model:               $MODEL"
echo "  Device:              $DEVICE"
echo "  GPUs Available:      $NUM_GPUS"
echo "  Storage Tiers:       $STORAGE_TIERS"
echo "  Tensor Parallel:     $TP_SIZES"
echo "  Profiling Dir:       $PROFILING_DIR"
echo "  Output Dir:          $OUTPUT_DIR"
echo ""
echo "================================================================================"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# ============================================================================
# Step 1: Profile All Components
# ============================================================================

echo ""
echo "================================================================================"
echo "STEP 1: Profile All Components"
echo "================================================================================"
echo ""
echo "This will profile:"
echo "  ✓ Model Loading (Cold Start)"
echo "  ✓ Storage I/O"
echo "  ✓ Scaling Operations"
echo "  ✓ Compute (MLP + Attention) - via Viduru"
echo "  ✓ Network (AllReduce) - via Viduru"
echo ""
echo "Estimated time: 20-40 minutes"
echo ""

read -p "Start profiling? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Skipping profiling. Using existing data..."
else
    echo ""
    echo "Starting comprehensive profiling..."
    echo ""

    python -m vidur.profiling.faasinfer.profile_all \
        --models $MODEL \
        --num_gpus $NUM_GPUS \
        --storage_tiers $STORAGE_TIERS \
        --num_tensor_parallel_workers $TP_SIZES \
        --max_tokens 4096 \
        --max_seq_len 16384 \
        --all

    echo ""
    echo "✓ Profiling completed successfully!"
    echo ""
fi

# ============================================================================
# Step 2: Generate sim_params.json for Each Configuration
# ============================================================================

echo ""
echo "================================================================================"
echo "STEP 2: Generate sim_params.json Files"
echo "================================================================================"
echo ""
echo "Generating simulation parameters for each tensor parallel configuration..."
echo ""

for TP in $TP_SIZES; do
    OUTPUT_FILE="$OUTPUT_DIR/sim_params_${MODEL##*/}_${DEVICE}_tp${TP}.json"

    echo "--------------------------------------------------------------------------------"
    echo "Configuration: ${MODEL##*/} on $DEVICE with TP=$TP"
    echo "--------------------------------------------------------------------------------"

    python -m vidur.profiling.faasinfer.generate_sim_params \
        --model "$MODEL" \
        --device "$DEVICE" \
        --tensor_parallel_size $TP \
        --profiling_data_dir "$PROFILING_DIR" \
        --output "$OUTPUT_FILE" \
        --pretty

    echo ""
    echo "✓ Generated: $OUTPUT_FILE"
    echo ""
done

echo ""
echo "================================================================================"
echo "STEP 2 COMPLETED: Generated sim_params.json for all configurations"
echo "================================================================================"
echo ""

# ============================================================================
# Step 3: Display and Validate Results
# ============================================================================

echo ""
echo "================================================================================"
echo "STEP 3: Validation and Summary"
echo "================================================================================"
echo ""

for TP in $TP_SIZES; do
    OUTPUT_FILE="$OUTPUT_DIR/sim_params_${MODEL##*/}_${DEVICE}_tp${TP}.json"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "--------------------------------------------------------------------------------"
        echo "Configuration: ${MODEL##*/} / $DEVICE / TP=$TP"
        echo "--------------------------------------------------------------------------------"

        # Extract and display key parameters using jq (if available)
        if command -v jq &> /dev/null; then
            GPU_MEM=$(jq -r '.gpu_memory_mb' "$OUTPUT_FILE")
            PREFILL=$(jq -r '.prefill_time_per_token_ms' "$OUTPUT_FILE")
            DECODE=$(jq -r '.decode_time_per_token_ms' "$OUTPUT_FILE")
            DRAM=$(jq -r '.load_time_dram_ms' "$OUTPUT_FILE")
            SSD=$(jq -r '.load_time_ssd_ms' "$OUTPUT_FILE")
            REMOTE=$(jq -r '.load_time_remote_ms' "$OUTPUT_FILE")

            echo "  GPU Memory:              $GPU_MEM MB"
            echo "  Prefill Time/Token:      $PREFILL ms"
            echo "  Decode Time/Token:       $DECODE ms"
            echo "  Load Time (DRAM):        $DRAM ms"
            echo "  Load Time (SSD):         $SSD ms"
            echo "  Load Time (Remote):      $REMOTE ms"
            echo ""

            # Validation checks
            echo "  Validation:"

            # Check 1: Prefill < Decode (usually true)
            if (( $(echo "$PREFILL < $DECODE" | bc -l) )); then
                echo "    ✓ Prefill time < Decode time"
            else
                echo "    ⚠ Warning: Prefill time >= Decode time (unusual)"
            fi

            # Check 2: DRAM < SSD < Remote
            if (( $(echo "$DRAM < $SSD && $SSD < $REMOTE" | bc -l) )); then
                echo "    ✓ Load times: DRAM < SSD < Remote"
            else
                echo "    ⚠ Warning: Load time ordering unexpected"
            fi

            # Check 3: Reasonable GPU memory
            if (( $(echo "$GPU_MEM > 1000 && $GPU_MEM < 100000" | bc -l) )); then
                echo "    ✓ GPU memory in reasonable range"
            else
                echo "    ⚠ Warning: GPU memory seems unusual"
            fi

        else
            echo "  (Install 'jq' for detailed validation)"
            cat "$OUTPUT_FILE"
        fi

        echo ""
    fi
done

# ============================================================================
# Step 4: Generate Summary Report
# ============================================================================

echo ""
echo "================================================================================"
echo "SUMMARY REPORT"
echo "================================================================================"
echo ""

SUMMARY_FILE="$OUTPUT_DIR/profiling_summary.txt"

cat > "$SUMMARY_FILE" <<EOF
FaaSInfer Profiling Summary
================================================================================

Model:  $MODEL
Device: $DEVICE
Date:   $(date)

Generated sim_params.json files:
EOF

for TP in $TP_SIZES; do
    OUTPUT_FILE="$OUTPUT_DIR/sim_params_${MODEL##*/}_${DEVICE}_tp${TP}.json"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  - $OUTPUT_FILE" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" <<EOF

Profiling Data Locations:
  - Model Loading:  $PROFILING_DIR/faasinfer/model_loading/
  - Storage I/O:    $PROFILING_DIR/faasinfer/storage_io/
  - Scaling:        $PROFILING_DIR/faasinfer/scaling/
  - Compute (MLP):  $PROFILING_DIR/compute/$DEVICE/
  - Compute (Attn): $PROFILING_DIR/compute/$DEVICE/
  - Network:        $PROFILING_DIR/network/

Next Steps:
  1. Review generated sim_params.json files
  2. Validate parameters match expected ranges
  3. Use sim_params.json in SimFaaSInfer capacity planning tool

Example usage with SimFaaSInfer:
  sim-faasinfer --config $OUTPUT_DIR/sim_params_${MODEL##*/}_${DEVICE}_tp1.json \\
                --workload your_trace.csv \\
                --output capacity_plan.json

For more details, see:
  - docs/faasinfer_profiling.md
  - docs/faasinfer_metrics_mapping.md
  - vidur/profiling/faasinfer/README.md

================================================================================
EOF

cat "$SUMMARY_FILE"
echo ""
echo "Summary saved to: $SUMMARY_FILE"

# ============================================================================
# Completion
# ============================================================================

echo ""
echo "================================================================================"
echo "✓ WORKFLOW COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "All sim_params.json files have been generated and validated."
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (No files found)"
echo ""
echo "You can now use these files with SimFaaSInfer for capacity planning."
echo ""
echo "================================================================================"
echo ""
