#!/bin/bash
# FaaSInfer Quick Start Script
#
# This script demonstrates how to profile a serverless LLM inference system
# using FaaSInfer's profiling methodology.
#
# Usage: bash examples/faasinfer_quick_start.sh

set -e  # Exit on error

echo "========================================="
echo "FaaSInfer Profiling Quick Start"
echo "========================================="
echo ""

# Configuration
MODEL="meta-llama/Meta-Llama-3-8B"
NUM_GPUS=4
STORAGE_TIERS="local_ssd nvme_cache"  # Start with local tiers
TP_WORKERS="1 2 4"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  GPUs: $NUM_GPUS"
echo "  Storage Tiers: $STORAGE_TIERS"
echo "  Tensor Parallel: $TP_WORKERS"
echo ""

# Option 1: Profile everything (recommended for first run)
echo "========================================="
echo "Option 1: Profile All Components"
echo "========================================="
echo ""
echo "This will run all FaaSInfer profiling components:"
echo "  1. Model Loading (Cold Start)"
echo "  2. Storage I/O"
echo "  3. Scaling Operations"
echo "  4. Compute (MLP + Attention)"
echo "  5. Network (AllReduce)"
echo ""
echo "Estimated time: 15-30 minutes"
echo ""

read -p "Run full profiling? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting full profiling..."
    python -m vidur.profiling.faasinfer.profile_all \
        --models $MODEL \
        --num_gpus $NUM_GPUS \
        --storage_tiers $STORAGE_TIERS \
        --num_tensor_parallel_workers $TP_WORKERS \
        --all

    echo ""
    echo "✓ Full profiling completed!"
    echo "Results saved to: data/profiling/faasinfer/"
fi

# Option 2: Profile individual components
echo ""
echo "========================================="
echo "Option 2: Profile Individual Components"
echo "========================================="
echo ""

read -p "Profile Model Loading (Cold Start)? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Profiling Model Loading..."
    python -m vidur.profiling.faasinfer.model_loading.main \
        --models $MODEL \
        --storage_tiers $STORAGE_TIERS \
        --num_gpus $NUM_GPUS \
        --num_tensor_parallel_workers $TP_WORKERS
    echo "✓ Model loading profiling completed!"
fi

read -p "Profile Storage I/O? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Profiling Storage I/O..."
    python -m vidur.profiling.faasinfer.storage_io.main \
        --storage_tiers $STORAGE_TIERS \
        --file_sizes_mb 100 500 1000
    echo "✓ Storage I/O profiling completed!"
fi

read -p "Profile Scaling Operations? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Profiling Scaling Operations..."
    python -m vidur.profiling.faasinfer.scaling.main \
        --models $MODEL \
        --num_instances 1 2 4 \
        --num_gpus $NUM_GPUS
    echo "✓ Scaling operations profiling completed!"
fi

echo ""
echo "========================================="
echo "Profiling Complete!"
echo "========================================="
echo ""
echo "Profiling data saved to:"
echo "  - Model Loading: data/profiling/faasinfer/model_loading/"
echo "  - Storage I/O:   data/profiling/faasinfer/storage_io/"
echo "  - Scaling:       data/profiling/faasinfer/scaling/"
echo "  - Compute:       data/profiling/compute/"
echo "  - Network:       data/profiling/network/"
echo ""
echo "Next steps:"
echo "  1. Review profiling results in CSV files"
echo "  2. Use FaaSInferExecutionTimePredictor in simulations"
echo "  3. See docs/faasinfer_profiling.md for details"
echo ""
