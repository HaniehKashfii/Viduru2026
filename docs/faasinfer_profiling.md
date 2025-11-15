# FaaSInfer Profiling Methodology

## Overview

FaaSInfer extends Viduru's profiling methodology to serverless (Function-as-a-Service) LLM inference systems. While Viduru profiles traditional "always-on" GPU serving, FaaSInfer adds profiling for serverless-specific characteristics:

- **Cold Start Overhead**: Time to load models from storage when scaling from zero
- **Storage I/O Performance**: Multi-tier storage (blob storage, SSD, NVMe cache)
- **Autoscaling Latency**: Instance spin-up/spin-down timing
- **Pay-per-use Costs**: Billing granularity and keep-alive optimization

## Profiling Methodology

FaaSInfer follows the exact same profiling philosophy as Viduru:

1. **One-time GPU Profiling** - Run actual workloads on real hardware
2. **Machine Learning Training** - Train ML models on profiled data
3. **CPU-only Simulation** - Use trained models to predict execution times

The key difference is the **addition of serverless-specific profiling components**:

### Profiling Components

```
┌─────────────────────────────────────────────────────────────┐
│                  FaaSInfer Profiling                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NEW: Serverless-Specific Components                        │
│  ├── Model Loading (Cold Start)                            │
│  ├── Storage I/O                                           │
│  └── Scaling Operations                                    │
│                                                             │
│  REUSED: From Viduru                                        │
│  ├── Compute (MLP, Attention)                              │
│  └── Network (AllReduce, Send-Recv)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## New Profiling Components

### 1. Model Loading (Cold Start) Profiling

**Location**: `vidur/profiling/faasinfer/model_loading/`

**What it profiles**: Time to load model checkpoints from storage tiers

**Algorithm**:
1. Load model configuration from HuggingFace
2. Simulate storage tier latency
3. Load model weights (with torch.load or equivalent)
4. Transfer model to GPU memory
5. Apply tensor parallelism if needed
6. Measure each step with CUDA timers

**Parameters Varied**:
- `model_name`: Different LLM models
- `storage_tier`: remote_blob, local_ssd, nvme_cache
- `num_tensor_parallel_workers`: 1, 2, 4, 8

**Metrics Collected**:
- `load_model_config`: Configuration loading time
- `load_model_weights`: Weight loading time from storage
- `model_to_gpu`: GPU transfer time
- `storage_tier_latency`: Storage access latency
- `apply_tensor_parallel`: TP setup time

**Output**: `data/profiling/faasinfer/model_loading/loading_{timestamp}.csv`

### 2. Storage I/O Profiling

**Location**: `vidur/profiling/faasinfer/storage_io/`

**What it profiles**: Raw I/O performance from different storage tiers

**Algorithm**:
1. Create test files of various sizes
2. Measure sequential read performance
3. Measure random read performance (optional)
4. Create and load PyTorch checkpoints
5. Measure checkpoint-to-GPU transfer
6. Calculate observed bandwidth

**Parameters Varied**:
- `storage_tier`: Different storage types
- `file_size_mb`: 100, 500, 1000, 2000, 5000 MB
- `block_size_kb`: For random I/O (default 4KB)

**Metrics Collected**:
- `sequential_read`: Sequential read time
- `random_read`: Random read time (if enabled)
- `load_checkpoint`: PyTorch checkpoint loading
- `checkpoint_to_gpu`: GPU transfer time
- `observed_bandwidth_mbps`: Measured throughput

**Output**: `data/profiling/faasinfer/storage_io/io_profile_{timestamp}.csv`

### 3. Scaling Operations Profiling

**Location**: `vidur/profiling/faasinfer/scaling/`

**What it profiles**: Autoscaling operations (instance lifecycle)

**Algorithm**:
1. Start single instance (container init + GPU allocation)
2. Measure instance startup components
3. Shutdown instance and measure cleanup time
4. Start multiple instances in parallel (Ray actors)
5. Measure parallel scale-up latency
6. Shutdown instances in parallel
7. Measure parallel scale-down latency

**Parameters Varied**:
- `model_name`: Different models
- `num_instances`: 1, 2, 4, 8 (for parallel scaling)
- `num_tensor_parallel_workers`: TP configuration

**Metrics Collected**:
- `instance_startup`: Total startup time
- `container_init`: Container initialization
- `gpu_allocation`: GPU resource allocation
- `cuda_context_init`: CUDA initialization
- `instance_shutdown`: Shutdown time
- `scale_up_N_instances`: Parallel scale-up for N instances
- `scale_down_N_instances`: Parallel scale-down

**Output**: `data/profiling/faasinfer/scaling/scaling_{timestamp}.csv`

## Reused Viduru Components

FaaSInfer reuses Viduru's existing profiling for compute and network:

### 4. Compute Profiling (Viduru)

**Uses**: `vidur/profiling/mlp/main.py` and `vidur/profiling/attention/main.py`

**What it profiles**: GPU kernel execution times

See [Viduru profiling documentation](./profiling.md) for details.

### 5. Network Profiling (Viduru)

**Uses**: `vidur/profiling/collectives/main.py`

**What it profiles**: Collective communication operations

See [Viduru profiling documentation](./profiling.md) for details.

## Profiling Workflow

### Step 1: Profile Model Loading (Cold Start)

```bash
python -m vidur.profiling.faasinfer.model_loading.main \
    --models meta-llama/Meta-Llama-3-8B \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --num_gpus 4 \
    --num_tensor_parallel_workers 1 2 4 \
    --profile_incremental_loading  # Optional: streaming load
```

**Expected Output**:
```
FaaSInfer Model Loading Profiling
Models: ['meta-llama/Meta-Llama-3-8B']
Storage Tiers: ['remote_blob', 'local_ssd', 'nvme_cache']
Tensor Parallel Workers: [1, 2, 4]
Profile Method: perf_counter
--------------------------------------------------------------------------------
Profiling: 100%|████████████| 9/9 [05:23<00:00, 35.9s/it]

Results saved to: data/profiling/faasinfer/model_loading/loading_2025-11-15_12-34-56.csv
Total configurations profiled: 9

SUMMARY: Cold Start Times (mean, in seconds)
model                          storage_tier
meta-llama/Meta-Llama-3-8B     local_ssd      2.345
                               nvme_cache     1.234
                               remote_blob    8.901
```

### Step 2: Profile Storage I/O

```bash
python -m vidur.profiling.faasinfer.storage_io.main \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --file_sizes_mb 100 500 1000 5000 \
    --profile_random_io
```

### Step 3: Profile Scaling Operations

```bash
python -m vidur.profiling.faasinfer.scaling.main \
    --models meta-llama/Meta-Llama-3-8B \
    --num_instances 1 2 4 8 \
    --num_gpus 8
```

### Step 4: Profile Compute (Viduru)

```bash
# MLP Profiling
python -m vidur.profiling.mlp.main \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 4 \
    --max_tokens 4096

# Attention Profiling
python -m vidur.profiling.attention.main \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 4 \
    --max_seq_len 16384
```

### Step 5: Profile Network (Viduru)

```bash
python -m vidur.profiling.collectives.main \
    --num_workers_per_node_combinations 1,2,4,8 \
    --collective all_reduce
```

### All-in-One: Profile Everything

```bash
python -m vidur.profiling.faasinfer.profile_all \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 4 \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --num_tensor_parallel_workers 1 2 4 \
    --all
```

## Execution Time Prediction

After profiling, FaaSInfer uses ML models to predict execution times:

### Predictor Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           FaaSInfer Execution Time Predictor                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Base Viduru Predictor                                      │
│  ├── Compute Time (MLP, Attention)                         │
│  └── Network Time (AllReduce, Send-Recv)                   │
│                                                             │
│  + FaaSInfer Extensions                                     │
│  ├── Cold Start Time (from model loading profiling)        │
│  ├── Storage I/O Time (from storage profiling)             │
│  └── Scaling Time (from scaling profiling)                 │
│                                                             │
│  = Total Execution Time                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Usage Example

```python
from vidur.execution_time_predictor.faasinfer_execution_time_predictor import (
    FaaSInferExecutionTimePredictor
)

# Configure FaaSInfer
faasinfer_config = {
    "storage_tier": "local_ssd",              # Storage tier to use
    "enable_cold_start": True,                # Enable cold start simulation
    "cold_start_probability": 0.1,            # 10% requests hit cold start
    "idle_timeout_seconds": 300,              # 5 min keep-alive
}

# Create predictor
predictor = FaaSInferExecutionTimePredictor(
    predictor_config=predictor_config,
    replica_config=replica_config,
    cache_config=cache_config,
    faasinfer_config=faasinfer_config,
)

# Get execution time (includes all overheads)
execution_time = predictor.get_batch_execution_time(batch, pipeline_stage=0)

# Get serverless metrics
metrics = predictor.get_serverless_metrics()
print(f"Cold start: {metrics['cold_start_time_ms']:.2f} ms")
print(f"Storage: {metrics['storage_bandwidth_mbps']:.2f} MB/s")
print(f"Startup: {metrics['instance_startup_time_ms']:.2f} ms")
```

## Key Metrics

### Serverless-Specific Metrics

| Metric | Description | Unit | Typical Range |
|--------|-------------|------|---------------|
| Cold Start Time | Model loading from storage | ms | 1,000 - 10,000 |
| Storage Bandwidth | I/O throughput | MB/s | 1,000 - 7,000 |
| Storage Latency | Access latency | ms | 0.1 - 50 |
| Instance Startup | Container + GPU init | ms | 50 - 500 |
| Instance Shutdown | Cleanup time | ms | 10 - 100 |
| Scale-Up Latency | Add N instances | ms | 100 - 2,000 |

### Storage Tier Characteristics

| Tier | Bandwidth | Latency | Use Case |
|------|-----------|---------|----------|
| Remote Blob (S3/Azure) | ~1.25 GB/s | ~50 ms | Cold storage |
| Local SSD | ~3.5 GB/s | ~1 ms | Warm cache |
| NVMe Cache | ~7 GB/s | ~0.1 ms | Hot cache |
| GPU Memory | N/A | N/A | Already loaded |

## Comparison: FaaSInfer vs. Viduru Profiling

| Aspect | Viduru | FaaSInfer |
|--------|--------|-----------|
| **Target System** | Always-on GPU serving | Serverless/FaaS inference |
| **Compute Profiling** | ✅ MLP, Attention | ✅ Same (reused) |
| **Network Profiling** | ✅ AllReduce, Send-Recv | ✅ Same (reused) |
| **Cold Start** | ❌ N/A | ✅ Model loading from storage |
| **Storage I/O** | ❌ N/A | ✅ Multi-tier profiling |
| **Scaling** | ❌ N/A | ✅ Instance lifecycle |
| **Billing Model** | Reserved capacity | Pay-per-use |
| **Use Case** | Dedicated GPU clusters | Serverless platforms |

## Best Practices

1. **Profile on target hardware**: Run profiling on the actual GPUs you'll use
2. **Multiple iterations**: Use sufficient warmup and measurement iterations
3. **Storage diversity**: Profile all storage tiers you might use
4. **Model variety**: Profile different model sizes for better coverage
5. **TP configurations**: Profile all tensor parallel configurations
6. **Incremental loading**: Enable if you plan to use streaming model loading

## Troubleshooting

### Common Issues

**Out of Memory**:
```bash
# Reduce file sizes for storage I/O profiling
--file_sizes_mb 100 500 1000  # Instead of 5000

# Or reduce model size
--models meta-llama/Llama-2-7b-hf  # Smaller model
```

**Ray Connection Issues**:
```bash
# Ensure Ray is properly initialized
ray start --head  # On head node

# Or disable Ray for single-GPU profiling
--disable_ray
```

**Missing Profiling Data**:
```bash
# Check output directories
ls -la data/profiling/faasinfer/*/

# Re-run specific component
python -m vidur.profiling.faasinfer.model_loading.main ...
```

## Future Extensions

Potential areas for enhancement:

1. **Model Migration**: Live model migration between instances
2. **Multi-Region**: Cross-region model distribution
3. **Checkpoint Formats**: SafeTensors, GGUF, custom formats
4. **Compression**: On-the-fly decompression profiling
5. **Keep-Alive Optimization**: Smart idle timeout selection
6. **Cost Modeling**: Detailed billing simulation

## References

- [Viduru Paper (MLSys'24)](https://arxiv.org/abs/2405.05465)
- [ServerlessLLM (OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu)
- [FaaSInfer README](../vidur/profiling/faasinfer/README.md)
