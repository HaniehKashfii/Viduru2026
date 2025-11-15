# FaaSInfer: Serverless LLM Inference Profiling

FaaSInfer extends Viduru's profiling methodology to serverless/FaaS-based LLM inference systems. It adds profiling for serverless-specific components while reusing Viduru's compute and network profiling infrastructure.

## Overview

Serverless LLM inference differs from traditional serving in several key aspects:

1. **Cold Starts**: Instances start from scratch, requiring model checkpoint loading
2. **Multi-tier Storage**: Models stored in blob storage, local SSDs, or NVMe caches
3. **Dynamic Scaling**: Instances spin up/down based on demand
4. **Pay-per-use**: Billing based on actual execution time, not reserved capacity

FaaSInfer profiles all these aspects to enable accurate simulation of serverless LLM serving.

## Architecture

```
FaaSInfer Profiling
├── Model Loading (Cold Start)
│   ├── Checkpoint loading from storage tiers
│   ├── Model-to-GPU transfer time
│   └── Incremental/streaming loading
│
├── Storage I/O
│   ├── Sequential read performance
│   ├── Random read performance
│   └── Checkpoint-specific I/O
│
├── Scaling Operations
│   ├── Instance startup time
│   ├── Instance shutdown time
│   ├── Parallel scale-up
│   └── Parallel scale-down
│
├── Compute (Reuses Viduru)
│   ├── MLP profiling
│   └── Attention profiling (prefill & decode)
│
└── Network (Reuses Viduru)
    ├── AllReduce (for TP)
    └── Send-Recv (for PP)
```

## Profiling Components

### 1. Model Loading Profiling

Profiles the time to load LLM checkpoints from different storage tiers (cold start overhead).

**What it measures:**
- `load_model_config`: Time to load model configuration
- `load_model_weights`: Time to load model weights from storage
- `model_to_gpu`: Time to transfer model to GPU memory
- `storage_tier_latency`: Storage access latency
- `incremental_load`: Incremental/streaming loading time

**Usage:**
```bash
python -m vidur.profiling.faasinfer.model_loading.main \
    --models meta-llama/Meta-Llama-3-8B \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --num_gpus 4 \
    --num_tensor_parallel_workers 1 2 4
```

**Output:** `data/profiling/faasinfer/model_loading/loading_{timestamp}.csv`

### 2. Storage I/O Profiling

Profiles raw I/O performance from different storage tiers.

**What it measures:**
- `sequential_read`: Sequential read throughput
- `random_read`: Random read performance
- `load_checkpoint`: PyTorch checkpoint loading time
- `checkpoint_to_gpu`: Checkpoint-to-GPU transfer time
- `observed_bandwidth_mbps`: Measured storage bandwidth

**Usage:**
```bash
python -m vidur.profiling.faasinfer.storage_io.main \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --file_sizes_mb 100 500 1000 5000 \
    --profile_random_io
```

**Output:** `data/profiling/faasinfer/storage_io/io_profile_{timestamp}.csv`

### 3. Scaling Operations Profiling

Profiles autoscaling operations (instance lifecycle management).

**What it measures:**
- `instance_startup`: Single instance startup time
- `instance_shutdown`: Single instance shutdown time
- `container_init`: Container initialization overhead
- `gpu_allocation`: GPU resource allocation time
- `cuda_context_init`: CUDA context initialization
- `scale_up_{N}_instances`: Time to spin up N instances in parallel
- `scale_down_{N}_instances`: Time to shut down N instances in parallel

**Usage:**
```bash
python -m vidur.profiling.faasinfer.scaling.main \
    --models meta-llama/Meta-Llama-3-8B \
    --num_instances 1 2 4 8 \
    --num_gpus 8
```

**Output:** `data/profiling/faasinfer/scaling/scaling_{timestamp}.csv`

### 4. Compute Profiling (Reuses Viduru)

Uses Viduru's existing MLP and Attention profiling infrastructure.

**What it measures:**
- MLP operations (up_proj, down_proj, activation)
- Attention operations (prefill, decode, RoPE, KV cache)
- Normalization and residual operations

See [Viduru profiling docs](../../README.md) for details.

### 5. Network Profiling (Reuses Viduru)

Uses Viduru's existing collective operations profiling.

**What it measures:**
- AllReduce (for Tensor Parallelism)
- Send-Recv (for Pipeline Parallelism)

See [Viduru profiling docs](../../README.md) for details.

## Quick Start: Complete Workflow

### Option 1: Automated End-to-End Workflow

Use the complete workflow script that profiles all components AND generates sim_params.json:

```bash
bash examples/faasinfer_complete_workflow.sh
```

This will:
1. Profile all components (model loading, storage I/O, scaling, compute, network)
2. Generate `sim_params.json` for each tensor parallel configuration
3. Validate the generated parameters
4. Create a summary report

### Option 2: Manual Step-by-Step

**Step 1: Profile All Components**

```bash
python -m vidur.profiling.faasinfer.profile_all \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 4 \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --num_tensor_parallel_workers 1 2 4 \
    --all
```

This will run:
1. Model loading profiling (cold start)
2. Storage I/O profiling
3. Scaling operations profiling
4. Compute profiling (MLP + Attention via Viduru)
5. Network profiling (AllReduce + Send-Recv via Viduru)

**Step 2: Generate sim_params.json**

```bash
python -m vidur.profiling.faasinfer.generate_sim_params \
    --model meta-llama/Meta-Llama-3-8B \
    --device a100 \
    --tensor_parallel_size 1 \
    --output sim_params.json \
    --pretty
```

This generates the six high-fidelity parameters required by SimFaaSInfer:
- `gpu_memory_mb` - GPU memory consumed by model
- `prefill_time_per_token_ms` - Time per prompt token (TTFT)
- `decode_time_per_token_ms` - Time per output token (TBT)
- `load_time_dram_ms` - Cold start from DRAM cache
- `load_time_ssd_ms` - Cold start from SSD cache
- `load_time_remote_ms` - Cold start from remote storage

## Storage Tiers

FaaSInfer supports profiling multiple storage tiers:

| Storage Tier | Typical Bandwidth | Typical Latency | Use Case |
|--------------|------------------|-----------------|----------|
| `remote_blob` | ~1.25 GB/s (10 Gbps) | 50 ms | S3, Azure Blob, GCS |
| `local_ssd` | ~3.5 GB/s | 1 ms | Local SSD storage |
| `nvme_cache` | ~7 GB/s | 0.1 ms | Fast NVMe cache |
| `gpu_memory` | N/A | N/A | Already loaded (hot start) |

Configure storage tiers in `vidur/profiling/faasinfer/common/storage_config.py`.

## Execution Time Predictor

FaaSInfer includes a specialized execution time predictor that extends Viduru's predictor with serverless-specific metrics:

```python
from vidur.execution_time_predictor.faasinfer_execution_time_predictor import (
    FaaSInferExecutionTimePredictor
)

# Initialize predictor
faasinfer_config = {
    "storage_tier": "local_ssd",
    "enable_cold_start": True,
    "cold_start_probability": 0.1,  # 10% of requests hit cold start
    "idle_timeout_seconds": 300,    # 5 min keep-alive
}

predictor = FaaSInferExecutionTimePredictor(
    predictor_config=predictor_config,
    replica_config=replica_config,
    cache_config=cache_config,
    faasinfer_config=faasinfer_config,
)

# Get execution time (includes cold start overhead)
execution_time = predictor.get_batch_execution_time(batch, pipeline_stage=0)

# Get serverless-specific metrics
metrics = predictor.get_serverless_metrics()
print(f"Cold start time: {metrics['cold_start_time_ms']} ms")
print(f"Storage bandwidth: {metrics['storage_bandwidth_mbps']} MB/s")
```

## Profiled Metrics Summary

### Cold Start Metrics
- **Total Loading Time**: End-to-end model loading time (ms)
- **Checkpoint Loading**: Time to load weights from storage (ms)
- **Model-to-GPU**: Time to transfer model to GPU (ms)
- **Storage Latency**: Storage access latency (ms)

### Storage I/O Metrics
- **Bandwidth**: Observed throughput (MB/s)
- **Sequential Read Time**: Time for sequential reads (ms)
- **Random Read Time**: Time for random reads (ms)
- **Checkpoint Load Time**: PyTorch checkpoint loading (ms)

### Scaling Metrics
- **Instance Startup**: Time to start a single instance (ms)
- **Instance Shutdown**: Time to shutdown a single instance (ms)
- **Container Init**: Container initialization overhead (ms)
- **GPU Allocation**: GPU resource allocation time (ms)
- **Scale-Up Latency**: Time to add N instances (ms)
- **Scale-Down Latency**: Time to remove N instances (ms)

### Compute Metrics (from Viduru)
- **MLP Operations**: up_proj, down_proj, activation (ms per token)
- **Attention Operations**: prefill, decode, RoPE (ms per token/batch)
- **Normalization**: layer_norm, rms_norm (ms per token)

### Network Metrics (from Viduru)
- **AllReduce**: Tensor parallel communication (ms per size)
- **Send-Recv**: Pipeline parallel communication (ms per size)

## Integration with Viduru Simulator

To use FaaSInfer profiling data in simulation:

1. **Profile all components** using the unified script
2. **Configure the simulator** to use FaaSInfer predictor:

```python
# In simulator config
execution_time_predictor_type = "faasinfer"
faasinfer_config = {
    "storage_tier": "local_ssd",
    "enable_cold_start": True,
    "cold_start_probability": 0.15,
}
```

3. **Run simulation** with serverless-aware metrics

The simulator will now account for:
- Cold start overhead on instance creation
- Storage I/O costs for model loading
- Scaling latencies for autoscaling operations
- Standard compute and network costs

## Output Format

All profiling results are saved as CSV files with the following structure:

### Model Loading CSV
```csv
model,storage_tier,num_tensor_parallel_workers,time_stats.load_model_weights.mean,...
meta-llama/Meta-Llama-3-8B,local_ssd,1,2345.67,...
```

### Storage I/O CSV
```csv
storage_tier,file_size_mb,sequential.observed_bandwidth_mbps,...
local_ssd,1000,3512.45,...
```

### Scaling CSV
```csv
model,startup.instance_startup.mean,shutdown.instance_shutdown.mean,...
meta-llama/Meta-Llama-3-8B,123.45,67.89,...
```

## Advanced Configuration

### Custom Storage Tiers

Add custom storage configurations in `storage_config.py`:

```python
StorageConfig(
    tier=StorageTier.CUSTOM,
    path="/custom/storage/path",
    bandwidth_mbps=5000,  # Custom bandwidth
    latency_ms=2,         # Custom latency
)
```

### Profiling Parameters

Adjust profiling granularity:

```bash
--warmup_iterations 2          # More warmup for stability
--measurement_iterations 10    # More measurements for accuracy
--chunk_size_mb 128           # Larger chunks for incremental loading
```

## Troubleshooting

### Out of Memory Errors
- Reduce `--file_sizes_mb` for storage I/O profiling
- Reduce `--num_instances` for scaling profiling
- Use smaller models for initial testing

### Ray Initialization Errors
- Ensure Ray is installed: `pip install ray`
- Check GPU availability: `nvidia-smi`
- Adjust `--num_gpus` to match available GPUs

### Profiling Data Not Found
- Ensure profiling scripts have completed successfully
- Check output directories: `data/profiling/faasinfer/`
- Use `--all` flag to profile all components

## Output: sim_params.json

The final output is a `sim_params.json` file containing six parameters for SimFaaSInfer:

```json
{
  "model_name": "meta-llama/Meta-Llama-3-8B",
  "device": "a100",
  "tensor_parallel_size": 1,
  "gpu_memory_mb": 15360.00,
  "prefill_time_per_token_ms": 0.0842,
  "decode_time_per_token_ms": 0.1634,
  "load_time_dram_ms": 1234.56,
  "load_time_ssd_ms": 2345.67,
  "load_time_remote_ms": 8901.23,
  "metadata": {
    "num_layers": 32,
    "embedding_dim": 4096,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "vocab_size": 128256,
    "estimated_parameters": 8030000000
  }
}
```

See `docs/faasinfer_metrics_mapping.md` for complete details on how each parameter is derived from profiling data.

## Citation

If you use FaaSInfer profiling in your research, please cite:

```bibtex
@inproceedings{viduru2024,
  title={Viduru: High-Fidelity LLM Inference System Simulator},
  booktitle={MLSys},
  year={2024}
}
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional storage tier implementations
- Model migration profiling
- Keep-alive/billing optimization
- Multi-region profiling support

## License

Same as Viduru (Microsoft Open Source Code of Conduct)
