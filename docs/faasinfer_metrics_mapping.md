# FaaSInfer: Complete Metrics Mapping

This document maps all profiled metrics to the final six simulation parameters required by SimFaaSInfer.

## Data Flow

```
Raw Profiling Data → ML Model Training → sim_params.json (6 parameters)
```

---

## I. Serverless-Specific Overheads (FaaSInfer Extensions)

### 1. Model Loading (Cold Start) Profiling

**Location**: `vidur/profiling/faasinfer/model_loading/main.py`

**Output**: `data/profiling/faasinfer/model_loading/loading_{timestamp}.csv`

| Metric Collected | Description | Used For |
|------------------|-------------|----------|
| `load_model_config` | Time to load model configuration from HuggingFace | Cold start overhead |
| `load_model_weights` | Time to load model weights from storage (torch.load) | **load_time_*_ms** parameters |
| `model_to_gpu` | Time to transfer model from CPU to GPU memory | **load_time_*_ms** parameters |
| `storage_tier_latency` | Raw latency of storage tier access | **load_time_*_ms** parameters |
| `apply_tensor_parallel` | Setup time for tensor parallelism | Cold start overhead |

**Profiling Command**:
```bash
python -m vidur.profiling.faasinfer.model_loading.main \
    --models meta-llama/Meta-Llama-3-8B \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --num_tensor_parallel_workers 1 2 4
```

**CSV Columns**:
```
model,storage_tier,num_tensor_parallel_workers,time_stats.load_model_config.mean,
time_stats.load_model_weights.mean,time_stats.model_to_gpu.mean,
time_stats.storage_tier_latency.mean,time_stats.apply_tensor_parallel.mean
```

---

### 2. Storage I/O Profiling

**Location**: `vidur/profiling/faasinfer/storage_io/main.py`

**Output**: `data/profiling/faasinfer/storage_io/io_profile_{timestamp}.csv`

| Metric Collected | Description | Used For |
|------------------|-------------|----------|
| `sequential_read` | Sequential read time for various file sizes | Bandwidth estimation |
| `random_read` | Random read time (optional, for block-level access) | Latency characterization |
| `load_checkpoint` | PyTorch checkpoint loading time | Model loading validation |
| `checkpoint_to_gpu` | Checkpoint-to-GPU transfer time | GPU transfer overhead |
| `observed_bandwidth_mbps` | Measured read/write throughput (MB/s) | Load time estimation |

**Profiling Command**:
```bash
python -m vidur.profiling.faasinfer.storage_io.main \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --file_sizes_mb 100 500 1000 5000
```

**CSV Columns**:
```
storage_tier,file_size_mb,theoretical_bandwidth_mbps,theoretical_latency_ms,
sequential.sequential_read.mean,sequential.observed_bandwidth_mbps,
checkpoint.load_checkpoint.mean,checkpoint.checkpoint_to_gpu.mean
```

---

### 3. Scaling Operations Profiling

**Location**: `vidur/profiling/faasinfer/scaling/main.py`

**Output**: `data/profiling/faasinfer/scaling/scaling_{timestamp}.csv`

| Metric Collected | Description | Used For |
|------------------|-------------|----------|
| `instance_startup` | Total instance startup time | Autoscaling latency |
| `container_init` | Container/Docker initialization time | Startup breakdown |
| `gpu_allocation` | GPU resource allocation time | Startup breakdown |
| `cuda_context_init` | CUDA context initialization time | Startup breakdown |
| `instance_shutdown` | Instance cleanup and termination time | Scaling latency |
| `scale_up_N_instances` | Parallel scale-up latency for N instances | Bulk scaling |
| `scale_down_N_instances` | Parallel scale-down latency for N instances | Bulk scaling |

**Profiling Command**:
```bash
python -m vidur.profiling.faasinfer.scaling.main \
    --models meta-llama/Meta-Llama-3-8B \
    --num_instances 1 2 4 8
```

**CSV Columns**:
```
model,num_tensor_parallel_workers,startup.instance_startup.mean,
startup.container_init.mean,startup.gpu_allocation.mean,
startup.cuda_context_init.mean,shutdown.instance_shutdown.mean,
scale_up_1.scale_up_1_instances.mean,...
```

**Note**: Scaling metrics are used for autoscaling simulation but don't directly contribute to the six core parameters. They're essential for simulating dynamic capacity.

---

## II. Core LLM Compute and Communication (Viduru Components)

### 4. Compute Profiling (MLP)

**Location**: `vidur/profiling/mlp/main.py` (Viduru)

**Output**: `data/profiling/compute/{device}/{model}/mlp.csv`

| Metric Collected | Description | Used For |
|------------------|-------------|----------|
| `mlp_up_proj` | Up-projection (linear expansion) time | **prefill_time_per_token_ms**, **decode_time_per_token_ms** |
| `mlp_down_proj` | Down-projection (linear contraction) time | **prefill_time_per_token_ms**, **decode_time_per_token_ms** |
| `mlp_act` | Activation function (SiLU/GELU) time | **prefill_time_per_token_ms**, **decode_time_per_token_ms** |
| `input_layernorm` | Pre-attention normalization time | Token processing time |
| `post_attention_layernorm` | Post-attention normalization time | Token processing time |
| `add` | Residual connection additions | Token processing time |

**Profiling Command**:
```bash
python -m vidur.profiling.mlp.main \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 4 \
    --max_tokens 4096 \
    --num_tensor_parallel_workers 1 2 4
```

**CSV Columns**:
```
num_tokens,num_tensor_parallel_workers,time_stats.mlp_up_proj.mean,
time_stats.mlp_down_proj.mean,time_stats.mlp_act.mean,
time_stats.input_layernorm.mean,time_stats.post_attention_layernorm.mean
```

---

### 5. Compute Profiling (Attention)

**Location**: `vidur/profiling/attention/main.py` (Viduru)

**Output**: `data/profiling/compute/{device}/{model}/attention.csv`

| Metric Collected | Description | Used For |
|------------------|-------------|----------|
| `attn_prefill` | Full sequence attention (all prompt tokens) | **prefill_time_per_token_ms** |
| `attn_decode` | Single-token attention (with KV cache) | **decode_time_per_token_ms** |
| `attn_rope` | RoPE position encoding time | Token processing time |
| `attn_kv_cache_save` | KV cache write time | Token processing time |
| `attn_pre_proj` | Q, K, V projections time | Token processing time |
| `attn_post_proj` | Output projection time | Token processing time |

**Profiling Command**:
```bash
python -m vidur.profiling.attention.main \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 4 \
    --max_seq_len 16384 \
    --max_batch_size 128
```

**CSV Columns**:
```
batch_size,prefill_chunk_size,kv_cache_size,is_prefill,
time_stats.attn_prefill.mean,time_stats.attn_decode.mean,
time_stats.attn_rope.mean,time_stats.attn_kv_cache_save.mean
```

---

### 6. Network Profiling (Collectives)

**Location**: `vidur/profiling/collectives/main.py` (Viduru)

**Output**: `data/profiling/network/{network_device}/allreduce.csv`

| Metric Collected | Description | Used For |
|------------------|-------------|----------|
| `all_reduce` | AllReduce operation time (Tensor Parallelism) | Communication overhead in TP |
| `send_recv` | Point-to-point communication (Pipeline Parallelism) | Communication overhead in PP |

**Profiling Command**:
```bash
python -m vidur.profiling.collectives.main \
    --num_workers_per_node_combinations 1,2,4,8 \
    --collective all_reduce
```

**CSV Columns**:
```
num_workers,size,time_stats.all_reduce.mean
```

**Note**: Network profiling contributes to overall token time when using tensor/pipeline parallelism, but is not directly a separate parameter in sim_params.json.

---

## III. Final Derived Metrics (sim_params.json)

### Parameter Generation

**Script**: `vidur/profiling/faasinfer/generate_sim_params.py`

**Command**:
```bash
python -m vidur.profiling.faasinfer.generate_sim_params \
    --model meta-llama/Meta-Llama-3-8B \
    --device a100 \
    --tensor_parallel_size 1 \
    --output sim_params.json
```

---

### 1. gpu_memory_mb

**Purpose**: Total GPU memory consumed by model weights

**Calculation**:
```python
num_params = estimate_from_model_config()
bytes_per_param = 2  # bfloat16/float16
total_memory = (num_params * bytes_per_param) / (1024^2)
memory_per_gpu = total_memory / tensor_parallel_size
memory_with_overhead = memory_per_gpu * 1.25  # 25% overhead for activations, KV cache
```

**Sources**:
- Model config (num_layers, embedding_dim, vocab_size)
- Tensor parallel size

**Example**:
```json
"gpu_memory_mb": 15360.00
```

---

### 2. prefill_time_per_token_ms

**Purpose**: Time to process each prompt token (TTFT estimation)

**Calculation**:
```python
# From MLP profiling
mlp_time = mlp_up_proj + mlp_down_proj + mlp_act

# From Attention profiling (prefill phase)
attn_time = attn_prefill

# Per-token time for one layer
time_per_layer = (mlp_time + attn_time) / num_tokens

# All layers
prefill_time_per_token = time_per_layer * num_layers
```

**Sources**:
- `data/profiling/compute/{device}/{model}/mlp.csv`
  - Columns: `time_stats.mlp_up_proj.mean`, `time_stats.mlp_down_proj.mean`, `time_stats.mlp_act.mean`
- `data/profiling/compute/{device}/{model}/attention.csv`
  - Column: `time_stats.attn_prefill.mean` (where `is_prefill=True`)

**Example**:
```json
"prefill_time_per_token_ms": 0.0842
```

**Typical Values**:
- Llama-3-8B on A100: ~0.08-0.12 ms/token
- Llama-3-70B on A100 (TP4): ~0.15-0.25 ms/token

---

### 3. decode_time_per_token_ms

**Purpose**: Time to generate each output token (TBT estimation)

**Calculation**:
```python
# From MLP profiling (single token)
mlp_time = mlp_up_proj + mlp_down_proj + mlp_act  # at num_tokens=1

# From Attention profiling (decode phase)
attn_time = attn_decode  # at batch_size=1, moderate kv_cache

# Per-token time for one layer
time_per_layer = mlp_time + attn_time

# All layers
decode_time_per_token = time_per_layer * num_layers
```

**Sources**:
- `data/profiling/compute/{device}/{model}/mlp.csv`
  - Filter: `num_tokens=1`
  - Columns: `time_stats.mlp_up_proj.mean`, `time_stats.mlp_down_proj.mean`, `time_stats.mlp_act.mean`
- `data/profiling/compute/{device}/{model}/attention.csv`
  - Filter: `is_prefill=False`, `batch_size=1`
  - Column: `time_stats.attn_decode.mean`

**Example**:
```json
"decode_time_per_token_ms": 0.1634
```

**Typical Values**:
- Llama-3-8B on A100: ~0.15-0.20 ms/token
- Llama-3-70B on A100 (TP4): ~0.30-0.50 ms/token

---

### 4. load_time_dram_ms

**Purpose**: Cold start time when loading from DRAM cache (fastest tier)

**Calculation**:
```python
# From model loading profiling for nvme_cache tier
load_time_dram = load_model_weights + model_to_gpu + storage_tier_latency
```

**Sources**:
- `data/profiling/faasinfer/model_loading/loading_{timestamp}.csv`
  - Filter: `storage_tier='nvme_cache'` (maps to DRAM)
  - Columns: `time_stats.load_model_weights.mean`, `time_stats.model_to_gpu.mean`, `time_stats.storage_tier_latency.mean`

**Fallback** (if no profiling data):
```python
model_size_mb = (num_params * 2) / (1024^2) / tensor_parallel_size
bandwidth_mbps = 7000  # ~7 GB/s for NVMe
latency_ms = 0.1
load_time_dram = latency + (model_size_mb / bandwidth_mbps) * 1000
```

**Example**:
```json
"load_time_dram_ms": 1234.56
```

**Typical Values**:
- Llama-3-8B: ~1-2 seconds
- Llama-3-70B (TP4): ~3-5 seconds

---

### 5. load_time_ssd_ms

**Purpose**: Cold start time when loading from SSD cache (medium tier)

**Calculation**:
```python
# From model loading profiling for local_ssd tier
load_time_ssd = load_model_weights + model_to_gpu + storage_tier_latency
```

**Sources**:
- `data/profiling/faasinfer/model_loading/loading_{timestamp}.csv`
  - Filter: `storage_tier='local_ssd'`
  - Columns: `time_stats.load_model_weights.mean`, `time_stats.model_to_gpu.mean`, `time_stats.storage_tier_latency.mean`

**Fallback**:
```python
model_size_mb = (num_params * 2) / (1024^2) / tensor_parallel_size
bandwidth_mbps = 3500  # ~3.5 GB/s for SSD
latency_ms = 1.0
load_time_ssd = latency + (model_size_mb / bandwidth_mbps) * 1000
```

**Example**:
```json
"load_time_ssd_ms": 2345.67
```

**Typical Values**:
- Llama-3-8B: ~2-4 seconds
- Llama-3-70B (TP4): ~5-10 seconds

---

### 6. load_time_remote_ms

**Purpose**: Cold start time when loading from remote storage (slowest tier)

**Calculation**:
```python
# From model loading profiling for remote_blob tier
load_time_remote = load_model_weights + model_to_gpu + storage_tier_latency
```

**Sources**:
- `data/profiling/faasinfer/model_loading/loading_{timestamp}.csv`
  - Filter: `storage_tier='remote_blob'`
  - Columns: `time_stats.load_model_weights.mean`, `time_stats.model_to_gpu.mean`, `time_stats.storage_tier_latency.mean`

**Fallback**:
```python
model_size_mb = (num_params * 2) / (1024^2) / tensor_parallel_size
bandwidth_mbps = 1250  # ~1.25 GB/s for network (10 Gbps)
latency_ms = 50.0
load_time_remote = latency + (model_size_mb / bandwidth_mbps) * 1000
```

**Example**:
```json
"load_time_remote_ms": 8901.23
```

**Typical Values**:
- Llama-3-8B: ~5-10 seconds
- Llama-3-70B (TP4): ~15-30 seconds

---

## Complete Example: sim_params.json

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

---

## Profiling Workflow Summary

### Step 1: Profile All Components
```bash
python -m vidur.profiling.faasinfer.profile_all \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 4 \
    --storage_tiers remote_blob local_ssd nvme_cache \
    --num_tensor_parallel_workers 1 2 4 \
    --all
```

### Step 2: Generate sim_params.json
```bash
python -m vidur.profiling.faasinfer.generate_sim_params \
    --model meta-llama/Meta-Llama-3-8B \
    --device a100 \
    --tensor_parallel_size 1 \
    --output sim_params.json \
    --pretty
```

### Step 3: Use in Capacity Planning
```bash
# Your SimFaaSInfer tool uses sim_params.json
sim-faasinfer --config sim_params.json --workload trace.csv
```

---

## Validation Checklist

Before using sim_params.json, verify:

- [ ] All profiling scripts have completed successfully
- [ ] CSV files exist in `data/profiling/`
- [ ] GPU memory estimate is reasonable for the model
- [ ] Prefill time < Decode time (typically)
- [ ] Load times: DRAM < SSD < Remote
- [ ] Values match expected ranges for the hardware

---

## Troubleshooting

**Missing profiling data**:
- The generator will use estimates based on model size and device
- For production use, always run actual profiling

**Unrealistic values**:
- Check profiling CSV files for anomalies
- Re-run profiling with more iterations
- Verify GPU is not throttled

**Parameter extraction errors**:
- Ensure model name matches exactly between profiling and generation
- Check tensor_parallel_size matches profiling configuration
- Verify CSV column names match expected format
