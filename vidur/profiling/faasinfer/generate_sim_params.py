"""Generate sim_params.json for SimFaaSInfer capacity planning.

This script takes all the profiled data (model loading, storage I/O, scaling,
compute, network) and generates the six high-fidelity parameters required by
the SimFaaSInfer capacity planning tool:

1. gpu_memory_mb - GPU memory consumed by model weights
2. prefill_time_per_token_ms - Time per prompt token (TTFT)
3. decode_time_per_token_ms - Time per output token (TBT)
4. load_time_dram_ms - Cold start from DRAM cache
5. load_time_ssd_ms - Cold start from SSD cache
6. load_time_remote_ms - Cold start from remote storage

Usage:
    python -m vidur.profiling.faasinfer.generate_sim_params \
        --model meta-llama/Meta-Llama-3-8B \
        --device a100 \
        --tensor_parallel_size 1 \
        --output sim_params.json
"""

import argparse
import json
import os
import glob
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from vidur.config.model_config import BaseModelConfig
from vidur.logger import init_logger

logger = init_logger(__name__)


class SimParamsGenerator:
    """Generator for SimFaaSInfer capacity planning parameters."""

    def __init__(
        self,
        model_name: str,
        device: str,
        tensor_parallel_size: int,
        profiling_data_dir: str = "data/profiling",
    ):
        self.model_name = model_name
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.profiling_data_dir = profiling_data_dir

        # Load model config for GPU memory calculation
        self.model_config = BaseModelConfig.create_from_name(model_name)

    def generate(self) -> Dict[str, Any]:
        """Generate all six sim_params from profiling data."""
        sim_params = {
            "model_name": self.model_name,
            "device": self.device,
            "tensor_parallel_size": self.tensor_parallel_size,
        }

        # 1. GPU Memory
        sim_params["gpu_memory_mb"] = self._calculate_gpu_memory()

        # 2 & 3. Prefill and Decode Time per Token
        prefill_time, decode_time = self._calculate_token_times()
        sim_params["prefill_time_per_token_ms"] = prefill_time
        sim_params["decode_time_per_token_ms"] = decode_time

        # 4, 5, 6. Load times for different storage tiers
        load_times = self._calculate_load_times()
        sim_params["load_time_dram_ms"] = load_times.get("dram", 0)
        sim_params["load_time_ssd_ms"] = load_times.get("ssd", 0)
        sim_params["load_time_remote_ms"] = load_times.get("remote", 0)

        # Add additional metadata
        sim_params["metadata"] = self._get_metadata()

        return sim_params

    def _calculate_gpu_memory(self) -> float:
        """Calculate GPU memory consumed by model weights.

        Formula:
            gpu_memory_mb = (num_parameters * bytes_per_param) / (1024^2)

        For bfloat16/float16: 2 bytes per parameter
        Account for tensor parallelism (weights are sharded)
        """
        # Get model parameters
        num_params = self._estimate_model_parameters()

        # Bytes per parameter (assuming bfloat16/float16)
        bytes_per_param = 2

        # Total memory in MB
        total_memory_mb = (num_params * bytes_per_param) / (1024 * 1024)

        # Account for tensor parallelism (weights are sharded)
        memory_per_gpu = total_memory_mb / self.tensor_parallel_size

        # Add overhead for KV cache, activations, etc. (roughly 20-30%)
        memory_with_overhead = memory_per_gpu * 1.25

        logger.info(
            f"Calculated GPU memory: {memory_with_overhead:.2f} MB "
            f"({num_params / 1e9:.2f}B params, TP={self.tensor_parallel_size})"
        )

        return round(memory_with_overhead, 2)

    def _estimate_model_parameters(self) -> int:
        """Estimate total model parameters from config."""
        # Formula for transformer parameters:
        # params ≈ 12 * n_layers * d_model^2 * (1 + 13/(12*d_model) + vocab_size/(12*n_layers*d_model))

        n_layers = self.model_config.num_layers
        d_model = self.model_config.embedding_dim
        vocab_size = self.model_config.vocab_size

        # Simplified calculation
        # Embedding: vocab_size * d_model
        embedding_params = vocab_size * d_model

        # Each layer has:
        # - Attention: 4 * d_model^2 (Q, K, V, O projections)
        # - MLP: 8 * d_model^2 (typically 2-3x expansion)
        # - LayerNorm: 2 * d_model
        layer_params = n_layers * (
            4 * d_model * d_model  # Attention
            + self.model_config.mlp_hidden_dim * d_model * 2  # MLP up and down
            + 2 * d_model  # LayerNorm
        )

        total_params = embedding_params + layer_params

        logger.info(f"Estimated {total_params / 1e9:.2f}B parameters for {self.model_name}")
        return total_params

    def _calculate_token_times(self) -> Tuple[float, float]:
        """Calculate prefill and decode time per token.

        These are derived from compute profiling (MLP + Attention).

        Returns:
            (prefill_time_per_token_ms, decode_time_per_token_ms)
        """
        # Load MLP profiling data
        mlp_data = self._load_compute_profiling("mlp.csv")

        # Load Attention profiling data
        attention_data = self._load_compute_profiling("attention.csv")

        if mlp_data is None or attention_data is None:
            logger.warning("Compute profiling data not found, using estimates")
            return self._estimate_token_times()

        # Filter by model and TP size
        mlp_filtered = mlp_data[
            (mlp_data["num_tensor_parallel_workers"] == self.tensor_parallel_size)
        ]
        attn_filtered = attention_data[
            (attention_data["num_tensor_parallel_workers"] == self.tensor_parallel_size)
        ]

        if mlp_filtered.empty or attn_filtered.empty:
            logger.warning("No matching profiling data, using estimates")
            return self._estimate_token_times()

        # Calculate prefill time per token (single token in prefill phase)
        # Use attention prefill time + MLP time for small num_tokens
        prefill_time_per_token = self._extract_prefill_time(mlp_filtered, attn_filtered)

        # Calculate decode time per token (single token in decode phase)
        # Use attention decode time + MLP time for single token
        decode_time_per_token = self._extract_decode_time(mlp_filtered, attn_filtered)

        logger.info(
            f"Calculated prefill time: {prefill_time_per_token:.4f} ms/token, "
            f"decode time: {decode_time_per_token:.4f} ms/token"
        )

        return prefill_time_per_token, decode_time_per_token

    def _extract_prefill_time(self, mlp_df: pd.DataFrame, attn_df: pd.DataFrame) -> float:
        """Extract prefill time per token from profiling data."""
        # For prefill, we care about the cost of processing each token in the prompt
        # Use a moderate num_tokens (e.g., 32 or 64) to get stable measurements

        # Get MLP time for moderate batch
        target_tokens = 32
        mlp_rows = mlp_df[mlp_df["num_tokens"] == target_tokens]

        if mlp_rows.empty:
            # Fallback to smallest available
            mlp_rows = mlp_df.nsmallest(1, "num_tokens")

        # Extract mean times for key operations
        mlp_time = 0
        if "time_stats.mlp_up_proj.mean" in mlp_rows.columns:
            mlp_time += mlp_rows["time_stats.mlp_up_proj.mean"].iloc[0]
        if "time_stats.mlp_down_proj.mean" in mlp_rows.columns:
            mlp_time += mlp_rows["time_stats.mlp_down_proj.mean"].iloc[0]
        if "time_stats.mlp_act.mean" in mlp_rows.columns:
            mlp_time += mlp_rows["time_stats.mlp_act.mean"].iloc[0]

        # Get attention prefill time
        attn_rows = attn_df[attn_df["is_prefill"] == True]
        if not attn_rows.empty:
            attn_time = attn_rows["time_stats.attn_prefill.mean"].iloc[0]
        else:
            attn_time = 0

        # Per-token time = (total_time) / num_tokens
        total_time = mlp_time + attn_time
        num_tokens = mlp_rows["num_tokens"].iloc[0] if not mlp_rows.empty else 32

        # Account for number of layers
        time_per_layer = total_time / num_tokens
        time_all_layers = time_per_layer * self.model_config.num_layers

        return round(time_all_layers, 4)

    def _extract_decode_time(self, mlp_df: pd.DataFrame, attn_df: pd.DataFrame) -> float:
        """Extract decode time per token from profiling data."""
        # For decode, each iteration processes a single new token

        # Get MLP time for single token
        mlp_rows = mlp_df[mlp_df["num_tokens"] == 1]

        if mlp_rows.empty:
            mlp_rows = mlp_df.nsmallest(1, "num_tokens")

        mlp_time = 0
        if "time_stats.mlp_up_proj.mean" in mlp_rows.columns:
            mlp_time += mlp_rows["time_stats.mlp_up_proj.mean"].iloc[0]
        if "time_stats.mlp_down_proj.mean" in mlp_rows.columns:
            mlp_time += mlp_rows["time_stats.mlp_down_proj.mean"].iloc[0]
        if "time_stats.mlp_act.mean" in mlp_rows.columns:
            mlp_time += mlp_rows["time_stats.mlp_act.mean"].iloc[0]

        # Get attention decode time (batch_size=1, moderate kv_cache)
        attn_rows = attn_df[
            (attn_df["is_prefill"] == False) & (attn_df["batch_size"] == 1)
        ]
        if not attn_rows.empty:
            # Use median KV cache size
            attn_time = attn_rows["time_stats.attn_decode.mean"].median()
        else:
            attn_time = 0

        # Account for number of layers
        time_per_layer = mlp_time + attn_time
        time_all_layers = time_per_layer * self.model_config.num_layers

        return round(time_all_layers, 4)

    def _estimate_token_times(self) -> Tuple[float, float]:
        """Provide rough estimates when profiling data is unavailable."""
        # Very rough estimates based on model size and device
        num_params_b = self._estimate_model_parameters() / 1e9

        # Rough heuristics (ms per token)
        if "h100" in self.device.lower():
            prefill_base = 0.01 * num_params_b  # ~0.08ms for 8B model
            decode_base = 0.02 * num_params_b  # ~0.16ms for 8B model
        elif "a100" in self.device.lower():
            prefill_base = 0.015 * num_params_b  # ~0.12ms for 8B model
            decode_base = 0.03 * num_params_b  # ~0.24ms for 8B model
        else:
            prefill_base = 0.02 * num_params_b
            decode_base = 0.04 * num_params_b

        # Account for TP
        prefill_time = prefill_base / self.tensor_parallel_size
        decode_time = decode_base / self.tensor_parallel_size

        logger.warning(f"Using estimated token times (no profiling data available)")
        return round(prefill_time, 4), round(decode_time, 4)

    def _calculate_load_times(self) -> Dict[str, float]:
        """Calculate load times for different storage tiers.

        Returns dict with keys: 'dram', 'ssd', 'remote'
        """
        # Load model loading profiling data
        loading_data = self._load_faasinfer_profiling("model_loading", "loading_*.csv")

        if loading_data is None:
            logger.warning("Model loading data not found, using estimates")
            return self._estimate_load_times()

        # Filter by model and TP size
        filtered = loading_data[
            (loading_data["model"] == self.model_name)
            & (loading_data["num_tensor_parallel_workers"] == self.tensor_parallel_size)
        ]

        if filtered.empty:
            logger.warning("No matching model loading data, using estimates")
            return self._estimate_load_times()

        # Extract load times for each storage tier
        load_times = {}

        # Map storage tiers to output names
        tier_mapping = {
            "nvme_cache": "dram",  # NVMe cache is closest to DRAM speed
            "local_ssd": "ssd",
            "remote_blob": "remote",
        }

        for storage_tier, output_name in tier_mapping.items():
            tier_data = filtered[filtered["storage_tier"] == storage_tier]

            if not tier_data.empty:
                # Get total loading time (weights + GPU transfer)
                load_time = tier_data["time_stats.load_model_weights.mean"].iloc[0]

                if "time_stats.model_to_gpu.mean" in tier_data.columns:
                    load_time += tier_data["time_stats.model_to_gpu.mean"].iloc[0]

                load_times[output_name] = round(load_time, 2)
            else:
                logger.warning(f"No data for storage tier {storage_tier}")

        # Fill missing values with estimates
        if "dram" not in load_times:
            load_times["dram"] = self._estimate_load_time_for_tier("dram")
        if "ssd" not in load_times:
            load_times["ssd"] = self._estimate_load_time_for_tier("ssd")
        if "remote" not in load_times:
            load_times["remote"] = self._estimate_load_time_for_tier("remote")

        logger.info(
            f"Load times: DRAM={load_times['dram']:.2f}ms, "
            f"SSD={load_times['ssd']:.2f}ms, Remote={load_times['remote']:.2f}ms"
        )

        return load_times

    def _estimate_load_times(self) -> Dict[str, float]:
        """Estimate load times based on model size and storage characteristics."""
        return {
            "dram": self._estimate_load_time_for_tier("dram"),
            "ssd": self._estimate_load_time_for_tier("ssd"),
            "remote": self._estimate_load_time_for_tier("remote"),
        }

    def _estimate_load_time_for_tier(self, tier: str) -> float:
        """Estimate load time for a specific storage tier."""
        # Get model size in MB
        num_params = self._estimate_model_parameters()
        model_size_mb = (num_params * 2) / (1024 * 1024)  # 2 bytes per param

        # Account for TP (weights are sharded)
        model_size_mb = model_size_mb / self.tensor_parallel_size

        # Storage tier bandwidths (MB/s)
        bandwidths = {
            "dram": 7000,  # ~7 GB/s for NVMe
            "ssd": 3500,  # ~3.5 GB/s for SSD
            "remote": 1250,  # ~1.25 GB/s for network
        }

        # Base latencies (ms)
        latencies = {
            "dram": 0.1,
            "ssd": 1.0,
            "remote": 50.0,
        }

        bandwidth = bandwidths.get(tier, 3500)
        latency = latencies.get(tier, 1.0)

        # Load time = latency + (size / bandwidth)
        load_time = latency + (model_size_mb / bandwidth) * 1000  # Convert to ms

        logger.info(
            f"Estimated load time for {tier}: {load_time:.2f}ms "
            f"(model size: {model_size_mb:.2f}MB)"
        )

        return round(load_time, 2)

    def _load_compute_profiling(self, filename: str) -> Optional[pd.DataFrame]:
        """Load compute profiling data (MLP or Attention)."""
        pattern = os.path.join(
            self.profiling_data_dir, "compute", self.device, self.model_name.replace("/", "_"), filename
        )

        files = glob.glob(pattern)
        if not files:
            # Try alternate pattern
            pattern = os.path.join(
                self.profiling_data_dir, "compute", self.device, "*", filename
            )
            files = glob.glob(pattern)

        if files:
            return pd.read_csv(files[0])
        return None

    def _load_faasinfer_profiling(
        self, component: str, pattern: str
    ) -> Optional[pd.DataFrame]:
        """Load FaaSInfer profiling data."""
        search_pattern = os.path.join(
            self.profiling_data_dir, "faasinfer", component, pattern
        )

        files = glob.glob(search_pattern)
        if files:
            # Return most recent file
            latest_file = max(files, key=os.path.getmtime)
            return pd.read_csv(latest_file)
        return None

    def _get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata about the profiling."""
        return {
            "num_layers": self.model_config.num_layers,
            "embedding_dim": self.model_config.embedding_dim,
            "num_q_heads": self.model_config.num_q_heads,
            "num_kv_heads": self.model_config.num_kv_heads,
            "vocab_size": self.model_config.vocab_size,
            "estimated_parameters": self._estimate_model_parameters(),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sim_params.json for SimFaaSInfer capacity planning"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., meta-llama/Meta-Llama-3-8B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device name (e.g., a100, h100)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--profiling_data_dir",
        type=str,
        default="data/profiling",
        help="Base directory for profiling data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sim_params.json",
        help="Output file path for sim_params.json",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("SimFaaSInfer Parameter Generation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"Profiling Data Dir: {args.profiling_data_dir}")
    print(f"Output: {args.output}")
    print("=" * 80)
    print()

    # Create generator
    generator = SimParamsGenerator(
        model_name=args.model,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        profiling_data_dir=args.profiling_data_dir,
    )

    # Generate parameters
    sim_params = generator.generate()

    # Save to file
    with open(args.output, "w") as f:
        if args.pretty:
            json.dump(sim_params, f, indent=2)
        else:
            json.dump(sim_params, f)

    print(f"\n✓ Generated sim_params.json: {args.output}")
    print("\n" + "=" * 80)
    print("SIMULATION PARAMETERS")
    print("=" * 80)
    print(f"GPU Memory:              {sim_params['gpu_memory_mb']:.2f} MB")
    print(f"Prefill Time/Token:      {sim_params['prefill_time_per_token_ms']:.4f} ms")
    print(f"Decode Time/Token:       {sim_params['decode_time_per_token_ms']:.4f} ms")
    print(f"Load Time (DRAM):        {sim_params['load_time_dram_ms']:.2f} ms")
    print(f"Load Time (SSD):         {sim_params['load_time_ssd_ms']:.2f} ms")
    print(f"Load Time (Remote):      {sim_params['load_time_remote_ms']:.2f} ms")
    print("=" * 80)

    # Pretty print full JSON
    if args.pretty:
        print("\nFull sim_params.json:")
        print(json.dumps(sim_params, indent=2))


if __name__ == "__main__":
    main()
