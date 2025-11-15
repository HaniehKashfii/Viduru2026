"""Main script for profiling model loading (cold start) in FaaSInfer.

This script profiles the time taken to load LLM checkpoints from different
storage tiers, which is critical for understanding cold start overhead in
serverless LLM inference systems.

Usage:
    python -m vidur.profiling.faasinfer.model_loading.main \
        --models meta-llama/Meta-Llama-3-8B \
        --storage_tiers remote_blob local_ssd nvme_cache \
        --num_gpus 4 \
        --num_tensor_parallel_workers 1 2 4

Output:
    CSV files in data/profiling/faasinfer/model_loading/{storage_tier}/{model}/loading.csv
"""

import argparse
import datetime
import os
from typing import Any, List

import pandas as pd
import ray
from tqdm import tqdm

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.faasinfer.model_loading.model_loader_wrapper import (
    ModelLoaderWrapper,
)
from vidur.profiling.faasinfer.common.storage_config import StorageTier, StorageConfig
from vidur.profiling.utils import ProfileMethod


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaaSInfer Model Loading (Cold Start) Profiling"
    )
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray (for single GPU profiling)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/profiling/faasinfer/model_loading",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--storage_tiers",
        type=str,
        nargs="+",
        default=["remote_blob", "local_ssd", "nvme_cache"],
        help="Storage tiers to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--profile_method",
        default="perf_counter",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=1,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--measurement_iterations",
        type=int,
        default=3,
        help="Number of measurement iterations",
    )
    parser.add_argument(
        "--profile_incremental_loading",
        action="store_true",
        help="Also profile incremental/streaming model loading",
    )
    parser.add_argument(
        "--chunk_size_mb",
        type=int,
        default=64,
        help="Chunk size for incremental loading (MB)",
    )

    args = parser.parse_args()
    return args


def profile_model_loading(
    args: argparse.Namespace,
    model: str,
    storage_tier: str,
    num_tensor_parallel_workers: int,
    pbar: Any,
):
    """Profile model loading for a specific configuration."""
    model_config = ModelConfig.from_model_name(model)

    # Skip TP configurations that don't make sense
    if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
        pbar.update(1)
        return None

    # Get storage config
    storage_configs = {
        sc.tier.value: sc for sc in StorageConfig.get_default_configs()
    }
    storage_config = storage_configs.get(storage_tier)

    if storage_config is None:
        print(f"Warning: Unknown storage tier {storage_tier}, skipping")
        pbar.update(1)
        return None

    # Create wrapper
    if args.disable_ray:
        wrapper = ModelLoaderWrapper(
            model_config=model_config,
            storage_config=storage_config,
            profile_method=args.profile_method,
            num_tensor_parallel_workers=num_tensor_parallel_workers,
        )
    else:
        # Use Ray for distributed profiling
        wrapper_class = ray.remote(num_cpus=1, num_gpus=1)(ModelLoaderWrapper)
        wrapper = wrapper_class.remote(
            model_config=model_config,
            storage_config=storage_config,
            profile_method=args.profile_method,
            num_tensor_parallel_workers=num_tensor_parallel_workers,
        )

    # Profile checkpoint loading
    if args.disable_ray:
        stats = wrapper.profile_checkpoint_loading(
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )
    else:
        stats = ray.get(
            wrapper.profile_checkpoint_loading.remote(
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )

    # Profile incremental loading if requested
    incremental_stats = None
    if args.profile_incremental_loading:
        if args.disable_ray:
            incremental_stats = wrapper.profile_incremental_loading(
                chunk_size_mb=args.chunk_size_mb,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        else:
            incremental_stats = ray.get(
                wrapper.profile_incremental_loading.remote(
                    chunk_size_mb=args.chunk_size_mb,
                    warmup_iterations=args.warmup_iterations,
                    measurement_iterations=args.measurement_iterations,
                )
            )

    # Build result row
    result = {
        "model": model,
        "storage_tier": storage_tier,
        "num_tensor_parallel_workers": num_tensor_parallel_workers,
        "num_layers": model_config.num_layers,
        "num_q_heads": model_config.num_q_heads,
        "num_kv_heads": model_config.num_kv_heads,
        "embedding_dim": model_config.embedding_dim,
        "vocab_size": model_config.vocab_size,
    }

    # Add loading stats
    for metric_name, metric_stats in stats.items():
        for stat_type, stat_value in metric_stats.items():
            result[f"time_stats.{metric_name}.{stat_type}"] = stat_value

    # Add incremental loading stats if available
    if incremental_stats:
        for metric_name, metric_stats in incremental_stats.items():
            for stat_type, stat_value in metric_stats.items():
                result[f"incremental.{metric_name}.{stat_type}"] = stat_value

    pbar.update(1)
    return result


def main():
    args = parse_args()

    # Initialize Ray if not disabled
    if not args.disable_ray:
        ray.init(ignore_reinit_error=True)

    print(f"FaaSInfer Model Loading Profiling")
    print(f"Models: {args.models}")
    print(f"Storage Tiers: {args.storage_tiers}")
    print(f"Tensor Parallel Workers: {args.num_tensor_parallel_workers}")
    print(f"Profile Method: {args.profile_method}")
    print("-" * 80)

    # Calculate total profiling tasks
    total_tasks = (
        len(args.models)
        * len(args.storage_tiers)
        * len(args.num_tensor_parallel_workers)
    )

    results = []

    with tqdm(total=total_tasks, desc="Profiling") as pbar:
        for model in args.models:
            for storage_tier in args.storage_tiers:
                for num_tp_workers in args.num_tensor_parallel_workers:
                    result = profile_model_loading(
                        args, model, storage_tier, num_tp_workers, pbar
                    )
                    if result:
                        results.append(result)

    # Save results
    if results:
        df = pd.DataFrame(results)

        # Create output directory structure
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(args.output_dir, f"loading_{timestamp}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        print(f"Total configurations profiled: {len(results)}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY: Cold Start Times (mean, in seconds)")
        print("=" * 80)
        if "time_stats.load_model_weights.mean" in df.columns:
            summary = df.groupby(["model", "storage_tier"])[
                "time_stats.load_model_weights.mean"
            ].mean()
            print(summary.to_string())
    else:
        print("No results to save")

    # Shutdown Ray
    if not args.disable_ray:
        ray.shutdown()


if __name__ == "__main__":
    main()
