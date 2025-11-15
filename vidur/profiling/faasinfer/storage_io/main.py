"""Main script for profiling storage I/O in FaaSInfer.

This script profiles raw I/O performance from different storage tiers,
measuring sequential reads, random reads, and checkpoint loading times.

Usage:
    python -m vidur.profiling.faasinfer.storage_io.main \
        --storage_tiers remote_blob local_ssd nvme_cache \
        --file_sizes_mb 100 500 1000 5000 \
        --num_gpus 1

Output:
    CSV files in data/profiling/faasinfer/storage_io/io_profile_{timestamp}.csv
"""

import argparse
import datetime
import os
from typing import Any

import pandas as pd
from tqdm import tqdm

from vidur.profiling.faasinfer.storage_io.storage_io_wrapper import StorageIOWrapper
from vidur.profiling.faasinfer.common.storage_config import StorageConfig
from vidur.profiling.utils import ProfileMethod


def parse_args():
    parser = argparse.ArgumentParser(description="FaaSInfer Storage I/O Profiling")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/profiling/faasinfer/storage_io",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--storage_tiers",
        type=str,
        nargs="+",
        default=["remote_blob", "local_ssd", "nvme_cache"],
        help="Storage tiers to profile",
    )
    parser.add_argument(
        "--file_sizes_mb",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000],
        help="File sizes to profile (MB)",
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
        default=2,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--measurement_iterations",
        type=int,
        default=5,
        help="Number of measurement iterations",
    )
    parser.add_argument(
        "--profile_random_io",
        action="store_true",
        help="Also profile random I/O performance",
    )
    parser.add_argument(
        "--random_block_size_kb",
        type=int,
        default=4,
        help="Block size for random I/O (KB)",
    )
    parser.add_argument(
        "--num_random_reads",
        type=int,
        default=100,
        help="Number of random reads per iteration",
    )

    args = parser.parse_args()
    return args


def profile_storage_io(
    args: argparse.Namespace,
    storage_tier: str,
    file_size_mb: int,
    pbar: Any,
):
    """Profile storage I/O for a specific configuration."""
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
    wrapper = StorageIOWrapper(
        storage_config=storage_config,
        profile_method=args.profile_method,
    )

    # Profile sequential read
    seq_stats = wrapper.profile_sequential_read(
        file_size_mb=file_size_mb,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )

    # Profile random read if requested
    random_stats = None
    if args.profile_random_io:
        random_stats = wrapper.profile_random_read(
            file_size_mb=file_size_mb,
            block_size_kb=args.random_block_size_kb,
            num_reads=args.num_random_reads,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )

    # Profile checkpoint read
    checkpoint_stats = wrapper.profile_checkpoint_read(
        checkpoint_size_mb=file_size_mb,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )

    # Build result row
    result = {
        "storage_tier": storage_tier,
        "file_size_mb": file_size_mb,
        "theoretical_bandwidth_mbps": wrapper.get_theoretical_bandwidth(),
        "theoretical_latency_ms": wrapper.get_theoretical_latency(),
    }

    # Add sequential read stats
    for metric_name, metric_stats in seq_stats.items():
        for stat_type, stat_value in metric_stats.items():
            result[f"sequential.{metric_name}.{stat_type}"] = stat_value

    # Calculate observed bandwidth for sequential reads
    if "sequential_read" in seq_stats:
        mean_time_ms = seq_stats["sequential_read"]["mean"]
        observed_bandwidth = (file_size_mb / mean_time_ms) * 1000  # MB/s
        result["sequential.observed_bandwidth_mbps"] = observed_bandwidth

    # Add random read stats
    if random_stats:
        for metric_name, metric_stats in random_stats.items():
            for stat_type, stat_value in metric_stats.items():
                result[f"random.{metric_name}.{stat_type}"] = stat_value

    # Add checkpoint stats
    for metric_name, metric_stats in checkpoint_stats.items():
        for stat_type, stat_value in metric_stats.items():
            result[f"checkpoint.{metric_name}.{stat_type}"] = stat_value

    pbar.update(1)
    return result


def main():
    args = parse_args()

    print(f"FaaSInfer Storage I/O Profiling")
    print(f"Storage Tiers: {args.storage_tiers}")
    print(f"File Sizes (MB): {args.file_sizes_mb}")
    print(f"Profile Method: {args.profile_method}")
    print("-" * 80)

    # Calculate total profiling tasks
    total_tasks = len(args.storage_tiers) * len(args.file_sizes_mb)

    results = []

    with tqdm(total=total_tasks, desc="Profiling Storage I/O") as pbar:
        for storage_tier in args.storage_tiers:
            for file_size_mb in args.file_sizes_mb:
                result = profile_storage_io(args, storage_tier, file_size_mb, pbar)
                if result:
                    results.append(result)

    # Save results
    if results:
        df = pd.DataFrame(results)

        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(args.output_dir, f"io_profile_{timestamp}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        print(f"Total configurations profiled: {len(results)}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY: Storage I/O Performance")
        print("=" * 80)
        if "sequential.observed_bandwidth_mbps" in df.columns:
            summary = df.groupby(["storage_tier"])[
                "sequential.observed_bandwidth_mbps"
            ].mean()
            print("\nObserved Sequential Read Bandwidth (MB/s):")
            print(summary.to_string())

        if "checkpoint.load_checkpoint.mean" in df.columns:
            summary = df.groupby(["storage_tier", "file_size_mb"])[
                "checkpoint.load_checkpoint.mean"
            ].mean()
            print("\nCheckpoint Load Time (ms):")
            print(summary.to_string())
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
