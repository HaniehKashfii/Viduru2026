"""Main script for profiling scaling operations in FaaSInfer.

This script profiles autoscaling operations including instance startup,
shutdown, scale-up, and scale-down latencies.

Usage:
    python -m vidur.profiling.faasinfer.scaling.main \
        --models meta-llama/Meta-Llama-3-8B \
        --num_instances 1 2 4 8 \
        --num_gpus 8

Output:
    CSV files in data/profiling/faasinfer/scaling/scaling_{timestamp}.csv
"""

import argparse
import datetime
import os
from typing import Any

import pandas as pd
import ray
from tqdm import tqdm

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.faasinfer.scaling.scaling_wrapper import ScalingWrapper
from vidur.profiling.utils import ProfileMethod


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaaSInfer Scaling Operations Profiling"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs available for profiling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/profiling/faasinfer/scaling",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Llama-2-7b-hf",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of instances for scale-up/down profiling",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1],
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
        default=5,
        help="Number of measurement iterations",
    )
    parser.add_argument(
        "--profile_single_instance",
        action="store_true",
        default=True,
        help="Profile single instance startup/shutdown",
    )
    parser.add_argument(
        "--profile_parallel_scaling",
        action="store_true",
        default=True,
        help="Profile parallel scale-up/scale-down",
    )

    args = parser.parse_args()
    return args


def profile_scaling(
    args: argparse.Namespace,
    model: str,
    num_tp_workers: int,
    pbar: Any,
):
    """Profile scaling operations for a specific configuration."""
    model_config = ModelConfig.from_model_name(model)

    # Skip TP configurations that don't make sense
    if model_config.no_tensor_parallel and num_tp_workers > 1:
        pbar.update(1)
        return None

    # Create wrapper
    wrapper = ScalingWrapper(
        model_config=model_config,
        profile_method=args.profile_method,
        num_tensor_parallel_workers=num_tp_workers,
    )

    result = {
        "model": model,
        "num_tensor_parallel_workers": num_tp_workers,
    }

    # Profile single instance operations
    if args.profile_single_instance:
        startup_stats = wrapper.profile_instance_startup(
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )

        shutdown_stats = wrapper.profile_instance_shutdown(
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )

        # Add startup stats
        for metric_name, metric_stats in startup_stats.items():
            for stat_type, stat_value in metric_stats.items():
                result[f"startup.{metric_name}.{stat_type}"] = stat_value

        # Add shutdown stats
        for metric_name, metric_stats in shutdown_stats.items():
            for stat_type, stat_value in metric_stats.items():
                result[f"shutdown.{metric_name}.{stat_type}"] = stat_value

    # Profile parallel scaling
    if args.profile_parallel_scaling:
        scale_up_results = wrapper.profile_scale_up(
            num_instances_to_add=args.num_instances,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )

        scale_down_results = wrapper.profile_scale_down(
            num_instances_to_remove=args.num_instances,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )

        # Add scale-up stats
        for num_instances, stats in scale_up_results.items():
            for metric_name, metric_stats in stats.items():
                for stat_type, stat_value in metric_stats.items():
                    result[
                        f"scale_up_{num_instances}.{metric_name}.{stat_type}"
                    ] = stat_value

        # Add scale-down stats
        for num_instances, stats in scale_down_results.items():
            for metric_name, metric_stats in stats.items():
                for stat_type, stat_value in metric_stats.items():
                    result[
                        f"scale_down_{num_instances}.{metric_name}.{stat_type}"
                    ] = stat_value

    pbar.update(1)
    return result


def main():
    args = parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_gpus=args.num_gpus)

    print(f"FaaSInfer Scaling Operations Profiling")
    print(f"Models: {args.models}")
    print(f"Tensor Parallel Workers: {args.num_tensor_parallel_workers}")
    print(f"Instance Counts: {args.num_instances}")
    print(f"Profile Method: {args.profile_method}")
    print("-" * 80)

    # Calculate total profiling tasks
    total_tasks = len(args.models) * len(args.num_tensor_parallel_workers)

    results = []

    with tqdm(total=total_tasks, desc="Profiling Scaling") as pbar:
        for model in args.models:
            for num_tp_workers in args.num_tensor_parallel_workers:
                result = profile_scaling(args, model, num_tp_workers, pbar)
                if result:
                    results.append(result)

    # Save results
    if results:
        df = pd.DataFrame(results)

        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(args.output_dir, f"scaling_{timestamp}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        print(f"Total configurations profiled: {len(results)}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY: Scaling Operations (mean times in ms)")
        print("=" * 80)

        if "startup.instance_startup.mean" in df.columns:
            print("\nInstance Startup Time (ms):")
            summary = df.groupby(["model"])["startup.instance_startup.mean"].mean()
            print(summary.to_string())

        if "shutdown.instance_shutdown.mean" in df.columns:
            print("\nInstance Shutdown Time (ms):")
            summary = df.groupby(["model"])["shutdown.instance_shutdown.mean"].mean()
            print(summary.to_string())

        # Print scale-up/down times for each instance count
        for num_instances in args.num_instances:
            col_name = f"scale_up_{num_instances}.scale_up_{num_instances}_instances.mean"
            if col_name in df.columns:
                print(f"\nScale-Up {num_instances} Instances (ms):")
                summary = df.groupby(["model"])[col_name].mean()
                print(summary.to_string())
    else:
        print("No results to save")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
