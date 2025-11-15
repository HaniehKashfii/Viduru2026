"""Unified profiling script for FaaSInfer.

This script runs all FaaSInfer profiling components:
1. Model Loading (Cold Start) Profiling
2. Storage I/O Profiling
3. Scaling Operations Profiling
4. Compute Profiling (reuses Viduru's MLP and Attention profiling)
5. Network Profiling (reuses Viduru's collectives profiling)

Usage:
    python -m vidur.profiling.faasinfer.profile_all \
        --models meta-llama/Meta-Llama-3-8B \
        --num_gpus 4 \
        --storage_tiers remote_blob local_ssd nvme_cache \
        --all

This provides a complete profiling dataset for serverless LLM inference simulation.
"""

import argparse
import subprocess
import sys
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaaSInfer Unified Profiling - Profile all components for serverless LLM inference"
    )

    # Common arguments
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["meta-llama/Meta-Llama-3-8B"],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of GPUs available for profiling",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Tensor parallel configurations to profile",
    )
    parser.add_argument(
        "--storage_tiers",
        type=str,
        nargs="+",
        default=["remote_blob", "local_ssd", "nvme_cache"],
        help="Storage tiers to profile",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="data/profiling/faasinfer",
        help="Base output directory for all profiling results",
    )

    # Component selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all profiling components",
    )
    parser.add_argument(
        "--model_loading",
        action="store_true",
        help="Profile model loading (cold start)",
    )
    parser.add_argument(
        "--storage_io",
        action="store_true",
        help="Profile storage I/O",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Profile scaling operations",
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Profile compute operations (MLP, Attention) - uses Viduru profiling",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Profile network operations (AllReduce, Send-Recv) - uses Viduru profiling",
    )

    # Profiling parameters
    parser.add_argument(
        "--profile_method",
        default="perf_counter",
        help="Profiling method to use",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens for compute profiling",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=16384,
        help="Maximum sequence length for attention profiling",
    )

    args = parser.parse_args()

    # If --all is specified, enable all components
    if args.all:
        args.model_loading = True
        args.storage_io = True
        args.scaling = True
        args.compute = True
        args.network = True

    # If no component is specified, show error
    if not any(
        [
            args.model_loading,
            args.storage_io,
            args.scaling,
            args.compute,
            args.network,
        ]
    ):
        parser.error(
            "Please specify at least one profiling component or use --all to profile everything"
        )

    return args


def run_command(cmd: List[str], description: str):
    """Run a command and handle errors."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed with exception: {e}")
        return False


def main():
    args = parse_args()

    print("=" * 80)
    print("FaaSInfer Unified Profiling")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Tensor Parallel: {args.num_tensor_parallel_workers}")
    print(f"Storage Tiers: {args.storage_tiers}")
    print("=" * 80)

    results = {
        "model_loading": False,
        "storage_io": False,
        "scaling": False,
        "compute": False,
        "network": False,
    }

    # 1. Model Loading Profiling (FaaSInfer-specific)
    if args.model_loading:
        cmd = [
            sys.executable,
            "-m",
            "vidur.profiling.faasinfer.model_loading.main",
            "--models",
            *args.models,
            "--num_gpus",
            str(args.num_gpus),
            "--storage_tiers",
            *args.storage_tiers,
            "--num_tensor_parallel_workers",
            *[str(x) for x in args.num_tensor_parallel_workers],
            "--profile_method",
            args.profile_method,
        ]
        results["model_loading"] = run_command(
            cmd, "Model Loading (Cold Start) Profiling"
        )

    # 2. Storage I/O Profiling (FaaSInfer-specific)
    if args.storage_io:
        cmd = [
            sys.executable,
            "-m",
            "vidur.profiling.faasinfer.storage_io.main",
            "--storage_tiers",
            *args.storage_tiers,
            "--profile_method",
            args.profile_method,
        ]
        results["storage_io"] = run_command(cmd, "Storage I/O Profiling")

    # 3. Scaling Operations Profiling (FaaSInfer-specific)
    if args.scaling:
        cmd = [
            sys.executable,
            "-m",
            "vidur.profiling.faasinfer.scaling.main",
            "--models",
            *args.models,
            "--num_gpus",
            str(args.num_gpus),
            "--num_tensor_parallel_workers",
            *[str(x) for x in args.num_tensor_parallel_workers],
            "--profile_method",
            args.profile_method,
        ]
        results["scaling"] = run_command(cmd, "Scaling Operations Profiling")

    # 4. Compute Profiling (Reuses Viduru's profiling)
    if args.compute:
        # MLP Profiling
        cmd_mlp = [
            sys.executable,
            "-m",
            "vidur.profiling.mlp.main",
            "--models",
            *args.models,
            "--num_gpus",
            str(args.num_gpus),
            "--num_tensor_parallel_workers",
            *[str(x) for x in args.num_tensor_parallel_workers],
            "--max_tokens",
            str(args.max_tokens),
            "--profile_method",
            args.profile_method,
        ]
        mlp_success = run_command(cmd_mlp, "Compute Profiling - MLP")

        # Attention Profiling
        cmd_attention = [
            sys.executable,
            "-m",
            "vidur.profiling.attention.main",
            "--models",
            *args.models,
            "--num_gpus",
            str(args.num_gpus),
            "--num_tensor_parallel_workers",
            *[str(x) for x in args.num_tensor_parallel_workers],
            "--max_seq_len",
            str(args.max_seq_len),
            "--profile_method",
            args.profile_method,
        ]
        attention_success = run_command(cmd_attention, "Compute Profiling - Attention")

        results["compute"] = mlp_success and attention_success

    # 5. Network Profiling (Reuses Viduru's profiling)
    if args.network:
        cmd = [
            sys.executable,
            "-m",
            "vidur.profiling.collectives.main",
            "--num_workers_per_node_combinations",
            ",".join([str(x) for x in args.num_tensor_parallel_workers]),
            "--collective",
            "all_reduce",
        ]
        results["network"] = run_command(cmd, "Network Profiling - AllReduce")

    # Print Summary
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    for component, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED" if success is False else "- SKIPPED"
        print(f"{component:20s}: {status}")
    print("=" * 80)

    # Overall success
    all_selected = [v for v in results.values() if v is not None]
    if all_selected and all(all_selected):
        print("\n✓ All selected profiling components completed successfully!")
        print(
            f"\nProfiling data saved to: {args.output_base_dir} and data/profiling/compute, data/profiling/network"
        )
        return 0
    else:
        print("\n✗ Some profiling components failed. Please check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
