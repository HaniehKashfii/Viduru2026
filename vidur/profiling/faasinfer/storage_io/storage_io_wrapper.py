"""Storage I/O Profiler for FaaSInfer.

This module profiles raw I/O performance from different storage tiers,
which is critical for understanding checkpoint loading performance in
serverless LLM inference.
"""

import os
import time
import tempfile
import numpy as np
import torch
from typing import Dict, Any, Optional

from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.faasinfer.common.storage_config import StorageTier, StorageConfig


class StorageIOWrapper:
    """Wrapper for profiling storage I/O operations."""

    def __init__(
        self,
        storage_config: StorageConfig,
        profile_method: str = "perf_counter",
    ):
        self.storage_config = storage_config

        # Initialize timer stats store
        TimerStatsStore(profile_method=profile_method)

        # Create storage path if it doesn't exist
        os.makedirs(self.storage_config.path, exist_ok=True)

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def profile_sequential_read(
        self,
        file_size_mb: int,
        warmup_iterations: int = 2,
        measurement_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Profile sequential read performance.

        Args:
            file_size_mb: Size of file to read (MB)
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations

        Returns:
            Dictionary with profiling statistics
        """
        timer_stats_store = TimerStatsStore()

        # Create test file
        test_file_path = self._create_test_file(file_size_mb)

        # Warmup
        for _ in range(warmup_iterations):
            self._read_file(test_file_path)

        timer_stats_store.clear_stats()

        # Measurement
        for _ in range(measurement_iterations):
            self._read_file(test_file_path)

        # Cleanup
        os.remove(test_file_path)

        return timer_stats_store.get_stats()

    def profile_random_read(
        self,
        file_size_mb: int,
        block_size_kb: int = 4,
        num_reads: int = 100,
        warmup_iterations: int = 2,
        measurement_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Profile random read performance.

        Args:
            file_size_mb: Size of file to read (MB)
            block_size_kb: Size of each read block (KB)
            num_reads: Number of random reads per iteration
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations

        Returns:
            Dictionary with profiling statistics
        """
        timer_stats_store = TimerStatsStore()

        # Create test file
        test_file_path = self._create_test_file(file_size_mb)

        # Warmup
        for _ in range(warmup_iterations):
            self._random_read_file(test_file_path, block_size_kb, num_reads)

        timer_stats_store.clear_stats()

        # Measurement
        for _ in range(measurement_iterations):
            self._random_read_file(test_file_path, block_size_kb, num_reads)

        # Cleanup
        os.remove(test_file_path)

        return timer_stats_store.get_stats()

    def profile_checkpoint_read(
        self,
        checkpoint_size_mb: int,
        warmup_iterations: int = 1,
        measurement_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Profile reading a model checkpoint (PyTorch format).

        Args:
            checkpoint_size_mb: Size of checkpoint to read (MB)
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations

        Returns:
            Dictionary with profiling statistics
        """
        timer_stats_store = TimerStatsStore()

        # Create dummy checkpoint
        checkpoint_path = self._create_dummy_checkpoint(checkpoint_size_mb)

        # Warmup
        for _ in range(warmup_iterations):
            self._load_checkpoint(checkpoint_path)

        timer_stats_store.clear_stats()

        # Measurement
        for _ in range(measurement_iterations):
            self._load_checkpoint(checkpoint_path)

        # Cleanup
        os.remove(checkpoint_path)

        return timer_stats_store.get_stats()

    def _create_test_file(self, size_mb: int) -> str:
        """Create a test file of specified size."""
        file_path = os.path.join(
            self.storage_config.path, f"test_file_{size_mb}mb.bin"
        )

        # Create file with random data
        chunk_size = 1024 * 1024  # 1 MB chunks
        with open(file_path, "wb") as f:
            for _ in range(size_mb):
                data = np.random.bytes(chunk_size)
                f.write(data)

        return file_path

    def _read_file(self, file_path: str):
        """Read entire file sequentially."""
        with CudaTimer("sequential_read"):
            with open(file_path, "rb") as f:
                # Read in chunks to simulate realistic I/O
                chunk_size = 1024 * 1024  # 1 MB
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

    def _random_read_file(self, file_path: str, block_size_kb: int, num_reads: int):
        """Perform random reads from file."""
        file_size = os.path.getsize(file_path)
        block_size = block_size_kb * 1024

        with CudaTimer("random_read"):
            with open(file_path, "rb") as f:
                for _ in range(num_reads):
                    # Random offset
                    offset = np.random.randint(0, max(1, file_size - block_size))
                    f.seek(offset)
                    f.read(block_size)

    def _create_dummy_checkpoint(self, size_mb: int) -> str:
        """Create a dummy PyTorch checkpoint."""
        checkpoint_path = os.path.join(
            self.storage_config.path, f"checkpoint_{size_mb}mb.pt"
        )

        # Create tensors to fill checkpoint
        # Estimate number of parameters needed
        num_params = (size_mb * 1024 * 1024) // 2  # 2 bytes per param (fp16)

        # Create dummy state dict
        state_dict = {
            "model": {
                f"layer_{i}": torch.randn(1000, 1000, dtype=torch.float16)
                for i in range(num_params // 1000000 + 1)
            }
        }

        # Save checkpoint
        with CudaTimer("create_checkpoint"):
            torch.save(state_dict, checkpoint_path)

        return checkpoint_path

    def _load_checkpoint(self, checkpoint_path: str):
        """Load a PyTorch checkpoint."""
        with CudaTimer("load_checkpoint"):
            state_dict = torch.load(checkpoint_path, map_location=self.device)

        # Move to GPU if available
        if self.device.type == "cuda":
            with CudaTimer("checkpoint_to_gpu"):
                # Simulate moving tensors to GPU
                for key in list(state_dict.get("model", {}).keys())[:5]:  # Sample
                    tensor = state_dict["model"][key]
                    _ = tensor.to(self.device)
                torch.cuda.synchronize()

    def get_theoretical_bandwidth(self) -> float:
        """Get theoretical bandwidth for this storage tier (MB/s)."""
        return self.storage_config.bandwidth_mbps or 0.0

    def get_theoretical_latency(self) -> float:
        """Get theoretical latency for this storage tier (ms)."""
        return self.storage_config.latency_ms or 0.0
