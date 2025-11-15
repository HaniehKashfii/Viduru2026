"""Scaling Operations Profiler for FaaSInfer.

This module profiles autoscaling operations in serverless LLM inference:
- Instance startup time (container initialization, GPU allocation)
- Instance shutdown/cleanup time
- Scale-up latency
- Scale-down grace period
"""

import time
import torch
import ray
from typing import Dict, Any, Optional, List

from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.common.model_config import ModelConfig


class ScalingWrapper:
    """Wrapper for profiling autoscaling operations."""

    def __init__(
        self,
        model_config: ModelConfig,
        profile_method: str = "perf_counter",
        num_tensor_parallel_workers: int = 1,
    ):
        self.model_config = model_config
        self.num_tensor_parallel_workers = num_tensor_parallel_workers

        # Initialize timer stats store
        TimerStatsStore(profile_method=profile_method)

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Ray actors for simulating instances
        self.active_instances: List[Any] = []

    def profile_instance_startup(
        self,
        warmup_iterations: int = 1,
        measurement_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Profile instance startup time.

        Measures the time to:
        1. Initialize Ray actor (container startup simulation)
        2. Allocate GPU resources
        3. Initialize CUDA context
        4. Ready for inference

        Args:
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations

        Returns:
            Dictionary with profiling statistics
        """
        timer_stats_store = TimerStatsStore()

        # Warmup
        for _ in range(warmup_iterations):
            self._startup_instance()
            self._shutdown_instance()

        timer_stats_store.clear_stats()

        # Measurement
        for _ in range(measurement_iterations):
            self._startup_instance()
            self._shutdown_instance()

        return timer_stats_store.get_stats()

    def profile_instance_shutdown(
        self,
        warmup_iterations: int = 1,
        measurement_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Profile instance shutdown time.

        Measures the time to:
        1. Flush pending requests (if any)
        2. Clean up GPU memory
        3. Terminate actor

        Args:
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations

        Returns:
            Dictionary with profiling statistics
        """
        timer_stats_store = TimerStatsStore()

        # Warmup
        for _ in range(warmup_iterations):
            self._startup_instance()
            self._shutdown_instance()

        timer_stats_store.clear_stats()

        # Measurement (startup happens in previous profiling)
        for _ in range(measurement_iterations):
            self._startup_instance()
            # Focus on shutdown timing
            with CudaTimer("instance_shutdown"):
                self._cleanup_instance()

        return timer_stats_store.get_stats()

    def profile_scale_up(
        self,
        num_instances_to_add: List[int] = [1, 2, 4, 8],
        warmup_iterations: int = 1,
        measurement_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Profile scale-up operation.

        Measures time to spin up multiple instances in parallel.

        Args:
            num_instances_to_add: List of instance counts to test
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations

        Returns:
            Dictionary with profiling statistics per instance count
        """
        all_results = {}

        for num_instances in num_instances_to_add:
            timer_stats_store = TimerStatsStore()

            # Warmup
            for _ in range(warmup_iterations):
                self._parallel_startup(num_instances)
                self._parallel_shutdown()

            timer_stats_store.clear_stats()

            # Measurement
            for _ in range(measurement_iterations):
                with CudaTimer(f"scale_up_{num_instances}_instances"):
                    self._parallel_startup(num_instances)
                self._parallel_shutdown()

            all_results[num_instances] = timer_stats_store.get_stats()

        return all_results

    def profile_scale_down(
        self,
        num_instances_to_remove: List[int] = [1, 2, 4, 8],
        warmup_iterations: int = 1,
        measurement_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Profile scale-down operation.

        Measures time to gracefully shutdown multiple instances.

        Args:
            num_instances_to_remove: List of instance counts to test
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations

        Returns:
            Dictionary with profiling statistics per instance count
        """
        all_results = {}

        for num_instances in num_instances_to_remove:
            timer_stats_store = TimerStatsStore()

            # Warmup
            for _ in range(warmup_iterations):
                self._parallel_startup(num_instances)
                self._parallel_shutdown()

            timer_stats_store.clear_stats()

            # Measurement
            for _ in range(measurement_iterations):
                self._parallel_startup(num_instances)
                with CudaTimer(f"scale_down_{num_instances}_instances"):
                    self._parallel_shutdown()

            all_results[num_instances] = timer_stats_store.get_stats()

        return all_results

    def _startup_instance(self):
        """Start up a single instance."""
        with CudaTimer("instance_startup"):
            # Simulate container initialization
            with CudaTimer("container_init"):
                time.sleep(0.01)  # Simulated container startup

            # GPU allocation and CUDA context initialization
            if self.device.type == "cuda":
                with CudaTimer("gpu_allocation"):
                    # Allocate a small tensor to initialize CUDA context
                    dummy_tensor = torch.zeros(1024, 1024, device=self.device)
                    torch.cuda.synchronize()

                with CudaTimer("cuda_context_init"):
                    # Initialize CUDA operations
                    _ = dummy_tensor @ dummy_tensor
                    torch.cuda.synchronize()

                # Cleanup
                del dummy_tensor
                torch.cuda.empty_cache()

    def _shutdown_instance(self):
        """Shutdown a single instance."""
        self._cleanup_instance()

    def _cleanup_instance(self):
        """Clean up instance resources."""
        with CudaTimer("gpu_cleanup"):
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        with CudaTimer("container_cleanup"):
            time.sleep(0.005)  # Simulated cleanup

    def _parallel_startup(self, num_instances: int):
        """Start up multiple instances in parallel using Ray."""
        # Create a simple actor class for simulation
        @ray.remote(num_cpus=0.1, num_gpus=0.1)
        class MockInstance:
            def __init__(self):
                self.device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )

            def initialize(self):
                if self.device.type == "cuda":
                    dummy = torch.zeros(100, 100, device=self.device)
                    torch.cuda.synchronize()
                    del dummy
                return True

            def cleanup(self):
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                return True

        # Start instances in parallel
        for _ in range(num_instances):
            actor = MockInstance.remote()
            self.active_instances.append(actor)

        # Wait for all to initialize
        ray.get([actor.initialize.remote() for actor in self.active_instances])

    def _parallel_shutdown(self):
        """Shutdown all active instances in parallel."""
        if self.active_instances:
            # Cleanup all instances
            ray.get([actor.cleanup.remote() for actor in self.active_instances])

            # Kill actors
            for actor in self.active_instances:
                ray.kill(actor)

            self.active_instances = []
