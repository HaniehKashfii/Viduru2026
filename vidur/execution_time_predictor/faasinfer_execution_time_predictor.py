"""FaaSInfer Execution Time Predictor.

This extends Viduru's execution time predictor with serverless-specific components:
1. Cold start overhead (model loading from storage)
2. Storage I/O costs
3. Instance scaling latency
4. Keep-alive/idle costs

The predictor uses the same ML-based approach as Viduru, but adds
serverless-specific metrics to the execution time calculation.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional

from vidur.config import BaseExecutionTimePredictorConfig, CacheConfig, ReplicaConfig
from vidur.entities import Batch, ExecutionTime
from vidur.entities.execution_time_predictor_request import (
    ExecutionTimePredictorRequest,
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from vidur.logger import init_logger

logger = init_logger(__name__)


class FaaSInferExecutionTimePredictor(SklearnExecutionTimePredictor):
    """Execution time predictor for serverless LLM inference (FaaSInfer).

    Extends the base Viduru predictor with:
    - Cold start overhead predictions
    - Storage tier loading times
    - Scaling operation latencies
    - Instance lifecycle costs
    """

    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        cache_config: CacheConfig,
        faasinfer_config: Optional[Dict] = None,
    ) -> None:
        # Initialize base predictor (handles compute, network, attention)
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            cache_config=cache_config,
        )

        # FaaSInfer-specific configuration
        self._faasinfer_config = faasinfer_config or {}
        self._storage_tier = self._faasinfer_config.get(
            "storage_tier", "local_ssd"
        )
        self._enable_cold_start = self._faasinfer_config.get(
            "enable_cold_start", True
        )
        self._cold_start_probability = self._faasinfer_config.get(
            "cold_start_probability", 0.1
        )
        self._idle_timeout_seconds = self._faasinfer_config.get(
            "idle_timeout_seconds", 300
        )

        # Load FaaSInfer-specific profiling data
        self._cold_start_predictions = self._load_cold_start_predictions()
        self._storage_io_predictions = self._load_storage_io_predictions()
        self._scaling_predictions = self._load_scaling_predictions()

        logger.info(f"FaaSInfer predictor initialized with storage tier: {self._storage_tier}")
        logger.info(f"Cold start enabled: {self._enable_cold_start}")

    def _load_cold_start_predictions(self) -> Dict:
        """Load cold start (model loading) time predictions."""
        predictions = {}

        # Try to load model loading profiling data
        loading_file = self._get_faasinfer_profiling_file(
            "model_loading", "loading_*.csv"
        )

        if loading_file and os.path.exists(loading_file):
            df = pd.read_csv(loading_file)

            # Filter by model and storage tier
            filtered = df[
                (df["model"] == self._replica_config.model_name)
                & (df["storage_tier"] == self._storage_tier)
                & (
                    df["num_tensor_parallel_workers"]
                    == self._replica_config.tensor_parallel_size
                )
            ]

            if not filtered.empty:
                row = filtered.iloc[0]

                # Extract cold start metrics
                predictions["total_loading_time"] = row.get(
                    "time_stats.load_model_weights.mean", 0
                )
                predictions["model_to_gpu_time"] = row.get(
                    "time_stats.model_to_gpu.mean", 0
                )
                predictions["storage_latency"] = row.get(
                    "time_stats.storage_tier_latency.mean", 0
                )

                logger.info(
                    f"Loaded cold start predictions: {predictions['total_loading_time']:.2f}ms"
                )
            else:
                logger.warning(
                    f"No cold start data found for {self._replica_config.model_name} "
                    f"with storage tier {self._storage_tier}"
                )
        else:
            logger.warning("Model loading profiling file not found")

        # Provide default cold start time if no data available
        if not predictions:
            # Rough estimate: 1-5 seconds for typical models
            predictions["total_loading_time"] = 3000  # ms
            predictions["model_to_gpu_time"] = 500  # ms
            predictions["storage_latency"] = 50  # ms
            logger.info("Using default cold start estimates")

        return predictions

    def _load_storage_io_predictions(self) -> Dict:
        """Load storage I/O performance predictions."""
        predictions = {}

        storage_io_file = self._get_faasinfer_profiling_file(
            "storage_io", "io_profile_*.csv"
        )

        if storage_io_file and os.path.exists(storage_io_file):
            df = pd.read_csv(storage_io_file)

            # Filter by storage tier
            filtered = df[df["storage_tier"] == self._storage_tier]

            if not filtered.empty:
                # Get bandwidth and latency
                row = filtered.iloc[0]
                predictions["bandwidth_mbps"] = row.get(
                    "sequential.observed_bandwidth_mbps", 3500
                )
                predictions["latency_ms"] = row.get("theoretical_latency_ms", 1)

                logger.info(
                    f"Loaded storage I/O predictions: {predictions['bandwidth_mbps']:.2f} MB/s"
                )
        else:
            # Defaults based on storage tier
            defaults = {
                "remote_blob": {"bandwidth_mbps": 1250, "latency_ms": 50},
                "local_ssd": {"bandwidth_mbps": 3500, "latency_ms": 1},
                "nvme_cache": {"bandwidth_mbps": 7000, "latency_ms": 0.1},
            }
            predictions = defaults.get(
                self._storage_tier, {"bandwidth_mbps": 3500, "latency_ms": 1}
            )
            logger.info(f"Using default storage I/O estimates for {self._storage_tier}")

        return predictions

    def _load_scaling_predictions(self) -> Dict:
        """Load scaling operation time predictions."""
        predictions = {}

        scaling_file = self._get_faasinfer_profiling_file(
            "scaling", "scaling_*.csv"
        )

        if scaling_file and os.path.exists(scaling_file):
            df = pd.read_csv(scaling_file)

            # Filter by model
            filtered = df[df["model"] == self._replica_config.model_name]

            if not filtered.empty:
                row = filtered.iloc[0]
                predictions["instance_startup_time"] = row.get(
                    "startup.instance_startup.mean", 100
                )
                predictions["instance_shutdown_time"] = row.get(
                    "shutdown.instance_shutdown.mean", 50
                )

                logger.info(
                    f"Loaded scaling predictions: startup={predictions['instance_startup_time']:.2f}ms"
                )
        else:
            # Default scaling times
            predictions["instance_startup_time"] = 100  # ms
            predictions["instance_shutdown_time"] = 50  # ms
            logger.info("Using default scaling estimates")

        return predictions

    def _get_faasinfer_profiling_file(
        self, component: str, pattern: str
    ) -> Optional[str]:
        """Get the most recent profiling file for a component."""
        base_dir = os.path.join(
            "data", "profiling", "faasinfer", component
        )

        if not os.path.exists(base_dir):
            return None

        # Find all files matching pattern
        import glob

        files = glob.glob(os.path.join(base_dir, pattern))

        if not files:
            return None

        # Return most recent file
        return max(files, key=os.path.getmtime)

    def get_batch_execution_time(
        self, batch: Batch, pipeline_stage: int
    ) -> ExecutionTime:
        """Get execution time including FaaSInfer-specific overheads."""
        # Get base execution time from Viduru predictor
        base_execution_time = super().get_batch_execution_time(batch, pipeline_stage)

        # Add FaaSInfer-specific overheads
        faasinfer_overhead = self._calculate_faasinfer_overhead(batch)

        # Combine base execution time with serverless overhead
        return self._add_serverless_overhead(base_execution_time, faasinfer_overhead)

    def _calculate_faasinfer_overhead(self, batch: Batch) -> Dict[str, float]:
        """Calculate FaaSInfer-specific overhead."""
        overhead = {
            "cold_start_time": 0.0,
            "storage_io_time": 0.0,
            "scaling_time": 0.0,
        }

        # Cold start overhead (probabilistic)
        if self._enable_cold_start:
            # In simulation, apply cold start based on probability
            # In reality, this would depend on instance state
            overhead["cold_start_time"] = (
                self._cold_start_predictions.get("total_loading_time", 0)
                * self._cold_start_probability
            )

        # Storage I/O overhead (for checkpoint access if needed)
        # This is typically only for cold starts, but we include it for completeness
        if overhead["cold_start_time"] > 0:
            overhead["storage_io_time"] = self._cold_start_predictions.get(
                "storage_latency", 0
            )

        # Scaling overhead (instance startup if new instance needed)
        # This is also probabilistic based on scaling events
        overhead["scaling_time"] = (
            self._scaling_predictions.get("instance_startup_time", 0) * 0.05
        )  # 5% probability

        return overhead

    def _add_serverless_overhead(
        self, base_execution_time: ExecutionTime, overhead: Dict[str, float]
    ) -> ExecutionTime:
        """Add serverless overhead to base execution time."""
        # Total serverless overhead
        total_overhead = sum(overhead.values())

        # Create new execution time with added overhead
        # Note: We add overhead to the overall execution time
        # In a real implementation, this might be more sophisticated
        # (e.g., cold start only affects first request)

        # For now, we'll add it to the prefill time (affects TTFT)
        new_execution_time = ExecutionTime(
            preprocess_time=base_execution_time.preprocess_time,
            model_time=base_execution_time.model_time + total_overhead,
            postprocess_time=base_execution_time.postprocess_time,
        )

        return new_execution_time

    def get_cold_start_time(self) -> float:
        """Get expected cold start time in milliseconds."""
        return self._cold_start_predictions.get("total_loading_time", 0)

    def get_storage_bandwidth_mbps(self) -> float:
        """Get storage bandwidth in MB/s."""
        return self._storage_io_predictions.get("bandwidth_mbps", 3500)

    def get_instance_startup_time(self) -> float:
        """Get instance startup time in milliseconds."""
        return self._scaling_predictions.get("instance_startup_time", 100)

    def get_serverless_metrics(self) -> Dict[str, float]:
        """Get all serverless-specific metrics."""
        return {
            "cold_start_time_ms": self.get_cold_start_time(),
            "storage_bandwidth_mbps": self.get_storage_bandwidth_mbps(),
            "storage_latency_ms": self._storage_io_predictions.get("latency_ms", 1),
            "instance_startup_time_ms": self.get_instance_startup_time(),
            "instance_shutdown_time_ms": self._scaling_predictions.get(
                "instance_shutdown_time", 50
            ),
            "storage_tier": self._storage_tier,
            "cold_start_probability": self._cold_start_probability,
        }
