"""FaaSInfer-specific configuration for profiling serverless LLM inference.

This extends the base Viduru profiling with serverless-specific parameters.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FaaSInferConfig:
    """Configuration for FaaSInfer profiling."""

    # Cold start parameters
    enable_cold_start_profiling: bool = True
    checkpoint_format: str = "safetensors"  # or "pytorch", "custom"

    # Model loading parameters
    use_streaming_load: bool = True  # Load while executing
    loading_parallelism: int = 4  # Parallel loading threads

    # Scaling parameters
    min_instances: int = 0  # Serverless can scale to zero
    max_instances: int = 100
    scale_up_threshold: float = 0.7  # CPU/GPU utilization threshold
    scale_down_threshold: float = 0.3

    # Migration parameters
    enable_live_migration: bool = True
    migration_chunk_size_mb: int = 64  # Chunk size for incremental migration

    # Storage tiers to profile
    storage_tiers: List[str] = None

    # Keep-alive parameters
    idle_timeout_seconds: int = 300  # 5 minutes default

    # Billing granularity (for cost modeling)
    billing_granularity_ms: int = 1000  # 1 second billing

    def __post_init__(self):
        if self.storage_tiers is None:
            self.storage_tiers = ["remote_blob", "local_ssd", "nvme_cache"]
