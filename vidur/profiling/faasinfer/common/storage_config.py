"""Storage tier configuration for FaaSInfer profiling.

This module defines different storage tiers for serverless LLM inference:
- Remote Storage (S3, Azure Blob, GCS) - Slowest, cheapest
- Local SSD - Medium speed
- GPU Memory - Fastest (for live instances)
- NVMe/Local Cache - Fast local storage
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class StorageTier(Enum):
    """Storage tiers for serverless model checkpoint loading."""

    REMOTE_BLOB = "remote_blob"  # S3, Azure Blob, GCS
    LOCAL_SSD = "local_ssd"  # Local SSD storage
    NVME_CACHE = "nvme_cache"  # Fast NVMe local cache
    GPU_MEMORY = "gpu_memory"  # Already loaded in GPU memory

    def __str__(self):
        return self.value


@dataclass
class StorageConfig:
    """Configuration for storage tier profiling."""

    tier: StorageTier
    path: str
    # Simulated or actual bandwidth (MB/s)
    bandwidth_mbps: Optional[float] = None
    # Simulated or actual latency (ms)
    latency_ms: Optional[float] = None

    @staticmethod
    def get_default_configs():
        """Get typical storage tier configurations."""
        return [
            StorageConfig(
                tier=StorageTier.REMOTE_BLOB,
                path="/tmp/remote_blob",
                bandwidth_mbps=1250,  # ~10 Gbps network
                latency_ms=50,
            ),
            StorageConfig(
                tier=StorageTier.LOCAL_SSD,
                path="/tmp/local_ssd",
                bandwidth_mbps=3500,  # ~3.5 GB/s
                latency_ms=1,
            ),
            StorageConfig(
                tier=StorageTier.NVME_CACHE,
                path="/tmp/nvme_cache",
                bandwidth_mbps=7000,  # ~7 GB/s
                latency_ms=0.1,
            ),
        ]
