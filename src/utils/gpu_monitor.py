"""
GPU monitoring utilities for optimizing VRAM usage during training.
"""
import subprocess
import time
from typing import Dict, Optional


def get_gpu_memory_usage(gpu_id: int = 0) -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dict with 'used_mb', 'total_mb', 'free_mb', 'utilization_pct'
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        used, total, util = result.stdout.strip().split(", ")
        used_mb = float(used)
        total_mb = float(total)

        return {
            "used_mb": used_mb,
            "total_mb": total_mb,
            "free_mb": total_mb - used_mb,
            "utilization_pct": float(util),
            "used_pct": (used_mb / total_mb) * 100,
        }
    except Exception as e:
        return {
            "used_mb": 0,
            "total_mb": 0,
            "free_mb": 0,
            "utilization_pct": 0,
            "used_pct": 0,
            "error": str(e),
        }


def print_gpu_usage(gpu_id: int = 0, prefix: str = ""):
    """Print formatted GPU usage."""
    stats = get_gpu_memory_usage(gpu_id)

    if "error" in stats:
        print(f"{prefix}GPU monitoring unavailable: {stats['error']}")
        return

    print(
        f"{prefix}GPU {gpu_id}: "
        f"{stats['used_mb']:.0f}/{stats['total_mb']:.0f} MB "
        f"({stats['used_pct']:.1f}% memory, "
        f"{stats['utilization_pct']:.0f}% compute)"
    )


class GPUMonitor:
    """
    Context manager to monitor GPU usage during training.

    Example:
        with GPUMonitor(gpu_id=0, interval=5) as monitor:
            model.fit(X, y)

        print(f"Peak GPU usage: {monitor.peak_mb} MB")
    """

    def __init__(self, gpu_id: int = 0, interval: float = 5.0, verbose: bool = True):
        self.gpu_id = gpu_id
        self.interval = interval
        self.verbose = verbose
        self.peak_mb = 0
        self.peak_util = 0
        self.samples = []
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()
        if self.verbose:
            print("=" * 60)
            print("Starting GPU monitoring...")
            print_gpu_usage(self.gpu_id, "Baseline: ")
            print("=" * 60)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start_time

        if self.verbose:
            print("=" * 60)
            print(f"GPU monitoring complete (elapsed: {elapsed:.1f}s)")
            print_gpu_usage(self.gpu_id, "Final: ")
            print(f"Peak memory: {self.peak_mb:.0f} MB ({self.peak_util:.1f}% compute)")

            if self.samples:
                avg_mem = sum(s["used_mb"] for s in self.samples) / len(self.samples)
                avg_util = sum(s["utilization_pct"] for s in self.samples) / len(self.samples)
                print(f"Average: {avg_mem:.0f} MB memory, {avg_util:.0f}% compute")

            print("=" * 60)

    def sample(self):
        """Manually sample GPU usage (call during training)."""
        stats = get_gpu_memory_usage(self.gpu_id)

        if "error" not in stats:
            self.peak_mb = max(self.peak_mb, stats["used_mb"])
            self.peak_util = max(self.peak_util, stats["utilization_pct"])
            self.samples.append(stats)

            if self.verbose:
                print_gpu_usage(self.gpu_id, "  → ")


def estimate_batch_size_for_gpu(
    n_features: int,
    gpu_memory_gb: float,
    safety_factor: float = 0.6,
) -> int:
    """
    Estimate optimal batch size for GPU based on available memory.

    Args:
        n_features: Number of input features
        gpu_memory_gb: Available GPU memory in GB
        safety_factor: Use only this fraction of GPU memory (default 0.6)

    Returns:
        Recommended batch size

    Example:
        # RTX 3080 with 10GB, 30 features
        batch_size = estimate_batch_size_for_gpu(30, 10.0)
        # Returns: ~500,000
    """
    # XGBoost GPU memory usage approximation:
    # - Input data: n_samples * n_features * 4 bytes (float32)
    # - Gradient/Hessian: n_samples * 8 bytes
    # - Tree structures: ~50-100 MB per tree (varies with depth)
    # - GPU page cache: configurable (default 32MB)

    available_bytes = gpu_memory_gb * 1e9 * safety_factor

    # Reserve memory for tree structures and overhead (~1GB)
    available_for_data = available_bytes - 1e9

    # Bytes per sample: features (float32) + gradients (float64)
    bytes_per_sample = n_features * 4 + 8

    # Calculate batch size
    batch_size = int(available_for_data / bytes_per_sample)

    # Round to nice number
    if batch_size > 1_000_000:
        batch_size = (batch_size // 100_000) * 100_000
    elif batch_size > 100_000:
        batch_size = (batch_size // 10_000) * 10_000
    else:
        batch_size = (batch_size // 1_000) * 1_000

    return max(10_000, batch_size)  # Minimum 10K samples


if __name__ == "__main__":
    # Test GPU monitoring
    print("Testing GPU monitoring...")
    print_gpu_usage(0)

    print("\nEstimated batch sizes:")
    for n_features in [10, 30, 50, 100]:
        batch_size = estimate_batch_size_for_gpu(n_features, 10.0)
        print(f"  {n_features} features → {batch_size:,} samples/batch")
