"""
Device detection and management utilities.
Handles GPU availability checks for PyTorch, XGBoost, and Spark RAPIDS.
"""
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about an available GPU."""
    index: int
    name: str
    memory_total: int  # bytes
    memory_free: int   # bytes
    compute_capability: Optional[str] = None


class DeviceManager:
    """Centralized device detection and management."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not DeviceManager._initialized:
            self._cuda_available: Optional[bool] = None
            self._gpu_info: Optional[List[GPUInfo]] = None
            self._torch_available: Optional[bool] = None
            self._rapids_available: Optional[bool] = None
            DeviceManager._initialized = True
    
    def check_cuda_available(self) -> bool:
        """Check if CUDA is available (caches result)."""
        if self._cuda_available is not None:
            return self._cuda_available
        
        # Try multiple methods
        self._cuda_available = False
        
        # Method 1: PyTorch
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                logger.info(f"CUDA available via PyTorch: {torch.cuda.device_count()} GPU(s)")
                return self._cuda_available
        except ImportError:
            pass
        
        # Method 2: nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._cuda_available = True
                logger.info("CUDA available via nvidia-smi")
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return self._cuda_available
    
    def get_gpu_info(self) -> List[GPUInfo]:
        """Get information about available GPUs."""
        if self._gpu_info is not None:
            return self._gpu_info
        
        self._gpu_info = []
        
        if not self.check_cuda_available():
            return self._gpu_info
        
        # Try PyTorch first
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self._gpu_info.append(GPUInfo(
                    index=i,
                    name=props.name,
                    memory_total=props.total_memory,
                    memory_free=props.total_memory,  # Approximate
                    compute_capability=f"{props.major}.{props.minor}",
                ))
            return self._gpu_info
        except ImportError:
            pass
        
        # Fallback: nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        self._gpu_info.append(GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            memory_total=int(parts[2]) * 1024 * 1024,  # MiB to bytes
                            memory_free=int(parts[3]) * 1024 * 1024,
                        ))
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass
        
        return self._gpu_info
    
    def get_best_gpu(self) -> Optional[int]:
        """Get the GPU with most free memory."""
        gpus = self.get_gpu_info()
        if not gpus:
            return None
        return max(gpus, key=lambda g: g.memory_free).index
    
    def check_torch_available(self) -> bool:
        """Check if PyTorch is available."""
        if self._torch_available is not None:
            return self._torch_available
        try:
            import torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False
        return self._torch_available
    
    def check_rapids_available(self) -> bool:
        """Check if RAPIDS (cuDF, cuML) is available for Spark GPU."""
        if self._rapids_available is not None:
            return self._rapids_available
        try:
            import cudf
            import cuml
            self._rapids_available = True
        except ImportError:
            self._rapids_available = False
        return self._rapids_available
    
    def get_torch_device(self, gpu_id: Optional[int] = None) -> str:
        """Get PyTorch device string."""
        if not self.check_cuda_available():
            return "cpu"
        if gpu_id is not None:
            return f"cuda:{gpu_id}"
        best = self.get_best_gpu()
        return f"cuda:{best}" if best is not None else "cuda:0"
    
    def get_xgboost_params(self, use_gpu: bool = True, gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """Get XGBoost device parameters."""
        if not use_gpu or not self.check_cuda_available():
            return {"device": "cpu", "tree_method": "hist"}
        
        gpu = gpu_id if gpu_id is not None else (self.get_best_gpu() or 0)
        return {
            "device": f"cuda:{gpu}",
            "tree_method": "hist",  # gpu_hist is deprecated, hist auto-selects
        }
    
    def configure_spark_rapids(self, spark_builder: Any, use_gpu: bool = True) -> Any:
        """Configure SparkSession for RAPIDS GPU acceleration."""
        if not use_gpu or not self.check_rapids_available():
            logger.info("Using Spark CPU mode")
            return spark_builder
        
        logger.info("Configuring Spark for RAPIDS GPU acceleration")
        return (
            spark_builder
            .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.rapids.sql.enabled", "true")
            .config("spark.rapids.memory.pinnedPool.size", "2G")
            .config("spark.sql.files.maxPartitionBytes", "512m")
            .config("spark.sql.shuffle.partitions", "200")
        )
    
    def print_device_summary(self):
        """Print a summary of available devices."""
        print("=" * 60)
        print("DEVICE SUMMARY")
        print("=" * 60)
        
        print(f"CUDA Available: {self.check_cuda_available()}")
        print(f"PyTorch Available: {self.check_torch_available()}")
        print(f"RAPIDS Available: {self.check_rapids_available()}")
        
        gpus = self.get_gpu_info()
        if gpus:
            print(f"\nGPUs ({len(gpus)} found):")
            for gpu in gpus:
                mem_gb = gpu.memory_total / (1024**3)
                print(f"  [{gpu.index}] {gpu.name} - {mem_gb:.1f} GB")
        else:
            print("\nNo GPUs found")
        
        print("=" * 60)


# Global singleton instance
device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """Get the global DeviceManager instance."""
    return device_manager
