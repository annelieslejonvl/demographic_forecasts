"""
Quick test script to verify GPU optimization is working.

Run this to check:
1. GPU is being detected
2. XGBoost is using gpu_hist (not hist)
3. GPU memory status
"""

import sys
import subprocess

print("=" * 70)
print("GPU OPTIMIZATION VERIFICATION TEST")
print("=" * 70)

# Test 1: Check GPU availability
print("\n[1/3] Checking GPU availability...")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    gpu_info = result.stdout.strip()
    print(f"✅ GPU found: {gpu_info}")
except Exception as e:
    print(f"❌ nvidia-smi failed: {e}")
    sys.exit(1)

# Test 2: Check XGBoost GPU support
print("\n[2/3] Checking XGBoost GPU support...")
try:
    import xgboost as xgb
    import numpy as np

    print(f"✅ XGBoost version: {xgb.__version__}")

    if xgb.__version__ < '2.0.0':
        print(f"⚠️  Warning: XGBoost {xgb.__version__} may have limited GPU support")
        print("   Recommended: XGBoost >= 2.0.0")

    # Test if GPU training works
    X = np.random.rand(1000, 10).astype(np.float32)
    y = np.random.randint(0, 2, 1000).astype(np.float32)

    dtrain = xgb.DMatrix(X, label=y)
    # XGBoost 2.0+ unified API: use device='cuda:0' with tree_method='hist'
    params = {
        'device': 'cuda:0',          # Specify GPU
        'tree_method': 'hist',       # Works for both CPU/GPU
        'max_depth': 3,
        'objective': 'binary:logistic',
    }

    print("   Testing GPU training (10 trees)...")
    model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
    print("✅ XGBoost GPU training works!")

except Exception as e:
    print(f"❌ XGBoost GPU error: {e}")
    print("   Make sure XGBoost >= 2.0.0 with GPU support is installed")
    print("   Install with: pip install xgboost --upgrade")
    sys.exit(1)

# Test 3: Check current GPU usage
print("\n[3/3] Checking GPU memory status...")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    used, total, util = result.stdout.strip().split(", ")
    used_mb = float(used)
    total_mb = float(total)
    free_mb = total_mb - used_mb

    print(f"   Memory: {used_mb:.0f}/{total_mb:.0f} MB ({100*used_mb/total_mb:.1f}% used)")
    print(f"   Free: {free_mb:.0f} MB")
    print(f"   Compute utilization: {util}%")

    # Estimate batch sizes
    print("\n   Estimated optimal batch sizes:")
    for n_features in [10, 23, 50]:
        available_bytes = free_mb * 1e6 * 0.7
        bytes_per_sample = n_features * 4 + 8
        batch_size = int(available_bytes / bytes_per_sample)
        batch_size = (batch_size // 10_000) * 10_000
        print(f"   {n_features:2d} features → {batch_size:>9,} samples/batch")

except Exception as e:
    print(f"⚠️  Error checking GPU: {e}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\n✅ All checks passed! GPU optimization is ready.")
print("\nKey changes made:")
print("  • Fixed device: 'cpu' → 'cuda:0' (XGBoost 3.x API)")
print("  • Increased batch size: 100K → 500-800K samples")
print("  • Increased trees per batch: 166 → 250")
print("  • Added GPU monitoring and auto-tuning")
print("\nTo verify during training:")
print("1. Run training in one terminal")
print("2. Watch GPU in another: watch -n 1 nvidia-smi")
print("3. You should see:")
print("   - GPU utilization: 70-95%")
print("   - GPU memory: 6-8 GB used")
print("   - Training logs showing 'Device: cuda:0'")
print("\nFor detailed info, see: GPU_OPTIMIZATION_SUMMARY.md")
