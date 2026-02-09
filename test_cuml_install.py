"""
Test cuML (RAPIDS) installation and GPU availability
"""

print("="*80)
print("RAPIDS cuML GPU TEST")
print("="*80)

# Test 1: Import cuML
print("\n1. Testing cuML import...")
try:
    import cuml
    print(f"   ✓ cuML version: {cuml.__version__}")
    cuml_works = True
except (ImportError, OSError) as e:
    print(f"   ✗ cuML import failed: {type(e).__name__}")
    print(f"   Error: {e}")
    cuml_works = False

# Test 2: Try GPU training
if cuml_works:
    print("\n2. Testing GPU training...")
    try:
        import numpy as np
        from cuml.linear_model import LogisticRegression

        # Create small test dataset
        X = np.random.randn(1000, 20).astype(np.float32)
        y = np.random.randint(0, 2, 1000).astype(np.float32)

        # Train on GPU
        model = LogisticRegression(max_iter=100)
        model.fit(X, y)

        # Predict
        y_pred = model.predict(X)

        print("   ✓ GPU training successful!")
        print(f"   ✓ Model trained on {len(X)} samples, {X.shape[1]} features")
        print(f"   ✓ Predictions: {y_pred[:5]}")

    except Exception as e:
        print(f"   ✗ GPU training failed: {type(e).__name__}")
        print(f"   Error: {e}")
        cuml_works = False

# Test 3: Check backend registration
print("\n3. Testing backend registration...")
try:
    from src.backends.sklearn.estimators import CUML_AVAILABLE, register_sklearn_backend

    print(f"   CUML_AVAILABLE: {CUML_AVAILABLE}")

    if CUML_AVAILABLE:
        print("   ✓ cuML will be used for GPU logistic regression")
    else:
        print("   ⚠ sklearn (CPU) will be used as fallback")

    # Try to register
    register_sklearn_backend()
    print("   ✓ Sklearn backend registered")

except Exception as e:
    print(f"   ✗ Backend registration failed: {type(e).__name__}")
    print(f"   Error: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if cuml_works:
    print("✓ RAPIDS cuML is working correctly!")
    print("  → You can use GPU logistic regression")
    print("  → Use: configs/models/sklearn_logistic_regression.yaml")
else:
    print("✗ RAPIDS cuML is NOT working")
    print("\nTo install cuML:")
    print("  Option 1 (conda - recommended):")
    print("    conda install -c rapidsai -c conda-forge -c nvidia \\")
    print("      cuml=24.02 cuda-version=12.2")
    print("\n  Option 2 (pip):")
    print("    pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12")

print("="*80)
