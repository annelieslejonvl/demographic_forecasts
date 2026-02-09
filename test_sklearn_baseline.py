"""
Test Sklearn Logistic Regression Baseline
Uses incremental training (SGDClassifier) - works on CPU without crashes
"""

print("="*80)
print("SKLEARN LOGISTIC REGRESSION BASELINE")
print("="*80)

# Step 1: Reload backends
import importlib
import sys

modules_to_reload = [
    'src.backends.sklearn.estimators',
    'src.backends.sklearn',
    'src.backends',
]

for mod in modules_to_reload:
    if mod in sys.modules:
        del sys.modules[mod]

from src.backends import initialize_backends
initialize_backends()

print("\n✅ Sklearn backend registered!")

# Step 2: Check backend availability
from src.backends.base import BackendFactory, BackendType

try:
    models = BackendFactory.list_models(BackendType.SKLEARN)
    print(f"✅ Sklearn models available: {', '.join(models)}")
except:
    print("❌ Sklearn backend not registered")
    print("   Make sure sklearn backend is properly initialized")

# Step 3: Run experiment
print("\n" + "="*80)
print("TRAINING LOGISTIC REGRESSION BASELINE")
print("Using incremental SGD - CPU only, no GPU needed")
print("="*80 + "\n")

from run import run_experiments

results = run_experiments(
    df=df2,  # Your full dataset
    experiment_name="sklearn_lr_baseline",
    runs=[
        ("configs/data/socioec_features_med_ext.yaml",
         "configs/models/sklearn_logistic_regression.yaml",
         "configs/datasets/default.yaml"),
    ],
    mode="final",
    enable_hyperparameter_tuning=False,
)

print("\n" + "="*80)
print("✅ BASELINE COMPLETE!")
print("="*80)
print("\nResults:")
print(f"  Model: Logistic Regression (SGDClassifier)")
print(f"  Backend: sklearn (CPU)")
print(f"  Training: Incremental (batch-by-batch)")
