"""
Test script for PyTorch Logistic Regression with incremental training.
This trains on the full dataset without loading everything into RAM.
"""

# Step 1: Reload backends to register the new logistic regression model
import importlib
import sys

modules_to_reload = [
    'src.backends.pytorch.estimators',
    'src.backends.pytorch',
    'src.backends',
]

for mod in modules_to_reload:
    if mod in sys.modules:
        del sys.modules[mod]

from src.backends import initialize_backends
initialize_backends()

print("✅ Backends reloaded and PyTorch Logistic Regression registered!")

# Step 2: Run experiment with incremental training
from run import run_experiments

print("\n" + "="*80)
print("Training PyTorch Logistic Regression with INCREMENTAL LEARNING")
print("This will process data in 1M row chunks - no memory crash!")
print("="*80 + "\n")

results = run_experiments(
    df=df2,  # Your full 88.5M row dataset
    experiment_name="pytorch_lr_incremental",
    runs=[
        ("configs/data/socioec_features_med_ext.yaml",
         "configs/models/pytorch_logistic_regression.yaml",
         "configs/datasets/default.yaml"),  # Uses full dataset now!
    ],
    mode="final",
    enable_hyperparameter_tuning=False,
)

print("\n✅ Training complete!")
