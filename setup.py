# demographic_forecasts/setup.py
from setuptools import setup, find_packages

setup(
    name="demographic_forecasts",
    version="0.5.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "xgboost>=1.7",
        "pyyaml>=6.0",
        "joblib>=1.2",
    ],
    extras_require={
        "spark": ["pyspark>=3.3"],
        "torch": ["torch>=2.0"],
    },
)