"""Preprocessing utility functions."""
from typing import Dict, Any, Optional, Union

from .config import PreprocessingConfig


def create_preprocessor(
    backend: str,
    config: Optional[Union[PreprocessingConfig, Dict[str, Any]]] = None,
):
    """Create preprocessor for specified backend."""
    from .spark import SparkPreprocessor
    from .sklearn_preprocessor import SklearnPreprocessor
    
    if isinstance(config, dict):
        config = PreprocessingConfig.from_dict(config)
    elif config is None:
        config = PreprocessingConfig()
    
    if backend == "spark":
        return SparkPreprocessor(config)
    else:
        return SklearnPreprocessor(config)
