"""Sampling configuration classes."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SamplingStrategy(Enum):
    NONE = "none"
    RANDOM = "random"
    STRATIFIED = "stratified"
    UNDERSAMPLE = "undersample"
    OVERSAMPLE = "oversample"
    HARD_NEGATIVE = "hard_negative"
    HARD_NEGATIVE_MIXED = "hard_negative_mixed"


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies."""
    
    strategy: SamplingStrategy = SamplingStrategy.NONE
    
    # General sampling
    fraction: float = 1.0
    seed: int = 42
    
    # Class-specific
    target_ratio: Optional[float] = None
    min_samples: int = 100
    
    # Hard negative mining
    hard_negative_fraction: float = 0.5
    hard_negative_percentile: float = 0.8
    hard_negative_method: str = "top_prob"  # top_prob, margin, entropy
    
    # Model for scoring
    scorer_model_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingConfig":
        # Accept both "type" and "strategy" keys
        strategy_str = d.get("type", d.get("strategy", "none"))
        
        # Handle case where strategy_str might be None
        if strategy_str is None:
            strategy_str = "none"
        
        # Convert to enum with error handling
        try:
            strategy = SamplingStrategy(strategy_str)
        except ValueError:
            valid_strategies = [s.value for s in SamplingStrategy]
            raise ValueError(
                f"Unknown sampling strategy: '{strategy_str}'. "
                f"Valid options are: {valid_strategies}"
            )
        
        return cls(
            strategy=strategy,
            fraction=d.get("fraction", 1.0),
            seed=d.get("seed", 42),
            target_ratio=d.get("target_ratio"),
            min_samples=d.get("min_samples", 100),
            hard_negative_fraction=d.get("hard_negative_fraction", 0.5),
            hard_negative_percentile=d.get("hard_negative_percentile", 0.8),
            hard_negative_method=d.get("hard_negative_method", "top_prob"),
        )
