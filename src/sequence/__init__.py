"""
LLM-style sequence model for demographic event prediction.

Converts panel data (person x year) into token sequences and uses
encoder (LSTM/GRU/Transformer) + prediction head architecture to
predict event probabilities at multiple time horizons.
"""
from .vocabulary import LifeEventVocabulary
from .dataset import SequenceDataset, StreamingSequenceDataset, CachedSequenceDataset, sequence_collate_fn
from .models import SequenceModel
from .estimator import PyTorchSequenceEstimator
from .evaluation import evaluate_sequence_predictions

DEFAULT_EVENTS = [
    'y_moved',
    'birth1_event',
    'birth2_event',
    'divorce_event',
    'getalifeother_event',
]

DEFAULT_HORIZONS = [1, 3, 5]

__all__ = [
    "LifeEventVocabulary",
    "SequenceDataset",
    "StreamingSequenceDataset",
    "CachedSequenceDataset",
    "sequence_collate_fn",
    "SequenceModel",
    "PyTorchSequenceEstimator",
    "evaluate_sequence_predictions",
    "DEFAULT_EVENTS",
    "DEFAULT_HORIZONS",
]
