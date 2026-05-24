"""
BarkSSL Model Module
Exports all model classes.
"""

from .base import BaseModel, TrainableModel, CompositeModel
from .canine_encoder import CanineEncoder, CanineWavLMEncoder, create_canine_encoder, count_parameters
from .classifier import (
    EmotionClassifier,
    WeightedEmotionClassifier,
    TransformerClassifier,
    create_emotion_classifier,
)
from .components import (
    CNNEncoder,
    TransformerEncoder,
    TransformerEncoderLayer,
    MeanPooling,
    AttentiveStatisticsPooling,
    RelativePositionBias,
    PositionalEncoding,
)

__all__ = [
    'BaseModel',
    'TrainableModel',
    'CompositeModel',
    'CanineEncoder',
    'CanineWavLMEncoder',
    'create_canine_encoder',
    'count_parameters',
    'EmotionClassifier',
    'WeightedEmotionClassifier',
    'TransformerClassifier',
    'create_emotion_classifier',
    'CNNEncoder',
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'MeanPooling',
    'AttentiveStatisticsPooling',
    'RelativePositionBias',
    'PositionalEncoding',
]