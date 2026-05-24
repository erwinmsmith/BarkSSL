"""
BarkSSL Predictor Module
Inference and prediction utilities.
"""

import os
import torch
import numpy as np
import librosa
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from tqdm import tqdm

from ..models.canine_encoder import CanineEncoder
from ..models.classifier import EmotionClassifier
from ..utils.device import get_device
from ..data.dataset import EmotionDataset


class EmotionPredictor:
    """
    Predictor for dog bark emotion recognition.

    Loads a trained model and performs inference on audio files.
    """

    EMOTION_LABELS = ['angry', 'anxious', 'happy', 'lonely', 'sad']
    EMOTION_EMOJI = {
        'angry': '😠',
        'anxious': '😰',
        'happy': '😊',
        'lonely': '😔',
        'sad': '😢',
    }

    def __init__(
        self,
        model: Union[EmotionClassifier, CanineEncoder],
        device: Optional[torch.device] = None,
        threshold: Optional[float] = None,
    ):
        """
        Initialize predictor.

        Args:
            model: Trained model (EmotionClassifier or CanineEncoder)
            device: Device to run on
            threshold: Confidence threshold for predictions
        """
        self.model = model
        self.device = device or get_device()
        self.threshold = threshold

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_single(
        self,
        waveform: np.ndarray,
        return_probs: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict emotion for a single audio.

        Args:
            waveform: Audio waveform as numpy array
            return_probs: Whether to return class probabilities

        Returns:
            Dictionary with:
                - emotion: Predicted emotion label
                - confidence: Prediction confidence
                - probabilities: Per-class probabilities (if return_probs=True)
                - embedding: Learned embedding
        """
        # Convert to tensor and add batch dimension
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform_tensor = waveform_tensor.to(self.device)

        # Predict
        if isinstance(self.model, EmotionClassifier):
            output = self.model(waveform_tensor, return_embedding=return_probs)
            logits = output['logits']
            embedding = output.get('embedding')
        else:
            output = self.model(waveform_tensor)
            hidden_states = output['hidden_states']
            # Simple mean pooling for embedding
            embedding = hidden_states.mean(dim=1)
            # Use a simple linear head if no classifier
            logits = output.get('logits')

        # Get probabilities
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()
        confidence = probs[pred_idx]

        result = {
            'emotion': self.EMOTION_LABELS[pred_idx],
            'confidence': float(confidence),
            'label_id': int(pred_idx),
        }

        if return_probs:
            result['probabilities'] = {
                label: float(prob)
                for label, prob in zip(self.EMOTION_LABELS, probs)
            }

        if embedding is not None:
            result['embedding'] = embedding.squeeze().cpu().numpy()

        return result

    def predict_file(
        self,
        filepath: str,
        target_sr: int = 16000,
        return_probs: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict emotion for an audio file.

        Args:
            filepath: Path to audio file
            target_sr: Target sampling rate
            return_probs: Whether to return class probabilities

        Returns:
            Dictionary with predictions
        """
        # Load audio
        waveform, sr = librosa.load(filepath, sr=target_sr, mono=True)

        # Predict
        result = self.predict_single(waveform, return_probs=return_probs)
        result['filepath'] = filepath

        return result

    def predict_batch(
        self,
        file_paths: List[str],
        target_sr: int = 16000,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict emotions for multiple audio files.

        Args:
            file_paths: List of audio file paths
            target_sr: Target sampling rate
            show_progress: Show progress bar

        Returns:
            List of prediction dictionaries
        """
        results = []

        iterator = tqdm(file_paths, desc="Predicting") if show_progress else file_paths

        for filepath in iterator:
            try:
                result = self.predict_file(filepath, target_sr, return_probs=True)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to predict {filepath}: {e}")
                results.append({
                    'filepath': filepath,
                    'error': str(e),
                })

        return results

    def predict_from_directory(
        self,
        directory: str,
        extensions: tuple = ('.wav', '.mp3', '.flac'),
        target_sr: int = 16000,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict emotions for all audio files in a directory.

        Args:
            directory: Path to directory
            extensions: Audio file extensions to include
            target_sr: Target sampling rate
            show_progress: Show progress bar

        Returns:
            List of prediction dictionaries
        """
        directory = Path(directory)
        file_paths = []

        for ext in extensions:
            file_paths.extend(directory.glob(f'*{ext}'))
            file_paths.extend(directory.glob(f'**/*{ext}'))

        return self.predict_batch(
            [str(p) for p in file_paths],
            target_sr=target_sr,
            show_progress=show_progress,
        )

    def get_class_distribution(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Get class distribution from prediction results.

        Args:
            results: List of prediction dictionaries

        Returns:
            Dictionary with class counts
        """
        distribution = {label: 0 for label in self.EMOTION_LABELS}

        for result in results:
            if 'emotion' in result:
                distribution[result['emotion']] += 1

        return distribution


class BarkSSLPredictor(EmotionPredictor):
    """
    BarkSSL-specific predictor with enhanced features.
    """

    def __init__(
        self,
        encoder: CanineEncoder,
        classifier: Optional[EmotionClassifier] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize BarkSSL predictor.

        Args:
            encoder: Pretrained canine encoder
            classifier: Optional emotion classifier (uses encoder directly if None)
            device: Device to run on
        """
        model = classifier if classifier else encoder
        super().__init__(model, device)

        self.encoder = encoder
        self.classifier = classifier

    def extract_embedding(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Extract embedding for a waveform.

        Args:
            waveform: Audio waveform

        Returns:
            Embedding vector
        """
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform_tensor = waveform_tensor.to(self.device)

        with torch.no_grad():
            output = self.encoder(waveform_tensor)
            hidden_states = output['hidden_states']
            embedding = hidden_states.mean(dim=1)

        return embedding.squeeze().cpu().numpy()

    def compare_embeddings(
        self,
        waveform1: np.ndarray,
        waveform2: np.ndarray,
    ) -> float:
        """
        Compare two waveforms via embedding similarity.

        Args:
            waveform1: First waveform
            waveform2: Second waveform

        Returns:
            Cosine similarity
        """
        emb1 = self.extract_embedding(waveform1)
        emb2 = self.extract_embedding(waveform2)

        # Cosine similarity
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(cos_sim)


def load_predictor_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> EmotionPredictor:
    """
    Load predictor from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to run on

    Returns:
        EmotionPredictor instance
    """
    device = device or get_device()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    scale = config.get('model.scale', 'small')
    model_config = config.get_model_config()

    encoder = CanineEncoder(
        scale=scale,
        hidden_dim=model_config.get('hidden_dim', 384),
        num_layers=model_config.get('num_layers', 6),
        num_heads=model_config.get('num_heads', 6),
        intermediate_size=model_config.get('intermediate_size', 1536),
        dropout=model_config.get('dropout', 0.1),
        kmeans_k=config.get('kmeans_k', 100),
    )

    # Try to load classifier if available
    try:
        classifier = EmotionClassifier(encoder=encoder, num_classes=config.get('num_classes', 5))
        classifier.load_state_dict(checkpoint.get('model_state_dict', {}))
        model = classifier
    except:
        # If no classifier, use encoder directly
        encoder.load_state_dict(checkpoint.get('model_state_dict', {}))
        model = encoder

    return EmotionPredictor(model=model, device=device)


def create_predictor(
    model: Union[EmotionClassifier, CanineEncoder],
    device: Optional[torch.device] = None,
) -> EmotionPredictor:
    """
    Factory function to create predictor.

    Args:
        model: Model to use for prediction
        device: Device to run on

    Returns:
        EmotionPredictor instance
    """
    return EmotionPredictor(model=model, device=device)