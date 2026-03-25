from __future__ import annotations

from pathlib import Path
from math import tanh
from typing import Any

import joblib

from app.schemas.stress import EEGFeatures
from app.services.eeg_dataset import (
    EEGDatasetService,
    FEATURE_VECTOR_LENGTH,
    LABEL_MAP,
    get_dataset_service,
)
from app.services.model_training import MODEL_ARTIFACT_PATH, MODEL_VERSION, train_and_save_model
from app.utils.scoring import classify_stress


class StressModelService:
    """Load and run the trained stress model."""

    def __init__(self, model_path: Path = MODEL_ARTIFACT_PATH) -> None:
        self.model_path = model_path
        self.dataset_service = get_dataset_service()
        self.model = self._load_or_train_model()

    def predict(
        self,
        eeg_features: EEGFeatures | None = None,
        feature_vector: list[float] | None = None,
    ) -> dict[str, int | str | float]:
        """Predict emotion-driven stress using the trained model."""

        features = eeg_features or EEGFeatures()
        resolved_vector = feature_vector or EEGDatasetService.features_to_vector(features)
        model_input = [resolved_vector]
        predicted_label = int(self.model.predict(model_input)[0])
        probabilities = {
            int(label): float(probability)
            for label, probability in zip(self.model.classes_, self.model.predict_proba(model_input)[0])
        }
        score = self._probabilities_to_score(probabilities)
        result = classify_stress(score)
        confidence = max(probabilities.values(), default=0.0)

        return {
            **result.__dict__,
            "emotion_label": LABEL_MAP.get(predicted_label, "unknown"),
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
        }

    def predict_live(self, eeg_features: EEGFeatures | None = None) -> dict[str, int | str | float]:
        """Score live ESP-derived band features directly so output responds to signal changes."""

        features = eeg_features or EEGFeatures()
        alpha = float(features.alpha or 0.0)
        beta = float(features.beta or 0.0)
        theta = float(features.theta or 0.0)
        delta = float(features.delta or 0.0)
        gamma = float(features.gamma or 0.0)

        activation = (beta * 1.15) + (gamma * 0.95)
        calming = (alpha * 1.1) + (theta * 0.9)
        heaviness = delta * 0.75
        balance = activation - calming
        composite_signal = (balance * 2.4) + (heaviness * 1.2)
        bounded_score = int(round(50 + (tanh(composite_signal) * 34)))
        bounded_score = max(0, min(100, bounded_score))
        result = classify_stress(bounded_score)

        if bounded_score >= 67:
            emotion_label = "negative"
        elif bounded_score <= 38:
            emotion_label = "positive"
        else:
            emotion_label = "neutral"

        contrast = abs(balance) + abs(delta - alpha) + abs(gamma - beta)
        confidence = max(0.2, min(0.96, 0.42 + (contrast * 0.9)))

        return {
            **result.__dict__,
            "emotion_label": emotion_label,
            "predicted_label": None,
            "confidence": round(confidence, 4),
        }

    def _load_or_train_model(self):
        if not self.model_path.exists():
            return train_and_save_model(self.dataset_service, self.model_path)

        artifact = joblib.load(self.model_path)
        model, metadata = self._resolve_artifact(artifact)

        if metadata.get("model_version") != MODEL_VERSION:
            return train_and_save_model(self.dataset_service, self.model_path)
        if int(getattr(model, "n_features_in_", -1)) != FEATURE_VECTOR_LENGTH:
            return train_and_save_model(self.dataset_service, self.model_path)

        return model

    @staticmethod
    def _resolve_artifact(artifact: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact["model"], artifact.get("metadata", {})
        return artifact, {}

    @staticmethod
    def _probabilities_to_score(probabilities: dict[int, float]) -> int:
        weighted_score = (
            probabilities.get(-1, 0.0) * 86
            + probabilities.get(0, 0.0) * 52
            + probabilities.get(1, 0.0) * 18
        )
        confidence_bonus = max(probabilities.values(), default=0.0) * 8
        return int(round(weighted_score + confidence_bonus))
