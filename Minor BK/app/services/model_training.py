from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.services.eeg_dataset import EEGDatasetService, FEATURE_VECTOR_LENGTH

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "stress_random_forest.joblib"
MODEL_METADATA_PATH = ARTIFACT_DIR / "stress_model_metadata.json"
MODEL_VERSION = "spectral-bandpower-v3"


def train_and_save_model(
    dataset_service: EEGDatasetService | None = None,
    output_path: Path = MODEL_ARTIFACT_PATH,
) -> ExtraTreesClassifier:
    service = dataset_service or EEGDatasetService()
    features, labels = service.build_training_data()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = ExtraTreesClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))
    metadata = {
        "model_version": MODEL_VERSION,
        "validation_accuracy": round(float(accuracy), 4),
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "class_labels": sorted(int(label) for label in set(labels.tolist())),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, output_path)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Saved model to {output_path}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Feature count: {FEATURE_VECTOR_LENGTH}")
    return model
