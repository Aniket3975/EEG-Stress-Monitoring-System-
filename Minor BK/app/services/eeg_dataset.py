from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
from scipy.io import loadmat

from app.schemas.stress import EEGFeatures

DATASET_DIR = Path(__file__).resolve().parents[2] / "Preprocessed_EEG"
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
FEATURE_CACHE_PATH = ARTIFACT_DIR / "eeg_feature_cache.joblib"
FEATURE_CACHE_VERSION = "spectral-cache-v1"
LABEL_MAP = {-1: "negative", 0: "neutral", 1: "positive"}
SAMPLING_RATE_HZ = 200
BAND_SPECS = (
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 50),
)
CHANNEL_COUNT = 62
FEATURE_VECTOR_LENGTH = (CHANNEL_COUNT * len(BAND_SPECS)) + (4 * len(BAND_SPECS)) + len(BAND_SPECS)


@dataclass(frozen=True)
class EEGDatasetSample:
    eeg_features: EEGFeatures
    feature_vector: np.ndarray
    label: int
    emotion_label: str
    subject_id: int
    trial_id: int
    source_file: str


class EEGDatasetService:
    """Load the bundled EEG dataset and iterate across every subject/trial."""

    def __init__(self, dataset_dir: Path = DATASET_DIR) -> None:
        self.dataset_dir = dataset_dir
        self._samples = self._load_or_build_samples()
        self._lock = Lock()
        self._cursor = 0

    def has_data(self) -> bool:
        return bool(self._samples)

    def sample_count(self) -> int:
        return len(self._samples)

    def get_sample(self, index: int | None = None, advance: bool = False) -> tuple[int, EEGDatasetSample]:
        if not self._samples:
            raise RuntimeError("No EEG dataset samples are available.")

        with self._lock:
            if index is not None:
                self._cursor = index % len(self._samples)
            elif advance:
                self._cursor = (self._cursor + 1) % len(self._samples)

            sample = self._samples[self._cursor]
            return self._cursor, sample

    def _load_or_build_samples(self) -> list[EEGDatasetSample]:
        cached_samples = self._load_cached_samples()
        if cached_samples is not None:
            return cached_samples

        samples = self._load_samples()
        self._save_cached_samples(samples)
        return samples

    def _load_cached_samples(self) -> list[EEGDatasetSample] | None:
        if not FEATURE_CACHE_PATH.exists():
            return None

        payload = joblib.load(FEATURE_CACHE_PATH)
        if not isinstance(payload, dict):
            return None
        if payload.get("cache_version") != FEATURE_CACHE_VERSION:
            return None

        samples = payload.get("samples")
        if not isinstance(samples, list):
            return None

        return samples

    @staticmethod
    def _save_cached_samples(samples: list[EEGDatasetSample]) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cache_version": FEATURE_CACHE_VERSION, "samples": samples}, FEATURE_CACHE_PATH)

    def _load_samples(self) -> list[EEGDatasetSample]:
        label_path = self.dataset_dir / "label.mat"
        if not label_path.exists():
            return []

        labels = loadmat(label_path)["label"].flatten().astype(int).tolist()
        samples: list[EEGDatasetSample] = []

        for mat_path in sorted(self.dataset_dir.glob("*.mat")):
            if mat_path.name == "label.mat":
                continue

            subject_id = int(mat_path.stem.split("_", maxsplit=1)[0])
            mat_data = loadmat(mat_path)

            for trial_id, label in enumerate(labels, start=1):
                trial_key = self._find_trial_key(mat_data, trial_id)
                if not trial_key:
                    continue

                signal = np.asarray(mat_data[trial_key], dtype=np.float64)
                eeg_features, feature_vector = self._extract_feature_bundle(signal)
                samples.append(
                    EEGDatasetSample(
                        eeg_features=eeg_features,
                        feature_vector=feature_vector,
                        label=label,
                        emotion_label=LABEL_MAP.get(label, "unknown"),
                        subject_id=subject_id,
                        trial_id=trial_id,
                        source_file=mat_path.name,
                    )
                )

        return samples

    @staticmethod
    def _find_trial_key(mat_data: dict[str, object], trial_id: int) -> str | None:
        for key in mat_data:
            if key.startswith("__"):
                continue

            numeric_suffix = "".join(character for character in key if character.isdigit())
            if numeric_suffix and int(numeric_suffix) == trial_id:
                return key

        return None

    @staticmethod
    def _extract_feature_bundle(signal: np.ndarray) -> tuple[EEGFeatures, np.ndarray]:
        freqs = np.fft.rfftfreq(signal.shape[1], d=1 / SAMPLING_RATE_HZ)
        power_spectrum = np.abs(np.fft.rfft(signal, axis=1)) ** 2

        channel_band_vectors: list[np.ndarray] = []
        display_band_values: dict[str, float] = {}

        for band_name, low_hz, high_hz in BAND_SPECS:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            band_power = np.log1p(power_spectrum[:, mask].mean(axis=1))
            channel_band_vectors.append(band_power)
            display_band_values[band_name] = float(np.mean(band_power))

        band_matrix = np.stack(channel_band_vectors, axis=1)
        feature_vector = np.concatenate(
            [
                band_matrix.reshape(-1),
                band_matrix.mean(axis=0),
                band_matrix.std(axis=0),
                np.percentile(band_matrix, 25, axis=0),
                np.percentile(band_matrix, 75, axis=0),
                band_matrix[: CHANNEL_COUNT // 2].mean(axis=0) - band_matrix[CHANNEL_COUNT // 2 :].mean(axis=0),
            ]
        ).astype(np.float32)

        total_band_power = max(sum(display_band_values.values()), 1e-9)
        normalized = {name: value / total_band_power for name, value in display_band_values.items()}

        eeg_features = EEGFeatures(
            alpha=round(normalized["alpha"], 6),
            beta=round(normalized["beta"], 6),
            theta=round(normalized["theta"], 6),
            delta=round(normalized["delta"], 6),
            gamma=round(normalized["gamma"], 6),
        )
        return eeg_features, feature_vector

    @staticmethod
    def features_to_vector(eeg_features: EEGFeatures) -> list[float]:
        values = np.array(
            [
                float(eeg_features.delta or 0.0),
                float(eeg_features.theta or 0.0),
                float(eeg_features.alpha or 0.0),
                float(eeg_features.beta or 0.0),
                float(eeg_features.gamma or 0.0),
            ],
            dtype=np.float32,
        )
        band_matrix = np.tile(values, (CHANNEL_COUNT, 1))
        feature_vector = np.concatenate(
            [
                band_matrix.reshape(-1),
                band_matrix.mean(axis=0),
                band_matrix.std(axis=0),
                np.percentile(band_matrix, 25, axis=0),
                np.percentile(band_matrix, 75, axis=0),
                band_matrix[: CHANNEL_COUNT // 2].mean(axis=0) - band_matrix[CHANNEL_COUNT // 2 :].mean(axis=0),
            ]
        )
        return feature_vector.astype(np.float32).tolist()

    def build_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        feature_rows = [sample.feature_vector for sample in self._samples]
        labels = [sample.label for sample in self._samples]
        return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.int16)


@lru_cache(maxsize=1)
def get_dataset_service() -> EEGDatasetService:
    return EEGDatasetService()
