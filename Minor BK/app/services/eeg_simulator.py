import random

from app.schemas.stress import EEGFeatures


def generate_simulated_eeg_features() -> EEGFeatures:
    """Generate placeholder EEG features until the real pipeline is connected."""

    return EEGFeatures(
        alpha=round(random.uniform(0.1, 1.0), 3),
        beta=round(random.uniform(0.1, 1.0), 3),
        theta=round(random.uniform(0.1, 1.0), 3),
        delta=round(random.uniform(0.1, 1.0), 3),
        gamma=round(random.uniform(0.1, 1.0), 3),
    )


def generate_simulated_score() -> int:
    """Create a random stress score for demo mode."""

    return random.randint(0, 100)

