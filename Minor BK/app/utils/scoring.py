from dataclasses import dataclass


@dataclass(frozen=True)
class StressResult:
    """Normalized stress output shared across the application."""

    stress_level: str
    color: str
    score: int


def clamp_score(score: float) -> int:
    """Clamp a numeric score into the supported 0-100 range."""

    return max(0, min(100, int(round(score))))


def classify_stress(score: float) -> StressResult:
    """Map a score to a stress label and UI color."""

    normalized_score = clamp_score(score)

    if normalized_score <= 35:
        return StressResult(stress_level="Low", color="green", score=normalized_score)
    if normalized_score <= 70:
        return StressResult(stress_level="Medium", color="yellow", score=normalized_score)
    return StressResult(stress_level="High", color="red", score=normalized_score)

