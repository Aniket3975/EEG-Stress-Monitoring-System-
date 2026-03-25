from pydantic import BaseModel, Field


class EEGFeatures(BaseModel):
    """Optional EEG feature payload for future real-hardware integration."""

    alpha: float | None = Field(default=None, description="Alpha band feature")
    beta: float | None = Field(default=None, description="Beta band feature")
    theta: float | None = Field(default=None, description="Theta band feature")
    delta: float | None = Field(default=None, description="Delta band feature")
    gamma: float | None = Field(default=None, description="Gamma band feature")


class PredictRequest(BaseModel):
    """Incoming request for stress prediction."""

    eeg_features: EEGFeatures | None = None


class PredictResponse(BaseModel):
    """API response returned to the dashboard or downstream clients."""

    stress_level: str
    color: str
    score: int = Field(ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    sample_index: int | None = None
    total_samples: int | None = None
    subject_id: int | None = None
    trial_id: int | None = None
    emotion_label: str | None = None
    reference_emotion_label: str | None = None
    eeg_features: EEGFeatures | None = None
    source: str | None = None
