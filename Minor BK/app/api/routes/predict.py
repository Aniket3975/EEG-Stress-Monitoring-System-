import time
from collections import deque
from threading import Lock

from fastapi import APIRouter, HTTPException

from app.models.stress_model import StressModelService
from app.schemas.stress import PredictRequest, PredictResponse
from app.services.serial_eeg import SerialEEGService
from app.utils.scoring import classify_stress

router = APIRouter(tags=["stress"])
stress_service = StressModelService()
serial_eeg_service = SerialEEGService()
_score_window: deque[tuple[float, int, float]] = deque()
_score_window_lock = Lock()
_last_serial_timestamp: float | None = None


@router.get("/predict", response_model=PredictResponse)
def predict_stress() -> PredictResponse:
    """Return the most recent live EEG prediction from the ESP serial feed."""

    global _last_serial_timestamp
    reading = serial_eeg_service.latest_reading()
    if reading is None:
        status = serial_eeg_service.status()
        detail = f"Waiting for live EEG input from serial port. Status: {status['status']}"
        if status["port"]:
            detail += f" ({status['port']})"
        raise HTTPException(status_code=503, detail=detail)

    live_result = stress_service.predict_live(reading.eeg_features)
    now = time.time()

    with _score_window_lock:
        if _last_serial_timestamp != reading.timestamp:
            _score_window.append((now, int(live_result["score"]), float(live_result["confidence"])))
            _last_serial_timestamp = reading.timestamp

        while _score_window and (now - _score_window[0][0]) > 10:
            _score_window.popleft()

        if _score_window:
            averaged_score = round(sum(item[1] for item in _score_window) / len(_score_window))
            averaged_confidence = sum(item[2] for item in _score_window) / len(_score_window)
        else:
            averaged_score = int(live_result["score"])
            averaged_confidence = float(live_result["confidence"])

    averaged_result = classify_stress(averaged_score)
    if averaged_score >= 67:
        emotion_label = "negative"
    elif averaged_score <= 38:
        emotion_label = "positive"
    else:
        emotion_label = "neutral"

    return PredictResponse(
        stress_level=averaged_result.stress_level,
        color=averaged_result.color,
        score=averaged_result.score,
        confidence=round(averaged_confidence, 4),
        emotion_label=emotion_label,
        eeg_features=reading.eeg_features,
        source="live",
    )


@router.post("/predict", response_model=PredictResponse)
def predict_stress_from_features(payload: PredictRequest) -> PredictResponse:
    """Accept live EEG band features for direct manual scoring."""

    if payload.eeg_features is None:
        raise HTTPException(status_code=400, detail="eeg_features are required for live prediction.")

    result = stress_service.predict_live(payload.eeg_features)
    return PredictResponse(
        stress_level=str(result["stress_level"]),
        color=str(result["color"]),
        score=int(result["score"]),
        confidence=float(result["confidence"]),
        emotion_label=str(result["emotion_label"]),
        eeg_features=payload.eeg_features,
        source="live",
    )


@router.get("/serial-status", tags=["stress"])
def serial_status() -> dict[str, object]:
    """Expose serial debug information for the connected ESP stream."""

    return serial_eeg_service.debug_snapshot()
