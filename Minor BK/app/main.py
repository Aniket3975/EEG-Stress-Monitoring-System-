from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.predict import router as predict_router, serial_eeg_service

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="EEG Stress Detection API",
    description="Dataset-backed EEG stress detection service with a dashboard.",
    version="1.0.0",
)

app.include_router(predict_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup_event() -> None:
    serial_eeg_service.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    serial_eeg_service.stop()


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the real-time monitoring dashboard."""

    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Basic health endpoint for deployment checks."""

    return {"status": "ok"}
