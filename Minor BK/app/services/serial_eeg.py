from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import serial
import numpy as np

from app.schemas.stress import EEGFeatures

DEFAULT_PORT_CANDIDATES = [
    "/dev/cu.usbserial-0001",
    "/dev/cu.usbserial*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
]
DEFAULT_BAUD_RATE = int(os.getenv("EEG_SERIAL_BAUD", "115200"))
DEFAULT_SAMPLE_RATE_HZ = int(os.getenv("EEG_SAMPLE_RATE_HZ", "200"))
SERIAL_POLL_INTERVAL_SECONDS = 1.0
SERIAL_RETRY_INTERVAL_SECONDS = 2.0
SMOOTHING_ALPHA = float(os.getenv("EEG_SMOOTHING_ALPHA", "0.18"))
FEATURE_NAMES = ("alpha", "beta", "theta", "delta", "gamma")
CSV_RE = re.compile(r"[-+]?\d*\.?\d+")
KEY_VALUE_RE = re.compile(r"(alpha|beta|theta|delta|gamma)\s*[:=]\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
SINGLE_VALUE_RE = re.compile(r"^\s*[-+]?\d*\.?\d+\s*$")
BAND_SPECS = (
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 50),
)
RAW_SAMPLE_WINDOW = int(os.getenv("EEG_SAMPLE_WINDOW", "256"))


@dataclass(frozen=True)
class SerialReading:
    eeg_features: EEGFeatures
    raw_line: str
    port: str
    baud_rate: int
    timestamp: float


class SerialEEGService:
    """Continuously read EEG features from a serial-connected ESP device."""

    def __init__(self, baud_rate: int = DEFAULT_BAUD_RATE, sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ) -> None:
        self.baud_rate = baud_rate
        self.sample_rate_hz = sample_rate_hz
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_reading: SerialReading | None = None
        self._status = "idle"
        self._port: str | None = None
        self._last_error: str | None = None
        self._last_raw_lines: deque[str] = deque(maxlen=10)
        self._raw_sample_buffer: deque[float] = deque(maxlen=RAW_SAMPLE_WINDOW)
        self._smoothed_features: EEGFeatures | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="serial-eeg-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def latest_reading(self) -> SerialReading | None:
        with self._lock:
            return self._latest_reading

    def status(self) -> dict[str, str | None]:
        with self._lock:
            return {
                "status": self._status,
                "port": self._port,
                "last_error": self._last_error,
                "last_raw_line": self._last_raw_lines[-1] if self._last_raw_lines else None,
            }

    def debug_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "status": self._status,
                "port": self._port,
                "baud_rate": self.baud_rate,
                "sample_rate_hz": self.sample_rate_hz,
                "raw_sample_buffer_size": len(self._raw_sample_buffer),
                "smoothing_alpha": SMOOTHING_ALPHA,
                "last_error": self._last_error,
                "last_raw_lines": list(self._last_raw_lines),
                "latest_reading": None
                if self._latest_reading is None
                else {
                    "port": self._latest_reading.port,
                    "baud_rate": self._latest_reading.baud_rate,
                    "timestamp": self._latest_reading.timestamp,
                    "raw_line": self._latest_reading.raw_line,
                    "eeg_features": self._latest_reading.eeg_features.model_dump(),
                },
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            port = self._detect_port()
            if not port:
                self._set_status("no_port", None)
                self._stop_event.wait(SERIAL_RETRY_INTERVAL_SECONDS)
                continue

            self._set_status("connecting", port)
            try:
                with serial.Serial(
                    port,
                    baudrate=self.baud_rate,
                    timeout=SERIAL_POLL_INTERVAL_SECONDS,
                    write_timeout=SERIAL_POLL_INTERVAL_SECONDS,
                ) as ser:
                    try:
                        ser.dtr = False
                        ser.rts = False
                        ser.reset_input_buffer()
                    except Exception:
                        pass
                    self._set_status("connected", port)
                    time.sleep(2)
                    buffer = ""
                    while not self._stop_event.is_set():
                        try:
                            waiting = max(1, ser.in_waiting)
                            raw = ser.read(waiting)
                        except serial.SerialException as exc:
                            self._set_error(str(exc))
                            break

                        if not raw:
                            continue

                        chunk = raw.decode("utf-8", errors="replace")
                        buffer += chunk

                        while "\n" in buffer or "\r" in buffer:
                            parts = re.split(r"[\r\n]+", buffer, maxsplit=1)
                            line = parts[0].strip()
                            buffer = parts[1] if len(parts) > 1 else ""
                            self._consume_line(line, port)

                        if len(buffer) > 512:
                            self._consume_line(buffer.strip(), port)
                            buffer = ""
            except (serial.SerialException, OSError) as exc:
                self._set_error(str(exc))
                self._set_status("disconnected", port)
                self._stop_event.wait(SERIAL_RETRY_INTERVAL_SECONDS)

    def _detect_port(self) -> str | None:
        configured = os.getenv("EEG_SERIAL_PORT")
        if configured:
            return configured

        for candidate in DEFAULT_PORT_CANDIDATES:
            if "*" in candidate:
                matches = sorted(str(path) for path in Path("/").glob(candidate.lstrip("/")))
                if matches:
                    return matches[0]
            elif Path(candidate).exists():
                return candidate
        return None

    def _set_status(self, status: str, port: str | None) -> None:
        with self._lock:
            self._status = status
            self._port = port

    def _set_error(self, error: str | None) -> None:
        with self._lock:
            self._last_error = error

    def _consume_line(self, line: str, port: str) -> None:
        if not line:
            return

        with self._lock:
            self._last_raw_lines.append(line)

        eeg_features = self._parse_line(line)
        if eeg_features is None:
            raw_value = self._parse_single_value(line)
            if raw_value is None:
                return
            eeg_features = self._raw_value_to_features(raw_value)
            if eeg_features is None:
                return

        eeg_features = self._smooth_features(eeg_features)

        reading = SerialReading(
            eeg_features=eeg_features,
            raw_line=line,
            port=port,
            baud_rate=self.baud_rate,
            timestamp=time.time(),
        )
        with self._lock:
            self._latest_reading = reading
            self._last_error = None

    @staticmethod
    def _parse_line(line: str) -> EEGFeatures | None:
        parsers = (
            SerialEEGService._parse_json,
            SerialEEGService._parse_key_value_pairs,
            SerialEEGService._parse_csv,
        )
        for parser in parsers:
            parsed = parser(line)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_json(line: str) -> EEGFeatures | None:
        start = line.find("{")
        end = line.rfind("}")
        if start != -1 and end != -1 and end > start:
            line = line[start : end + 1]
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        values = {name: payload.get(name) for name in FEATURE_NAMES}
        if any(value is None for value in values.values()):
            return None
        return EEGFeatures(**{name: float(value) for name, value in values.items()})

    @staticmethod
    def _parse_key_value_pairs(line: str) -> EEGFeatures | None:
        values: dict[str, float] = {}
        for key, raw_value in KEY_VALUE_RE.findall(line):
            key = key.strip().lower()
            try:
                values[key] = float(raw_value)
            except ValueError:
                return None

        if len(values) != len(FEATURE_NAMES):
            return None
        return EEGFeatures(**values)

    @staticmethod
    def _parse_csv(line: str) -> EEGFeatures | None:
        if any(character.isalpha() for character in line):
            return None
        if "," in line:
            parts = [part.strip() for part in line.split(",") if part.strip()]
        else:
            parts = [part.strip() for part in line.split() if part.strip()]
        if len(parts) != len(FEATURE_NAMES):
            return None
        if any(not SINGLE_VALUE_RE.fullmatch(part) for part in parts):
            return None
        numbers = parts
        return EEGFeatures(**{name: float(value) for name, value in zip(FEATURE_NAMES, numbers)})

    @staticmethod
    def _parse_single_value(line: str) -> float | None:
        if not SINGLE_VALUE_RE.fullmatch(line):
            return None
        try:
            return float(line.strip())
        except ValueError:
            return None

    def _raw_value_to_features(self, raw_value: float) -> EEGFeatures | None:
        with self._lock:
            self._raw_sample_buffer.append(raw_value)
            if len(self._raw_sample_buffer) < min(64, RAW_SAMPLE_WINDOW):
                return None
            samples = list(self._raw_sample_buffer)

        signal = np.asarray(samples, dtype=np.float64)
        signal = signal - np.mean(signal)
        if not np.any(signal):
            return None

        freqs = np.fft.rfftfreq(signal.shape[0], d=1 / self.sample_rate_hz)
        power = np.abs(np.fft.rfft(signal)) ** 2
        band_values: dict[str, float] = {}

        for band_name, low_hz, high_hz in BAND_SPECS:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            if not np.any(mask):
                band_values[band_name] = 0.0
            else:
                band_values[band_name] = float(np.mean(power[mask]))

        total_power = max(sum(band_values.values()), 1e-9)
        normalized = {name: value / total_power for name, value in band_values.items()}
        return EEGFeatures(
            alpha=round(normalized["alpha"], 6),
            beta=round(normalized["beta"], 6),
            theta=round(normalized["theta"], 6),
            delta=round(normalized["delta"], 6),
            gamma=round(normalized["gamma"], 6),
        )

    def _smooth_features(self, eeg_features: EEGFeatures) -> EEGFeatures:
        with self._lock:
            previous = self._smoothed_features

        if previous is None:
            smoothed = eeg_features
        else:
            smoothed = EEGFeatures(
                alpha=round(self._ema(previous.alpha, eeg_features.alpha), 6),
                beta=round(self._ema(previous.beta, eeg_features.beta), 6),
                theta=round(self._ema(previous.theta, eeg_features.theta), 6),
                delta=round(self._ema(previous.delta, eeg_features.delta), 6),
                gamma=round(self._ema(previous.gamma, eeg_features.gamma), 6),
            )

        with self._lock:
            self._smoothed_features = smoothed
        return smoothed

    @staticmethod
    def _ema(previous: float | None, current: float | None) -> float:
        prev = float(previous or 0.0)
        curr = float(current or 0.0)
        return (prev * (1.0 - SMOOTHING_ALPHA)) + (curr * SMOOTHING_ALPHA)
