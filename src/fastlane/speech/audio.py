"""Audio helpers shared by STT, TTS and the voice server: everything is mono float32 at 16 kHz inside the
pipeline (what Whisper and Silero VAD expect); TTS output is resampled on the way in."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

RATE = 16_000


def resample(audio: np.ndarray, src_rate: int, dst_rate: int = RATE) -> np.ndarray:
    """Linear-interpolation resampling; good enough for speech going into STT or out to a browser."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32)
    n = int(round(len(audio) * dst_rate / src_rate))
    x = np.linspace(0, len(audio) - 1, n)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def to_mono_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return resample(audio, rate, RATE)


def load(path: str | Path | bytes) -> np.ndarray:
    """A WAV/FLAC/OGG file (or its bytes) as mono float32 at 16 kHz."""
    import soundfile as sf

    data, rate = sf.read(io.BytesIO(path) if isinstance(path, bytes) else str(path), dtype="float32", always_2d=False)
    return to_mono_16k(data, rate)


def wav_bytes(audio: np.ndarray, rate: int = RATE) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Little-endian 16-bit PCM (what the browser page sends) as float32 in [-1, 1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
