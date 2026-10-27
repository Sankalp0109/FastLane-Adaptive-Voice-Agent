"""Speech to text (Phase 7.1). One interface, several backends, so the STT comparison (7.5) and the voice
server (7.4) use exactly the same code.

    stt = get_stt("mlx")                       # mlx-whisper large-v3-turbo on Apple Silicon (default)
    stt = get_stt("faster-whisper:small.en")   # CTranslate2 on CPU, any machine
    stt = get_stt("sarvam")                    # Sarvam AI cloud, Indian English (needs SARVAM_API_KEY)
    t = stt.transcribe(audio_16k, prompt=domain_prompt())

`domain_prompt()` lists the world's films, cinemas, branches and cities. Whisper uses it as preceding text, which
biases it towards spelling those names right (an Indian-English caller saying "Bezawada" or "Kaala Samudram").
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from fastlane.speech.audio import RATE, wav_bytes


# Greedy first; if the output looks like a repetition loop (compression ratio > 2.4) or is unlikely, Whisper decodes
# again with a little randomness. A fixed 0.0 turns that off: on Svarah, short clips then looped ("Eight" ->
# "A-T-T-T-..."), which dominated the WER.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class Transcript:
    text: str
    ms: float  # wall time of the transcription
    backend: str
    audio_s: float = 0.0  # length of the audio, for the real-time factor


class STT(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcript: ...


@lru_cache(maxsize=1)
def domain_prompt() -> str:
    """Names a caller may say, as a short sentence list (Whisper reads at most ~224 prompt tokens)."""
    from fastlane.config import FESTIVALS, WORLD_DB
    from fastlane.world.db import World

    w = World.open(WORLD_DB) if WORLD_DB.exists() else None
    films = [r["title"] for r in w.q("SELECT title FROM movies ORDER BY title")] if w else []
    cinemas = [f"{r['name']} {r['area']}" for r in w.q("SELECT name, area FROM cinemas ORDER BY name")] if w else []
    branches = [r["area"] for r in w.q("SELECT area FROM restaurants ORDER BY area")] if w else []
    parts = [
        "FastLane booking call in Hyderabad.",
        ("Films: " + ", ".join(films) + ".") if films else "",
        ("Cinemas: " + ", ".join(cinemas) + ".") if cinemas else "",
        ("Tandoor Tales " + ", ".join(branches) + ".") if branches else "",
        "Buses to Bengaluru, Vijayawada (Bezawada), Chennai.",
        "Festivals: " + ", ".join(list(FESTIVALS)[:7]) + ".",
    ]
    return " ".join(p for p in parts if p)


class MLXWhisper:
    """mlx-whisper on Apple Silicon (Metal). large-v3-turbo: ~1.6 GB, fast enough for live turns on an M-series Mac."""

    def __init__(self, repo: str = "mlx-community/whisper-large-v3-turbo"):
        os.environ.setdefault("MLX_MPI_LIBNAME", "/nonexistent/libmpi.dylib")  # Anaconda ships MPICH; MLX rejects it
        import mlx_whisper

        self._mw = mlx_whisper
        self.repo = repo
        self.name = f"mlx:{repo.split('/')[-1]}"

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        t0 = time.perf_counter()
        out = self._mw.transcribe(np.asarray(audio, dtype=np.float32), path_or_hf_repo=self.repo, language="en",
                                  initial_prompt=prompt, temperature=FALLBACK_TEMPERATURES, condition_on_previous_text=False,
                                  verbose=None)
        return Transcript(out["text"].strip(), (time.perf_counter() - t0) * 1000, self.name, len(audio) / RATE)


class FasterWhisper:
    """faster-whisper (CTranslate2), int8 on CPU: portable baseline that runs on any laptop."""

    def __init__(self, size: str = "small.en", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(size, device=device, compute_type=compute_type)
        self.name = f"faster-whisper:{size}"

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(np.asarray(audio, dtype=np.float32), language="en", initial_prompt=prompt,
                                            beam_size=1, temperature=list(FALLBACK_TEMPERATURES), condition_on_previous_text=False,
                                            vad_filter=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text, (time.perf_counter() - t0) * 1000, self.name, len(audio) / RATE)


class SarvamSTT:
    """Sarvam AI speech-to-text (cloud, trained on Indian speech). Needs SARVAM_API_KEY in .env.
    The domain prompt is not sent: the API transcribes without a vocabulary hint."""

    URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, model: str = "saarika:v2.5", language: str = "en-IN"):
        from fastlane.models.env import get_key

        self.key = get_key("SARVAM_API_KEY")
        self.model, self.language = model, language
        self.name = f"sarvam:{model}"

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        import json
        import urllib.request
        import uuid

        boundary = uuid.uuid4().hex
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                 for k, v in (("model", self.model), ("language_code", self.language))]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="turn.wav"\r\n'
                     f"Content-Type: audio/wav\r\n\r\n".encode() + wav_bytes(audio) + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(self.URL, data=body, method="POST", headers={
            "api-subscription-key": self.key, "Content-Type": f"multipart/form-data; boundary={boundary}"})
        import urllib.error

        for attempt in range(8):  # the API rate-limits bursts (HTTP 429): wait and retry; waiting is not latency
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = json.loads(resp.read()).get("transcript", "")
                return Transcript(text.strip(), (time.perf_counter() - t0) * 1000, self.name, len(audio) / RATE)
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 7:
                    raise
                time.sleep(float(e.headers.get("Retry-After") or min(60, 2 ** attempt)))
        raise RuntimeError("unreachable")


def get_stt(spec: str = "mlx") -> STT:
    """'mlx[:repo]', 'faster-whisper[:size]' or 'sarvam[:model]'."""
    kind, _, arg = spec.partition(":")
    if kind == "mlx":
        return MLXWhisper(arg or "mlx-community/whisper-large-v3-turbo")
    if kind == "faster-whisper":
        return FasterWhisper(arg or "small.en")
    if kind == "sarvam":
        return SarvamSTT(arg or "saarika:v2.5")
    raise ValueError(f"unknown STT backend {spec!r}")
