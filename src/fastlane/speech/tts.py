"""Text to speech (Phase 7.2). Replies are spoken sentence by sentence, so the caller hears the first sentence
while the rest is still being synthesised ("time to first audio" is what the caller feels).

    tts = get_tts()                                  # Kokoro-82M, local ONNX on CPU
    for chunk in tts.stream("Just to confirm: two tickets. Shall I book it?"):
        play(chunk.audio)                            # float32, 16 kHz

Kokoro has no Indian-English voice; `hf_alpha` / `hm_omega` (Hindi speakers) read English with an Indian accent,
`af_heart` is the clearest. Sarvam Bulbul (cloud, needs SARVAM_API_KEY) is the Indian-English option.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fastlane.speech.audio import RATE, resample

KOKORO_REPO = "fastrtc/kokoro-onnx"


@dataclass
class Speech:
    text: str
    audio: np.ndarray  # float32 mono at 16 kHz
    ms: float  # synthesis wall time

    @property
    def seconds(self) -> float:
        return len(self.audio) / RATE


class TTS(Protocol):
    name: str

    def speak(self, text: str) -> Speech: ...


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'₹])")


def sentences(text: str) -> list[str]:
    """Split a reply into sentences for streaming; keeps "Rs. 300" and "6 p.m." together."""
    text = " ".join(text.replace("**", "").split())
    parts = [p.strip() for p in _SENTENCE.split(text) if p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and (len(merged[-1]) < 12 or re.search(r"\b(Rs|No|Dr|Mr|Mrs|p\.m|a\.m)\.$", merged[-1])):
            merged[-1] += " " + p
        else:
            merged.append(p)
    # A short lead-in ("Just to confirm:") is spoken on its own, so the caller hears something while the long
    # read-back is still being synthesised.
    if merged:
        head, colon, rest = merged[0].partition(": ")
        if colon and len(head) <= 25 and len(rest) > 40:
            merged[0:1] = [head + ":", rest]
    return merged


def for_speech(text: str) -> str:
    """Written forms a TTS reads badly: rupee sign, booking references spelled with spaces stay as they are."""
    text = text.replace("₹", "rupees ").replace("Rs.", "rupees").replace("Rs ", "rupees ")
    text = re.sub(r"\b(\d{1,2}):00\b", r"\1", text)
    return text


class KokoroTTS:
    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        from huggingface_hub import hf_hub_download
        from kokoro_onnx import Kokoro

        model = hf_hub_download(KOKORO_REPO, "kokoro-v1.0.onnx")
        voices = hf_hub_download(KOKORO_REPO, "voices-v1.0.bin")
        self.k = Kokoro(model, voices)
        self.voice, self.speed = voice, speed
        self.name = f"kokoro:{voice}"

    def voices(self) -> list[str]:
        return sorted(self.k.get_voices())

    def speak(self, text: str) -> Speech:
        t0 = time.perf_counter()
        samples, rate = self.k.create(for_speech(text), voice=self.voice, speed=self.speed, lang="en-us")
        return Speech(text, resample(np.asarray(samples, dtype=np.float32), rate), (time.perf_counter() - t0) * 1000)


class SarvamTTS:
    """Sarvam Bulbul (cloud): Indian-English voices. Needs SARVAM_API_KEY."""

    URL = "https://api.sarvam.ai/text-to-speech"

    def __init__(self, speaker: str = "anushka", model: str = "bulbul:v2"):
        from fastlane.models.env import get_key

        self.key = get_key("SARVAM_API_KEY")
        self.speaker, self.model = speaker, model
        self.name = f"sarvam:{model}:{speaker}"

    def speak(self, text: str) -> Speech:
        import base64
        import json
        import urllib.request

        from fastlane.speech.audio import load

        body = json.dumps({"text": for_speech(text), "target_language_code": "en-IN", "speaker": self.speaker,
                           "model": self.model, "speech_sample_rate": RATE}).encode()
        req = urllib.request.Request(self.URL, data=body, method="POST",
                                     headers={"api-subscription-key": self.key, "Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as resp:
            wav = base64.b64decode(json.loads(resp.read())["audios"][0])
        return Speech(text, load(wav), (time.perf_counter() - t0) * 1000)


def stream(tts: TTS, text: str) -> Iterator[Speech]:
    for s in sentences(text):
        yield tts.speak(s)


def get_tts(spec: str = "kokoro") -> TTS:
    """'kokoro[:voice]' or 'sarvam[:speaker]'."""
    kind, _, arg = spec.partition(":")
    if kind == "kokoro":
        return KokoroTTS(arg or "af_heart")
    if kind == "sarvam":
        return SarvamTTS(arg or "anushka")
    raise ValueError(f"unknown TTS backend {spec!r}")
