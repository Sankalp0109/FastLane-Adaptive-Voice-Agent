"""Turn-taking (Phase 7.3): Silero VAD decides when the caller starts and stops speaking.

    det = TurnDetector()
    for ev in det.feed(chunk_16k):        # any chunk size
        ev.kind == "start"   -> caller started speaking
        ev.kind == "partial" -> ev.audio = speech so far (every ~0.6 s), for partial transcripts / early lookup
        ev.kind == "end"     -> ev.audio = the whole turn, cut after `silence_ms` of silence

The end of a turn is only known `silence_ms` after the caller stops; that wait is part of the latency the caller
feels and is reported separately from model and STT time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastlane.speech.audio import RATE

WINDOW = 512  # Silero VAD works on 32 ms windows at 16 kHz


@dataclass
class VADEvent:
    kind: str  # "start" | "partial" | "end"
    audio: np.ndarray | None = None


class TurnDetector:
    def __init__(self, silence_ms: int = 600, partial_every_s: float = 0.6, threshold: float = 0.5,
                 preroll_ms: int = 300, max_turn_s: float = 30.0):
        from silero_vad import VADIterator, load_silero_vad

        self.vad = VADIterator(load_silero_vad(onnx=True), threshold=threshold, sampling_rate=RATE,
                               min_silence_duration_ms=silence_ms, speech_pad_ms=100)
        self.silence_ms = silence_ms
        self.partial_every = int(partial_every_s * RATE)
        self.preroll = int(preroll_ms * RATE / 1000)
        self.max_turn = int(max_turn_s * RATE)
        self.reset()

    def reset(self) -> None:
        self.vad.reset_states()
        self._rest = np.zeros(0, dtype=np.float32)  # samples not yet filling a whole window
        self._recent = np.zeros(0, dtype=np.float32)  # pre-roll kept while silent
        self._turn: list[np.ndarray] = []
        self._turn_len = 0
        self._last_partial = 0
        self.in_speech = False

    def feed(self, audio: np.ndarray) -> list[VADEvent]:
        events: list[VADEvent] = []
        data = np.concatenate([self._rest, np.asarray(audio, dtype=np.float32)])
        n = len(data) // WINDOW * WINDOW
        self._rest = data[n:]
        for i in range(0, n, WINDOW):
            w = data[i:i + WINDOW]
            ev = self.vad(w, return_seconds=False)
            if self.in_speech:
                self._turn.append(w)
                self._turn_len += WINDOW
            else:
                self._recent = np.concatenate([self._recent, w])[-self.preroll:]
            if ev and "start" in ev and not self.in_speech:
                self.in_speech = True
                self._turn, self._turn_len, self._last_partial = [self._recent.copy()], len(self._recent), 0
                events.append(VADEvent("start"))
            elif ev and "end" in ev and self.in_speech:
                events.append(VADEvent("end", np.concatenate(self._turn)))
                self._end_turn()
            elif self.in_speech and self._turn_len >= self.max_turn:
                events.append(VADEvent("end", np.concatenate(self._turn)))  # monologue guard
                self._end_turn()
            elif self.in_speech and self._turn_len - self._last_partial >= self.partial_every:
                self._last_partial = self._turn_len
                events.append(VADEvent("partial", np.concatenate(self._turn)))
        return events

    def _end_turn(self) -> None:
        self.in_speech = False
        self._turn, self._turn_len = [], 0
        self._recent = np.zeros(0, dtype=np.float32)
