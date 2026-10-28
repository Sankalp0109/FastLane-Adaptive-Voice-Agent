"""Speech-noise copies by a TTS -> STT round trip (Phase 7.6; replaces the rule-based edits in noise.py).

    fastlane train roundtrip                 # speak and transcribe every caller line in data/train (cached, resumable)
    fastlane train renoise                   # rebuild the noise copies in data/train from the cache

Each caller line is spoken by a Kokoro voice (half of them the Hindi-accented hf_/hm_ voices, speed 0.9-1.15),
sent through a phone channel (300-3400 Hz band, background noise at 12-30 dB SNR) and transcribed by the STT the
voice demo runs (Whisper large-v3-turbo with the domain word list). The adapter so trains on the mistakes of
the recogniser it is deployed behind, not on hand-written ones.

A transcript is used only if the line's key details survive: places, films, dates, names, numbers and yes/no.
The training target stays the same as for the clean line, so a transcript that lost "4" would teach the model
to answer "quantity: 4" from nothing (inventing). The rule-based edits did exactly that ("four" -> "for");
transcripts that lose a detail are counted in the report but not trained on.
"""

from __future__ import annotations

import json
import random
import re
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fastlane.config import DATA_DIR
from fastlane.speech.audio import RATE
from fastlane.speech.wer import Entities, normalise

CACHE = DATA_DIR / "speech" / "roundtrip" / "transcripts.jsonl"
INDIAN_VOICES = ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]
OTHER_VOICES = ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bm_george"]
YES = {"yes", "yeah", "yep", "haan", "han", "ha", "ok", "okay", "sure", "correct", "right", "fine"}
# Short forms the world's entity lists leave out (they hold full month names and aliases longer than 3 letters).
SHORT_FORMS = {"jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
               "hyd", "blr", "bza", "vja", "maa"}
NO = {"no", "nope", "nahi", "nahin", "not", "dont", "cant", "wont", "wrong"}


@dataclass
class Trip:
    text: str
    variant: int
    voice: str
    speed: float
    snr_db: float
    hyp: str
    stt_ms: float
    wer: float = 0.0
    lost: list[str] = field(default_factory=list)  # key details missing from the transcript

    @property
    def usable(self) -> bool:
        return bool(self.hyp.strip()) and not self.lost


# ------------------------------------------------------------------ the phone channel
def phone_channel(audio: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """Telephone band (300-3400 Hz) and band-limited background noise at `snr_db`, with a little silence around."""
    pad = np.zeros(int(0.25 * RATE), dtype=np.float32)
    x = np.concatenate([pad, np.asarray(audio, dtype=np.float32), pad])
    x = _band(x)
    noise = _band(rng.standard_normal(len(x)).astype(np.float32))
    speech_power = float(np.mean(_band(np.asarray(audio, dtype=np.float32)) ** 2)) or 1e-8
    noise *= np.sqrt(speech_power / (10 ** (snr_db / 10)) / (float(np.mean(noise ** 2)) or 1e-8))
    y = (x + noise) * rng.uniform(0.5, 1.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _band(x: np.ndarray, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    spec[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spec, len(x)).astype(np.float32)


# ------------------------------------------------------------------ which transcripts are safe to train on
def people_names() -> list[str]:
    from fastlane.world.names import FEMALE_FIRST, MALE_FIRST, SURNAMES

    return sorted(set(MALE_FIRST) | set(FEMALE_FIRST) | set(SURNAMES))


def lost_details(ref: str, hyp: str, entities: Entities) -> list[str]:
    """Key details of `ref` missing from `hyp` (both raw text), e.g. ["numbers:4", "yes/no"]."""
    r, h = normalise(ref), normalise(hyp)
    lost = []
    hyp_numbers = re.findall(r"\b\d+\b", h)
    for kind, items in entities.find(r).items():
        for x in items:
            if kind == "numbers" and int(x) == 0:  # "17:00" said as "seventeen"
                continue
            if not (x in hyp_numbers if kind == "numbers" else re.search(rf"\b{re.escape(x)}\b", h)):
                lost.append(f"{kind}:{x}")
    rw, hw = set(r.split()), set(h.split())
    lost += [f"short:{w}" for w in sorted(rw & SHORT_FORMS - hw)]  # "26 Oct" heard as "26 Arc", "Hyd" as "Hid"
    squashed = re.sub(r"[^A-Z0-9]", "", hyp.upper())
    lost += [f"reference:{ref_id}" for ref_id in booking_refs(ref) if ref_id not in squashed]
    if _polarity(r) != _polarity(h):  # "no" heard as "now" would confirm a read-back the caller rejected
        lost.append("yes/no")
    return lost


def booking_refs(text: str) -> list[str]:
    """Booking references in a caller line ("MV GVPUZ", "RTFUJTN", "M V P K N 6 V"), without spaces."""
    refs = []
    for run in re.findall(r"\b[A-Z0-9]+(?: [A-Z0-9]+)*\b", text):
        ref_id = run.replace(" ", "")
        if len(ref_id) >= 6 and re.search(r"[A-Z]", ref_id) and not set(run.split()) & {"AM", "PM", "AC"}:
            refs.append(ref_id)
    return refs


def _polarity(norm: str) -> str | None:
    words = set(norm.split())
    return "no" if words & NO else "yes" if words & YES else None


def _wer(ref: str, hyp: str) -> float:
    import jiwer

    r = normalise(ref)
    return round(jiwer.wer(r, normalise(hyp) or "<empty>"), 4) if r else 0.0


# ------------------------------------------------------------------ building the cache
def caller_lines(train_dir: Path) -> list[str]:
    lines: set[str] = set()
    for path in sorted(train_dir.glob("*/*.jsonl")):
        if path.stem not in ("train", "val"):
            continue
        for row in path.read_text().splitlines():
            ex = json.loads(row)
            if ex["meta"].get("noisy"):
                continue
            lines.update(ln[len("Caller: "):] for ln in ex["messages"][1]["content"].split("\n") if ln.startswith("Caller: "))
    return sorted(line for line in lines if line.strip())


def load_cache(path: Path = CACHE) -> list[Trip]:
    if not path.exists():
        return []
    return [Trip(**json.loads(row)) for row in path.read_text().splitlines() if row.strip()]


def run(train_dir: Path, variants: int = 2, stt_spec: str = "mlx", limit: int | None = None,
        cache: Path = CACHE, progress=print) -> dict:
    from fastlane.speech.stt import domain_prompt, get_stt
    from fastlane.speech.tts import KokoroTTS

    lines = caller_lines(train_dir)[:limit] if limit else caller_lines(train_dir)
    done = {(t.text, t.variant) for t in load_cache(cache)}
    todo = [(line, v) for line in lines for v in range(variants) if (line, v) not in done]
    progress(f"{len(lines)} caller lines x {variants} variants: {len(done)} cached, {len(todo)} to do")
    if not todo:
        return summary(load_cache(cache))
    tts, stt, prompt = KokoroTTS(), get_stt(stt_spec), domain_prompt()
    entities = Entities.from_world(people_names())
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a") as f:
        for i, (line, v) in enumerate(todo, 1):
            seed = zlib.crc32(f"{v}|{line}".encode())
            rng = random.Random(seed)
            voice = rng.choice(INDIAN_VOICES) if rng.random() < 0.5 else rng.choice(OTHER_VOICES)
            speed, snr = round(rng.uniform(0.9, 1.15), 2), round(rng.uniform(12, 30), 1)
            tts.voice, tts.speed = voice, speed
            audio = phone_channel(tts.speak(line).audio, np.random.default_rng(seed), snr)
            t = stt.transcribe(audio, prompt=prompt)
            trip = Trip(line, v, voice, speed, snr, t.text, round(t.ms, 1), _wer(line, t.text), lost_details(line, t.text, entities))
            f.write(json.dumps(asdict(trip), ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                progress(f"{i}/{len(todo)} done")
    return summary(load_cache(cache))


def rejudge(cache: Path = CACHE) -> dict:
    """Score the cached transcripts again (after a change to the checks) without speaking anything."""
    entities = Entities.from_world(people_names())
    trips = load_cache(cache)
    for t in trips:
        t.wer, t.lost = _wer(t.text, t.hyp), lost_details(t.text, t.hyp, entities)
    cache.write_text("".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in trips))
    return summary(trips)


def summary(trips: list[Trip]) -> dict:
    if not trips:
        return {}
    lost = Counter(x.split(":")[0] for t in trips for x in t.lost)
    by_voice: dict[str, list[Trip]] = {}
    for t in trips:
        by_voice.setdefault("indian" if t.voice in INDIAN_VOICES else "other", []).append(t)
    return {
        "lines": len({t.text for t in trips}), "transcripts": len(trips),
        "usable": round(sum(t.usable for t in trips) / len(trips), 4),
        "changed_and_usable": round(sum(t.usable and normalise(t.hyp) != normalise(t.text) for t in trips) / len(trips), 4),
        "mean_wer": round(sum(t.wer for t in trips) / len(trips), 4),
        "lost_by_kind": dict(lost.most_common()),
        "by_voice": {k: {"n": len(v), "mean_wer": round(sum(t.wer for t in v) / len(v), 4),
                         "usable": round(sum(t.usable for t in v) / len(v), 4)} for k, v in sorted(by_voice.items())},
        "stt_ms_median": sorted(t.stt_ms for t in trips)[len(trips) // 2],
    }


# ------------------------------------------------------------------ using it in the training data
class RoundTripNoise:
    """Replaces caller lines with usable round-trip transcripts (lines without one stay as they are)."""

    def __init__(self, trips: list[Trip]):
        self.table: dict[str, list[str]] = {}
        for t in trips:
            if t.usable:
                self.table.setdefault(t.text, []).append(t.hyp)

    def line(self, text: str, rng: random.Random) -> str:
        options = self.table.get(text)
        return rng.choice(options) if options else text

    def user_content(self, content: str, rng: random.Random) -> str:
        return "\n".join("Caller: " + self.line(ln[len("Caller: "):], rng) if ln.startswith("Caller: ") else ln
                         for ln in content.split("\n"))
