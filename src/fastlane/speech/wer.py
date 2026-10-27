"""STT comparison (Phase 7.5): word error rate overall and on the words a booking depends on.

    fastlane speech wer --set tts --backends mlx mlx:noprompt faster-whisper:small.en
    fastlane speech wer --set svarah --n 300            # real Indian-English speech (needs HF_TOKEN, gated dataset)
    fastlane speech wer --set dir:recordings/           # own recordings: a.wav + a.txt pairs

Sets
  tts      real caller lines from the pilot runs (LLM caller), spoken by Kokoro voices incl. two Indian-accented
           ones (hf_alpha, hm_omega). Controlled and entity-dense, but synthetic speech.
  svarah   AI4Bharat Svarah: Indian-English speakers from many states (test split, random sample).
  dir:P    own recordings: every P/x.wav with its transcript in P/x.txt.

Key-detail accuracy: for every place, film, date word, number and person name in the reference, is it in the
transcript? (A wrong digit in a time or a wrong cinema breaks the booking even when the WER is low.)
"""

from __future__ import annotations

import json
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fastlane.config import DATA_DIR, FESTIVALS, FESTIVAL_ALIASES

_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
        "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
        "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
MONTHS = "january february march april may june july august september october november december".split()
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()


def normalise(text: str) -> str:
    """Lowercase, no punctuation, numbers as digits ("forty-two" -> 42, "23rd" -> 23, "7 p.m." -> 7 pm)."""
    t = re.sub(r"(\d),(\d)", r"\1\2", text.lower())  # "1,500" -> "1500"
    t = t.replace("₹", " rupees ").replace("rs.", "rupees").replace("-", " ")
    t = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", t)
    t = re.sub(r"(\d)\s?([ap]m)\b", r"\1 \2", t)  # "9pm" -> "9 pm"
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"o'\s?clock", "oclock", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t).replace("'", "")
    words, out, i = t.split(), [], 0
    while i < len(words):
        w = words[i]
        if w in _TENS:
            n = _TENS[w]
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if nxt in _UNITS[1:10]:
                n += _UNITS.index(nxt)
                i += 1
            elif nxt in _ORD and _ORD[nxt] < 10:
                n += _ORD[nxt]
                i += 1
            out.append(str(n))
        elif w in _UNITS:
            out.append(str(_UNITS.index(w)))
        elif w in _ORD:
            out.append(str(_ORD[w]))
        else:
            out.append(w)
        i += 1
    return " ".join(out)


# ------------------------------------------------------------------ key details
@dataclass
class Entities:
    places: list[str]
    films: list[str]
    dates: list[str]
    names: list[str]

    @classmethod
    def from_world(cls, people: list[str], world_db: str | Path | None = None) -> "Entities":
        from fastlane.config import WORLD_DB
        from fastlane.gateway.matching import CITY_ALIASES
        from fastlane.world.db import World

        w = World.open(world_db or WORLD_DB)
        places = {r["name"] for r in w.q("SELECT name FROM cinemas")} | {r["area"] for r in w.q("SELECT area FROM cinemas")}
        places |= {r["area"] for r in w.q("SELECT area FROM restaurants")} | {"tandoor tales"}
        places |= set(CITY_ALIASES) | {a for al in CITY_ALIASES.values() for a in al if len(a) > 3}
        films = {r["title"] for r in w.q("SELECT title FROM movies")}
        dates = set(MONTHS) | set(WEEKDAYS) | {f.lower() for f in FESTIVALS} | set(FESTIVAL_ALIASES) | {"tomorrow", "today"}
        return cls(*(sorted({normalise(x) for x in group if x}, key=len, reverse=True)
                     for group in (places, films, dates, people)))

    def find(self, ref_norm: str) -> dict[str, list[str]]:
        """Key details in a normalised reference, by kind (longest phrases first, no double counting)."""
        found: dict[str, list[str]] = {"places": [], "films": [], "dates": [], "names": [], "numbers": []}
        taken = ref_norm
        for kind in ("films", "places", "dates", "names"):
            for phrase in getattr(self, kind):
                if re.search(rf"\b{re.escape(phrase)}\b", taken):
                    found[kind].append(phrase)
                    taken = re.sub(rf"\b{re.escape(phrase)}\b", " ", taken)
        found["numbers"] = re.findall(r"\b\d+\b", taken)
        return found


def detail_hits(found: dict[str, list[str]], hyp_norm: str) -> dict[str, tuple[int, int]]:
    out = {}
    for kind, items in found.items():
        hyp_numbers = re.findall(r"\b\d+\b", hyp_norm)
        hits = sum(1 for x in items if (x in hyp_numbers if kind == "numbers" else re.search(rf"\b{re.escape(x)}\b", hyp_norm)))
        out[kind] = (hits, len(items))
    return out


# ------------------------------------------------------------------ test sets
@dataclass
class Clip:
    id: str
    audio: np.ndarray
    reference: str
    speaker: str = ""


def tts_set(n: int | None = None, seed: int = 0) -> tuple[list[Clip], list[str]]:
    """Caller lines from the pilot runs, spoken by Kokoro voices (cached as WAV under data/speech/tts_set/)."""
    from fastlane.speech.audio import load
    from fastlane.speech.tts import KokoroTTS

    lines, people = [], set()
    for run in sorted((DATA_DIR / "runs").glob("pilot*/fastlane.jsonl")):
        for row in map(json.loads, run.read_text().splitlines()):
            for t in row["turns"]:
                if len(t["caller"].split()) >= 3:
                    lines.append(t["caller"])
    for s in map(json.loads, (DATA_DIR / "test_calls.jsonl").read_text().splitlines()):
        people.update(p.split(",")[0].strip() for p in re.split(r";", s["facts"].get("passengers (name, age, gender)", "")) if p.strip())
        people.add(s["caller_name"])
    lines = sorted(set(lines))
    random.Random(seed).shuffle(lines)
    lines = lines[:n] if n else lines
    out_dir = DATA_DIR / "speech" / "tts_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    voices = ["hf_alpha", "hm_omega", "af_heart", "bm_george"]
    engines: dict[str, KokoroTTS] = {}
    clips = []
    for i, line in enumerate(lines):
        voice = voices[i % len(voices)]
        wav = out_dir / f"{i:04d}_{voice}.wav"
        if not wav.exists():
            import soundfile as sf

            if voice not in engines:
                engines[voice] = KokoroTTS(voice)
            sf.write(wav, engines[voice].speak(line).audio, 16000)
        clips.append(Clip(wav.stem, load(wav), line, voice))
    return clips, sorted(people)


def svarah_set(n: int = 300, seed: int = 0) -> list[Clip]:
    """Random sample of the Svarah test split (gated on Hugging Face: accept the terms, put HF_TOKEN in .env)."""
    import io
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from fastlane.models.env import get_key
    from fastlane.speech.audio import load

    token = os.environ.get("HF_TOKEN") or get_key("HF_TOKEN")
    rng = random.Random(seed)
    clips: list[Clip] = []
    shards = [f"data/test-0000{i}-of-00003.parquet" for i in range(3)]
    per_shard = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    for shard, k in zip(shards, per_shard):
        path = hf_hub_download("ai4bharat/Svarah", shard, repo_type="dataset", token=token)
        table = pq.read_table(path)
        cols = table.column_names
        text_col = next(c for c in ("text", "transcript", "sentence") if c in cols)
        idx = rng.sample(range(table.num_rows), min(k, table.num_rows))
        for i in idx:
            row = table.slice(i, 1).to_pylist()[0]
            audio = row["audio_filepath"] if "audio_filepath" in row else row.get("audio")
            data = audio["bytes"] if isinstance(audio, dict) else audio
            speaker = f"{row.get('primary_language', '')}/{row.get('gender', '')}"  # first language, for a breakdown
            clips.append(Clip(f"svarah-{Path(shard).stem}-{i}", load(io.BytesIO(data).getvalue()), row[text_col], speaker))
    return clips


def dir_set(path: str) -> list[Clip]:
    from fastlane.speech.audio import load

    return [Clip(w.stem, load(w), w.with_suffix(".txt").read_text().strip()) for w in sorted(Path(path).glob("*.wav"))
            if w.with_suffix(".txt").exists()]


# ------------------------------------------------------------------ run
@dataclass
class Result:
    backend: str
    clips: int
    wer: float
    detail_accuracy: dict[str, float] = field(default_factory=dict)
    detail_counts: dict[str, int] = field(default_factory=dict)
    rtf_median: float = 0.0  # transcription time / audio time
    ms_median: float = 0.0


def evaluate(backend_spec: str, clips: list[Clip], entities: Entities | None, rows_out: Path | None = None) -> Result:
    import jiwer

    from fastlane.speech.stt import domain_prompt, get_stt

    spec, _, flag = backend_spec.partition("+")
    stt = get_stt(spec)
    prompt = None if flag == "noprompt" or spec.startswith("sarvam") else domain_prompt()
    refs, hyps, rtfs, mss = [], [], [], []
    hits: dict[str, list[int]] = {}
    rows = []
    if not spec.startswith("sarvam"):
        stt.transcribe(np.zeros(8000, dtype=np.float32))  # warm-up (local models compile kernels on first use)
    errors = 0
    for c in clips:
        try:
            t = stt.transcribe(c.audio, prompt=prompt)
        except Exception as e:  # one failed request must not end a long run; it is reported and left out
            errors += 1
            rows.append({"id": c.id, "speaker": c.speaker, "ref": c.reference, "hyp": None, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        ref, hyp = normalise(c.reference), normalise(t.text)
        refs.append(ref or "_")
        hyps.append(hyp)
        rtfs.append(t.ms / 1000 / max(t.audio_s, 1e-6))
        mss.append(t.ms)
        row = {"id": c.id, "speaker": c.speaker, "ref": c.reference, "hyp": t.text, "ms": round(t.ms)}
        if entities is not None:
            dh = detail_hits(entities.find(ref), hyp)
            row["details"] = dh
            for kind, (h, n) in dh.items():
                hits.setdefault(kind, [0, 0])
                hits[kind][0] += h
                hits[kind][1] += n
        rows.append(row)
    if rows_out is not None:
        rows_out.parent.mkdir(parents=True, exist_ok=True)
        rows_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    name = stt.name + ("" if prompt or spec.startswith("sarvam") else " (no prompt)")  # Sarvam takes no word list
    return Result(name, len(clips) - errors, round(jiwer.wer(refs, hyps), 4),
                  {k: round(h / n, 3) for k, (h, n) in hits.items() if n}, {k: n for k, (h, n) in hits.items()},
                  round(statistics.median(rtfs), 3), round(statistics.median(mss)))


def rescore(rows_path: Path, entities: Entities | None) -> dict:
    """Score saved rows again (after a change to normalise or the detail lists) without transcribing."""
    import jiwer

    rows = [r for r in (json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()) if r.get("hyp") is not None]
    refs = [normalise(r["ref"]) or "_" for r in rows]
    hyps = [normalise(r["hyp"]) for r in rows]
    hits: dict[str, list[int]] = {}
    for ref, hyp in zip(refs, hyps):
        if entities is None:
            continue
        for kind, (h, n) in detail_hits(entities.find(ref), hyp).items():
            hits.setdefault(kind, [0, 0])
            hits[kind][0] += h
            hits[kind][1] += n
    return {"rows": rows_path.name, "clips": len(rows), "wer": round(jiwer.wer(refs, hyps), 4),
            "detail_accuracy": {k: round(h / n, 3) for k, (h, n) in hits.items() if n}, "detail_counts": {k: n for k, (h, n) in hits.items()},
            "ms_median": round(statistics.median(r["ms"] for r in rows))}


def compare(set_spec: str, backends: list[str], n: int | None, out_dir: Path) -> list[dict]:
    if set_spec == "tts":
        clips, people = tts_set(n)
        entities = Entities.from_world(people)
    elif set_spec == "svarah":
        clips, entities = svarah_set(n or 300), None
    elif set_spec.startswith("dir:"):
        clips, entities = dir_set(set_spec[4:]), Entities.from_world([])
    else:
        raise ValueError(set_spec)
    results = []
    for b in backends:
        slug = re.sub(r"[^a-z0-9.]+", "-", b.lower())
        r = evaluate(b, clips, entities, out_dir / f"rows_{set_spec.split(':')[0]}_{slug}.jsonl")
        print(json.dumps(asdict(r)), flush=True)
        results.append(asdict(r))
    out = out_dir / f"wer_{set_spec.split(':')[0]}.json"
    earlier = json.loads(out.read_text()) if out.exists() else []
    names = {r["backend"] for r in results}
    merged = [r for r in earlier if r.get("backend") not in names] + results  # keep other backends' earlier results
    out.write_text(json.dumps(merged, indent=2))
    return results
