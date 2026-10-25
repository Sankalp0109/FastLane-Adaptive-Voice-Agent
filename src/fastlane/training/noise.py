"""Speech-recognition noise for caller lines (Table 5: "speech-noise copies, ~30% of caller lines").

The proposal's method is a TTS -> STT round trip; that needs the Phase 7 speech stack. Until then
these rule-based perturbations imitate the errors recognisers make on Indian-English calls: names and
places misheard, number words confused, small words dropped, no punctuation or casing.
"""

from __future__ import annotations

import random
import re

MISHEARD = {
    "Kukatpally": ["Kukat pally", "Kukatpalli", "kuka pally"], "Gachibowli": ["Gachi bowli", "gachibouli"],
    "Ameerpet": ["Amirpet", "Ameer pet"], "Secunderabad": ["Sikandarabad", "Secundrabad"],
    "Bengaluru": ["Bangalore", "Bengal uru"], "Vijayawada": ["Vijaywada", "vijay wada"], "Chennai": ["Chenai", "Chennai"],
    "Jubilee": ["Jubli", "Jubilee"], "Banjara": ["Banjaara", "Banjar"], "Hitech": ["High tech", "Hi-tech"],
    "Samudram": ["Samudhram", "Samudram"], "Kaala": ["Kala", "Kaala"], "Dastak": ["Dustak", "Dastak"],
    "Godavari": ["Godawari", "Godavri"], "Dhamaka": ["Dhamakaa", "Damaka"], "Diwali": ["Divali", "Deewali"],
    "Dussehra": ["Dasara", "Dushera"], "Tandoor": ["Tandur", "Tandoor"], "recliner": ["recliners", "reclining"],
    "prime": ["prime", "pry me"], "sleeper": ["sleeper", "sleepar"],
}
NUMBER_SWAPS = {"two": ["to", "2"], "four": ["for", "4"], "eight": ["ate", "8"], "three": ["tree", "3"], "2": ["two", "to"],
                "4": ["four", "for"], "1": ["one", "won"], "8": ["eight"], "6": ["six"]}
DROPPABLE = {"the", "a", "please", "for", "on", "at", "is", "ji", "na"}


def asr_noise(text: str, rng: random.Random, n_edits: int | None = None) -> str:
    """Return a plausible mis-transcription of `text` (1-3 edits, lower-case, no punctuation)."""
    words = text.split()
    if not words:
        return text
    n_edits = n_edits or rng.choice([1, 1, 2, 3])
    for _ in range(n_edits):
        kind = rng.choice(["name", "name", "number", "drop", "split"])
        idx = list(range(len(words)))
        rng.shuffle(idx)
        for i in idx:
            w = words[i]
            core = re.sub(r"[^\w-]", "", w)
            if kind == "name":
                key = next((k for k in MISHEARD if k.lower() == core.lower()), None)
                if key:
                    words[i] = rng.choice(MISHEARD[key])
                    break
            elif kind == "number" and core.lower() in NUMBER_SWAPS:
                words[i] = rng.choice(NUMBER_SWAPS[core.lower()])
                break
            elif kind == "drop" and core.lower() in DROPPABLE and len(words) > 3:
                words.pop(i)
                break
            elif kind == "split" and len(core) > 8 and core.isalpha():
                cut = rng.randint(3, len(core) - 3)
                words[i] = core[:cut] + " " + core[cut:]
                break
    out = " ".join(words)
    out = re.sub(r"[,.!?;]", "", out)
    return out.lower() if rng.random() < 0.7 else out


def noisy_user_content(content: str, rng: random.Random, p_line: float = 1.0) -> str:
    """Apply noise to the 'Caller: ...' lines of a small-model user message."""
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Caller: ") and rng.random() < p_line:
            lines[i] = "Caller: " + asr_noise(line[len("Caller: "):], rng)
    return "\n".join(lines)
