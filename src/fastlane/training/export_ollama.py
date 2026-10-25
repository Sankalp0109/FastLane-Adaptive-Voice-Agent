"""Import a fine-tuned GGUF into Ollama with the base model's chat template.

    fastlane train ollama-import lora-...Q4_K_M.gguf --name fastlane-qwen2.5-3b --base qwen2.5:3b
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


def modelfile_for(gguf: Path, base: str) -> str:
    """The base model's Modelfile (TEMPLATE, stop tokens, parameters) with FROM pointing at our GGUF."""
    base_mf = subprocess.run(["ollama", "show", base, "--modelfile"], check=True, capture_output=True, text=True).stdout
    lines = [ln for ln in base_mf.splitlines() if not ln.startswith("#")]
    mf = "\n".join(lines)
    mf = re.sub(r"^FROM .*$", f"FROM {gguf.resolve()}", mf, count=1, flags=re.M)
    # Greedy decoding: routing and tool calls must be deterministic.
    mf = re.sub(r"^PARAMETER temperature .*$", "", mf, flags=re.M) + "\nPARAMETER temperature 0\n"
    return mf


def import_gguf(gguf: str | Path, name: str, base: str = "qwen2.5:3b", quantize: str | None = None) -> str:
    """`gguf` may be a .gguf file or a Hugging Face safetensors directory (Ollama converts it itself)."""
    gguf = Path(gguf)
    if not gguf.exists():
        raise FileNotFoundError(gguf)
    mf = modelfile_for(gguf, base)
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(mf)
        path = f.name
    cmd = ["ollama", "create", name, "-f", path]
    if quantize:
        cmd[2:2] = ["--quantize", quantize]  # Ollama quantises f16 weights on import
    subprocess.run(cmd, check=True)
    return name
