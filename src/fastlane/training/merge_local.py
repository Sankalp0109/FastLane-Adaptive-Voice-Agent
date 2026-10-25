"""Finish a trained adapter on this machine: merge it into the base model and import it into Ollama.

    fastlane train finish path/to/adapter --name fastlane-qwen2.5-3b

Used when training ran elsewhere (Kaggle/Colab) and only the LoRA adapter came back. Needs ~15 GB free disk
(base download + merged weights); Ollama converts the safetensors and quantises to Q4_K_M on import.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastlane.config import PROJECT_DIR
from fastlane.training.export_ollama import import_gguf

WORK = PROJECT_DIR / "runs"
LLAMA_CPP = WORK / "llama.cpp"


def merge(adapter_dir: Path, base: str, out_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {base} (float16, CPU) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    print(f"merged model -> {out_dir}", flush=True)
    return out_dir


def to_gguf(merged_dir: Path, out_file: Path) -> Path:
    if not (LLAMA_CPP / "convert_hf_to_gguf.py").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP)], check=True)
    # The converter imports llama.cpp's bundled gguf-py package; only these extras are needed.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "safetensors"], check=True)
    subprocess.run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(merged_dir),
                    "--outfile", str(out_file), "--outtype", "f16"], check=True)
    return out_file


def to_mlx(hf_dir: str | Path, out_dir: Path, bits: int = 4) -> Path:
    """Quantise Hugging Face weights to MLX (4-bit by default) with the installed mlx_lm package."""
    from fastlane.training.mlx_train import mlx_env

    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(hf_dir), "--mlx-path", str(out_dir),
                    "-q", "--q-bits", str(bits)], check=True, env=mlx_env())
    return out_dir


def finish_mlx(adapter_dir: str | Path, name: str, base: str = "Qwen/Qwen2.5-3B-Instruct", keep_merged: bool = False) -> Path:
    """Default route: merge the adapter with our own code, then quantise to 4-bit MLX for `fastlane serve`.
    (This Ollama version cannot import fine-tuned Qwen2 weights or LoRA adapters.)"""
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"{adapter_dir} has no adapter_config.json (point at the folder with adapter_model.safetensors)")
    merged_dir = WORK / f"{name}-merged"
    if not (merged_dir / "config.json").exists():
        merge(adapter_dir, base, merged_dir)
    out = to_mlx(merged_dir, WORK / f"mlx-{name}")
    if not keep_merged:
        import shutil

        shutil.rmtree(merged_dir, ignore_errors=True)
    return out


def finish(adapter_dir: str | Path, name: str, base: str = "Qwen/Qwen2.5-3B-Instruct", ollama_base: str = "qwen2.5:3b",
           keep_intermediate: bool = False, via_llama_cpp: bool = False) -> str:
    """Ollama route: merge, then let Ollama import the weights (Ollama builds that support the architecture),
    or convert with llama.cpp's script first (via_llama_cpp=True)."""
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"{adapter_dir} has no adapter_config.json (point at the folder with adapter_model.safetensors)")
    merged_dir = WORK / f"{name}-merged"
    if not (merged_dir / "config.json").exists():
        merge(adapter_dir, base, merged_dir)
    source = merged_dir
    if via_llama_cpp:
        source = to_gguf(merged_dir, WORK / f"{name}.f16.gguf")
    created = import_gguf(source, name, ollama_base, quantize="q4_K_M")
    if not keep_intermediate:
        import shutil

        shutil.rmtree(merged_dir, ignore_errors=True)
        (WORK / f"{name}.f16.gguf").unlink(missing_ok=True)
    return created
