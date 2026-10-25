"""LoRA fine-tuning on Apple Silicon with MLX (no CUDA, no upload needed).

    fastlane train mlx --adapter all_domains --model mlx-community/Qwen2.5-3B-Instruct-4bit

Writes the MLX data layout (train.jsonl / valid.jsonl with "messages" only), runs mlx_lm's LoRA
trainer with the prompt masked (loss on the target JSON only), and saves the adapter under
runs/mlx-<adapter>-<model>/. Serve it with:

    python -m mlx_lm server --model <model> --adapter-path <adapter dir> --port 8080

and point the small-model role at it: FASTLANE_SMALL_LORA=mlx:<model>. (With an Anaconda-based Python,
prefix MLX commands with MLX_MPI_LIBNAME=/nonexistent/libmpi.dylib; see mlx_env.)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastlane.config import PROJECT_DIR
from fastlane.training.build_dataset import TRAIN_DIR

RUNS = PROJECT_DIR / "runs"


def write_mlx_data(adapter: str) -> Path:
    src = TRAIN_DIR / adapter
    dst = src / "mlx"
    dst.mkdir(parents=True, exist_ok=True)
    for name, out in (("train.jsonl", "train.jsonl"), ("val.jsonl", "valid.jsonl")):
        with (dst / out).open("w") as f:
            for line in (src / name).read_text().splitlines():
                if line.strip():
                    f.write(json.dumps({"messages": json.loads(line)["messages"]}, ensure_ascii=False) + "\n")
    return dst


def train(adapter: str, model: str, iters: int, batch: int = 2, lr: float = 1e-4, layers: int = 16,
          max_seq: int = 2048, rank: int = 16, dry_run: bool = False) -> Path:
    data = write_mlx_data(adapter)
    out = RUNS / f"mlx-{adapter}-{model.split('/')[-1]}"
    out.mkdir(parents=True, exist_ok=True)
    config = out / "lora_config.yaml"
    config.write_text(
        f"model: {model}\ntrain: true\ndata: {data}\nfine_tune_type: lora\nmask_prompt: true\n"
        f"num_layers: {layers}\nbatch_size: {batch}\niters: {iters}\nlearning_rate: {lr}\nmax_seq_length: {max_seq}\n"
        f"grad_checkpoint: true\nsteps_per_report: 10\nsteps_per_eval: 100\nval_batches: 25\nsave_every: 200\n"
        f"adapter_path: {out / 'adapter'}\nseed: 2026\n"
        f"lora_parameters:\n  rank: {rank}\n  scale: 2.0\n  dropout: 0.05\n"
    )
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config)]
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=mlx_env())
    return out


def mlx_env() -> dict:
    """Single-machine MLX must not load an MPI library. Anaconda ships MPICH, which MLX rejects
    (it requires Open MPI) and aborts on, so point MLX's MPI loader at nothing."""
    import os

    return {**os.environ, "MLX_MPI_LIBNAME": "/nonexistent/libmpi.dylib"}
