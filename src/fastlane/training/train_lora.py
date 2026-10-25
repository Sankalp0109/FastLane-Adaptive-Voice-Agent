"""QLoRA fine-tuning of Qwen2.5-1.5B/3B-Instruct for the FastLane small model (CUDA: Colab/Kaggle T4).

    python -m fastlane.training.train_lora --data data/train/all_domains --model Qwen/Qwen2.5-3B-Instruct \
        --out runs/lora-3b-all

Each example's last message is the target JSON; loss is computed on it only (prompt-completion
format), so the model learns the step output, not the conversation it is conditioned on.
On Apple Silicon use the MLX path instead (fastlane.training.mlx_train).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_split(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            ex = json.loads(line)
            msgs = ex["messages"]
            rows.append({"prompt": msgs[:-1], "completion": [msgs[-1]]})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory with train.jsonl and val.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--no-4bit", action="store_true", help="full-precision base (CPU/MPS smoke tests)")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N examples (smoke tests)")
    ap.add_argument("--merge", action="store_true", help="also save the base model merged with the adapter")
    ap.add_argument("--base-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto",
                    help="dtype for the non-quantised base weights (auto: float16 on CUDA, float32 on CPU)")
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data = Path(args.data)
    train_rows, val_rows = load_split(data / "train.jsonl"), load_split(data / "val.jsonl")
    if args.limit:
        train_rows, val_rows = train_rows[: args.limit], val_rows[: max(2, args.limit // 10)]
    train_ds, val_ds = Dataset.from_list(train_rows), Dataset.from_list(val_rows)

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in transformers 4.56
    base_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
        args.base_dtype, torch.float16 if cuda else torch.float32)
    model_kwargs: dict = {dtype_key: base_dtype}
    if cuda and not args.no_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if cuda:
        # Turing GPUs (T4) have no fast bfloat16: keep every non-quantised floating weight in float16.
        # 4-bit weights are stored as uint8 and are left alone.
        n_bf16 = 0
        for param in model.parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)
                n_bf16 += 1
        print(f"base model: {n_bf16} bfloat16 tensors moved to float16; "
              f"dtypes now {sorted({str(p.dtype) for p in model.parameters()})}", flush=True)

    peft_config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    import inspect
    import math

    steps_per_epoch = math.ceil(len(train_rows) / (args.batch * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    cfg_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)),  # explicit count: warmup_ratio was removed in transformers 5
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=cuda,
        fp16=cuda,
        max_length=args.max_len,
        completion_only_loss=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100 if args.max_steps < 0 else max(1, args.max_steps),
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        seed=2026,
    )
    # Library versions move fast (Kaggle installs the latest): drop options this version does not know.
    known = inspect.signature(SFTConfig.__init__).parameters
    dropped = sorted(k for k in cfg_kwargs if k not in known)
    if dropped:
        print(f"note: this TRL version has no {dropped}; continuing without them")
    cfg = SFTConfig(**{k: v for k, v in cfg_kwargs.items() if k in known})
    print(f"{len(train_rows)} train / {len(val_rows)} val examples, {total_steps} optimizer steps", flush=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok, peft_config=peft_config)
    # fp16 mixed precision keeps fp32 master copies of the trainable weights. Some library versions create
    # the LoRA weights in the checkpoint's bfloat16, which the fp16 grad scaler rejects (and a T4 has no
    # bfloat16), so make every trainable parameter float32 before the optimizer is built.
    cast = 0
    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
            cast += 1
    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    print(f"trainable tensors: {len(trainable)} ({sum(p.numel() for p in trainable) / 1e6:.1f}M params), "
          f"cast to float32: {cast}, dtypes: {sorted({str(p.dtype) for p in trainable})}", flush=True)
    trainer.train()
    adapter_dir = Path(args.out) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    metrics = trainer.evaluate()
    (Path(args.out) / "eval.json").write_text(json.dumps(metrics, indent=2))
    print(f"adapter -> {adapter_dir}; eval {metrics}")

    if args.merge:
        # The adapter is already saved; a merge failure (library version clashes) must not lose it.
        try:
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(args.model, **{dtype_key: torch.float16})
            merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
            merged_dir = Path(args.out) / "merged"
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tok.save_pretrained(merged_dir)
            print(f"merged model -> {merged_dir} (convert with llama.cpp convert_hf_to_gguf.py for Ollama)")
        except Exception as e:  # noqa: BLE001
            print(f"MERGE FAILED ({type(e).__name__}: {e}). The adapter is saved at {adapter_dir}; "
                  f"merge it locally with `fastlane train finish <adapter dir>`.", flush=True)
            sys.exit(3)


if __name__ == "__main__":
    main()
