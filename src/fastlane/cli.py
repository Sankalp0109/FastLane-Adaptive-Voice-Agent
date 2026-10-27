"""Command-line entry point: `fastlane <command>`."""

from __future__ import annotations

import argparse
import sys

from fastlane.config import SYSTEMS, WORLD_DB


def _cmd_systems(args: argparse.Namespace) -> int:
    for cfg in SYSTEMS.values():
        print(f"{cfg.name:9s} {cfg.description}")
    return 0


def _cmd_world_build(args: argparse.Namespace) -> int:
    from fastlane.world.seed import build_world

    path = build_world(args.out)
    from fastlane.world.db import World

    with World.open(path) as world:
        for table, n in world.table_counts().items():
            print(f"{table:22s} {n:>8d}")
    print(f"world written to {path}")
    return 0


def _cmd_docs_build(args: argparse.Namespace) -> int:
    from fastlane.retrieval.build_docs import build_docs
    from fastlane.retrieval.index import load_chunks

    for path in build_docs():
        print(f"wrote {path.name}")
    chunks = load_chunks()
    words = sum(len(c.text.split()) for c in chunks)
    print(f"{len({c.doc_id for c in chunks})} documents, {len(chunks)} chunks, {words} words (~{words // 450} pages)")
    return 0


def _cmd_docs_search(args: argparse.Namespace) -> int:
    import json

    from fastlane.retrieval.index import PolicyIndex

    print(json.dumps(PolicyIndex().search(args.query, args.domain), indent=2))
    return 0


def build_agent(system: str, phone: str, world=None):
    from fastlane.agent.orchestrator import Agent
    from fastlane.config import get_system
    from fastlane.models.registry import client_for, small_client
    from fastlane.world.db import World

    cfg = get_system(system)
    small, compact = small_client(cfg)
    large = client_for("large") if cfg.large_model else None
    return Agent(world or World.fresh(), phone, cfg, small=small, large=large, compact_small_prompt=compact)


def _cmd_chat(args: argparse.Namespace) -> int:
    agent = build_agent(args.system, args.phone)
    print(f"[{args.system}] small={getattr(agent.small, 'name', None)} large={getattr(agent.large, 'name', None)}")
    print("agent>", agent.greet())
    lines = iter(args.say) if args.say else None
    while not agent.ended:
        try:
            text = next(lines) if lines is not None else input("you> ")
        except (StopIteration, EOFError):
            break
        if lines is not None:
            print("you>", text)
        reply = agent.respond(text)
        tr = agent.traces[-1]
        print("agent>", reply)
        if args.verbose:
            print(f"   [{tr.handled_by}] model {tr.model_ms:.0f} ms, wall {tr.wall_ms:.0f} ms"
                  + (f", why: {tr.escalation_reason}" if tr.escalation_reason else "")
                  + (f", tools: {[(t.tool, t.error or 'ok') for t in tr.tools]}" if tr.tools else ""))
    agent.close()
    return 0


def _cmd_speech_wer(args: argparse.Namespace) -> int:
    from pathlib import Path

    from fastlane.speech.wer import compare

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    compare(args.set, args.backends, args.n, out)
    return 0


def _cmd_voice(args: argparse.Namespace) -> int:
    from fastlane.speech.server import serve

    serve(args.system, args.stt, args.tts, args.phone, port=args.port, silence_ms=args.silence_ms, use_prompt=not args.no_prompt)
    return 0


def _cmd_testset_build(args: argparse.Namespace) -> int:
    from collections import Counter

    from fastlane.eval.scenarios import TEST_CALLS, build_test_calls

    scripts = build_test_calls()
    counts = Counter((s.domain, s.category) for s in scripts)
    for (d, c), n in sorted(counts.items()):
        print(f"{d:12s} {c:12s} {n:>4d}")
    print(f"{len(scripts)} test calls -> {TEST_CALLS}")
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from fastlane.config import get_system
    from fastlane.eval.runner import load_test_calls, run, sample_calls
    from fastlane.models.registry import client_for, small_client

    calls = sample_calls(load_test_calls(), args.calls, args.category, args.domain)
    if args.only:
        calls = [c for c in calls if c.call_id in set(args.only)]
    caller = client_for(args.caller)
    for system in args.system:
        cfg = get_system(system)
        small, compact = small_client(cfg)
        large = client_for("large") if cfg.large_model else None
        run(system, calls, list(range(args.seeds)), args.run, small, large, caller, compact_small_prompt=compact)
    return 0


def _cmd_eval_report(args: argparse.Namespace) -> int:
    from fastlane.eval.report import write_report

    path = write_report(args.run)
    print(path.read_text().split("\n## ")[0])
    print(f"report -> {path}")
    return 0


def _cmd_train_data(args: argparse.Namespace) -> int:
    import json

    from fastlane.training.build_dataset import TRAIN_DIR, build

    report = build(args.plans)
    print(json.dumps({k: v for k, v in report.items()}, indent=2))
    print(f"-> {TRAIN_DIR}")
    return 0


def _cmd_train_mlx(args: argparse.Namespace) -> int:
    from fastlane.training.mlx_train import train

    out = train(args.adapter, args.model, args.iters, batch=args.batch, lr=args.lr, layers=args.layers,
                max_seq=args.max_seq, rank=args.rank, dry_run=args.dry_run)
    print(f"adapter -> {out / 'adapter'}")
    return 0


def _cmd_ollama_import(args: argparse.Namespace) -> int:
    from fastlane.training.export_ollama import import_gguf

    print(f"ollama model created: {import_gguf(args.gguf, args.name, args.base)}")
    return 0


def _cmd_colab_bundle(args: argparse.Namespace) -> int:
    import zipfile

    from fastlane.config import PACKAGE_DIR
    from fastlane.training.build_dataset import TRAIN_DIR

    out = TRAIN_DIR / "colab_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for adapter in ("all_domains", "movies_buses"):
            for name in ("train.jsonl", "val.jsonl", "stats.json"):
                z.write(TRAIN_DIR / adapter / name, f"data/{adapter}/{name}")
        z.write(PACKAGE_DIR / "training" / "train_lora.py", "train_lora.py")
    print(f"upload this to the notebook: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def _cmd_train_finish(args: argparse.Namespace) -> int:
    if args.ollama:
        from fastlane.training.merge_local import finish

        print(f"ollama model created: {finish(args.adapter, args.name, args.base, args.ollama_base, args.keep, args.via_llama_cpp)}")
        return 0
    from fastlane.training.merge_local import finish_mlx

    out = finish_mlx(args.adapter, args.name, args.base, args.keep)
    print(f"MLX model -> {out}\nserve it with: fastlane serve   (then use spec mlx:{out.relative_to(out.parents[1])})")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import subprocess
    import sys

    from fastlane.training.mlx_train import mlx_env

    print(f"MLX server on http://127.0.0.1:{args.port}/v1 (default model {args.model}; other paths load on request)", flush=True)
    return subprocess.call([sys.executable, "-m", "mlx_lm", "server", "--model", args.model, "--port", str(args.port),
                            "--host", "127.0.0.1"], env=mlx_env())


def _cmd_train_check(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from fastlane.training.build_dataset import TRAIN_DIR
    from fastlane.training.check_model import check, summarise

    if args.rescore:
        rows = [json.loads(l) for l in Path(args.rescore).read_text().splitlines() if l.strip()]
        if args.domain:
            rows = [r for r in rows if r["meta"]["domain"] in args.domain]
        result = summarise(rows, Path(args.rescore).stem)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0
    val = Path(args.val) if args.val else TRAIN_DIR / "all_domains" / "val.jsonl"
    where = None
    if args.domain:
        where = lambda m: m["domain"] in args.domain  # noqa: E731
    result = check(args.model, val, args.n, args.full_prompt, save_rows=Path(args.save_rows) if args.save_rows else None, where=where)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def _cmd_label_sheet(args: argparse.Namespace) -> int:
    from fastlane.training.build_dataset import TRAIN_DIR
    from fastlane.training.kappa import export_sheet

    print(f"label sheet -> {export_sheet(TRAIN_DIR / 'all_domains', TRAIN_DIR / 'label_sheet.csv', n=args.n)}")
    return 0


def _cmd_kappa(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from fastlane.training.kappa import report

    print(json.dumps(report(Path(args.sheet)), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastlane", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("systems", help="list the system presets (FastLane, baselines, variants)")
    p.set_defaults(func=_cmd_systems)

    world = sub.add_parser("world", help="practice booking world").add_subparsers(dest="world_command", required=True)
    p = world.add_parser("build", help="build the SQLite practice world")
    p.add_argument("--out", default=str(WORLD_DB))
    p.set_defaults(func=_cmd_world_build)

    docs = sub.add_parser("docs", help="policy documents").add_subparsers(dest="docs_command", required=True)
    p = docs.add_parser("build", help="regenerate the generated policy documents")
    p.set_defaults(func=_cmd_docs_build)
    p = docs.add_parser("search", help="search the policy index")
    p.add_argument("query")
    p.add_argument("--domain", default=None)
    p.set_defaults(func=_cmd_docs_search)

    ts = sub.add_parser("testset", help="the 360 scripted test calls").add_subparsers(dest="testset_command", required=True)
    p = ts.add_parser("build", help="generate data/test_calls.jsonl from the world")
    p.set_defaults(func=_cmd_testset_build)

    ev = sub.add_parser("eval", help="run simulated calls and report metrics").add_subparsers(dest="eval_command", required=True)
    p = ev.add_parser("run", help="run systems on test calls (resumable)")
    p.add_argument("--system", nargs="+", default=["fastlane"])
    p.add_argument("--run", default="pilot", help="run name; results go to data/runs/<run>/")
    p.add_argument("--calls", type=int, default=None, help="stratified sample size (default: all 360)")
    p.add_argument("--category", nargs="*", default=None)
    p.add_argument("--domain", nargs="*", default=None)
    p.add_argument("--only", nargs="*", default=None, help="specific call ids")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--caller", default="caller", help="caller simulator model role or provider:model")
    p.set_defaults(func=_cmd_eval_run)
    p = ev.add_parser("report", help="summarise a run into REPORT.md, summary.json and a plot")
    p.add_argument("--run", default="pilot")
    p.set_defaults(func=_cmd_eval_report)

    tr = sub.add_parser("train", help="training data and LoRA").add_subparsers(dest="train_command", required=True)
    p = tr.add_parser("data", help="generate practice-call training data for the small model")
    p.add_argument("--plans", type=int, default=650)
    p.set_defaults(func=_cmd_train_data)
    p = tr.add_parser("mlx", help="LoRA fine-tuning on Apple Silicon with MLX")
    p.add_argument("--adapter", default="all_domains", choices=["all_domains", "movies_buses"])
    p.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    p.add_argument("--iters", type=int, default=1800, help="~2 epochs at batch 2 for ~3.7k examples")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_train_mlx)
    p = tr.add_parser("finish", help="merge a downloaded LoRA adapter, convert to GGUF and import into Ollama")
    p.add_argument("adapter", help="folder with adapter_config.json and adapter_model.safetensors")
    p.add_argument("--name", default="fastlane-qwen2.5-3b")
    p.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--ollama-base", default="qwen2.5:3b", help="Ollama model whose chat template to reuse")
    p.add_argument("--keep", action="store_true", help="keep the merged weights")
    p.add_argument("--via-llama-cpp", action="store_true", help="with --ollama: convert with llama.cpp's script first")
    p.add_argument("--ollama", action="store_true", help="import into Ollama instead of building an MLX model")
    p.set_defaults(func=_cmd_train_finish)
    p = tr.add_parser("ollama-import", help="import a fine-tuned GGUF into Ollama")
    p.add_argument("gguf")
    p.add_argument("--name", default="fastlane-qwen2.5-3b")
    p.add_argument("--base", default="qwen2.5:3b")
    p.set_defaults(func=_cmd_ollama_import)
    p = tr.add_parser("colab-bundle", help="zip the training data + script for the Colab/Kaggle notebook")
    p.set_defaults(func=_cmd_colab_bundle)
    p = tr.add_parser("check", help="offline step accuracy of a small model on held-out validation steps")
    p.add_argument("--model", default="ollama:fastlane-qwen2.5-3b")
    p.add_argument("--val", default=None)
    p.add_argument("--n", type=int, default=None, help="random sample of validation steps")
    p.add_argument("--full-prompt", action="store_true", help="use the full instruction prompt (for untuned models)")
    p.add_argument("--domain", nargs="*", default=None, help="only steps from these domains")
    p.add_argument("--out", default=None, help="write the summary JSON here")
    p.add_argument("--save-rows", default=None, help="write per-step results (JSONL) here")
    p.add_argument("--rescore", default=None, help="score a saved --save-rows file again instead of calling a model")
    p.set_defaults(func=_cmd_train_check)
    p = tr.add_parser("label-sheet", help="export 300 turns for two annotators (Cohen's kappa)")
    p.add_argument("--n", type=int, default=300)
    p.set_defaults(func=_cmd_label_sheet)
    p = tr.add_parser("kappa", help="Cohen's kappa from a filled label sheet")
    p.add_argument("sheet")
    p.set_defaults(func=_cmd_kappa)

    p = sub.add_parser("serve", help="serve the small models with the local MLX server (Apple Silicon)")
    p.add_argument("--model", default="runs/mlx-fastlane-qwen2.5-3b")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("chat", help="talk to an agent system in text")
    p.add_argument("--system", default="fastlane")
    p.add_argument("--phone", default="9876500001", help="caller ID of the simulated caller")
    p.add_argument("--say", nargs="*", help="non-interactive: caller lines to send in order")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_cmd_chat)

    sp = sub.add_parser("speech", help="speech tools (Phase 7)").add_subparsers(dest="speech_command", required=True)
    p = sp.add_parser("wer", help="compare STT backends: WER and key-detail accuracy")
    p.add_argument("--set", default="tts", help="tts | svarah | dir:PATH (wav + txt pairs)")
    p.add_argument("--backends", nargs="+", default=["mlx", "mlx+noprompt", "faster-whisper:small.en"],
                   help="STT specs; add '+noprompt' to drop the domain word list")
    p.add_argument("--n", type=int, default=None, help="number of clips (svarah default 300)")
    p.add_argument("--out", default="results/stt_comparison")
    p.set_defaults(func=_cmd_speech_wer)

    p = sub.add_parser("voice", help="browser voice demo: mic -> VAD -> STT -> agent -> TTS (Phase 7)")
    p.add_argument("--system", default="fastlane")
    p.add_argument("--phone", default="9876500001", help="caller ID the call comes from")
    p.add_argument("--stt", default="mlx", help="mlx[:repo] | faster-whisper[:size] | sarvam[:model]")
    p.add_argument("--tts", default="kokoro", help="kokoro[:voice] | sarvam[:speaker]")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--silence-ms", type=int, default=600, help="pause that ends the caller's turn")
    p.add_argument("--no-prompt", action="store_true", help="do not give Whisper the list of film/cinema/city names")
    p.set_defaults(func=_cmd_voice)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
