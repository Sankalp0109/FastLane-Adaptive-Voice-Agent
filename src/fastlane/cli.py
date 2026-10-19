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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastlane", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("systems", help="list the system presets (FastLane, baselines, variants)")
    p.set_defaults(func=_cmd_systems)

    world = sub.add_parser("world", help="practice booking world").add_subparsers(dest="world_command", required=True)
    p = world.add_parser("build", help="build the SQLite practice world")
    p.add_argument("--out", default=str(WORLD_DB))
    p.set_defaults(func=_cmd_world_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
