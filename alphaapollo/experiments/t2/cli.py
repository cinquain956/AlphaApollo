from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict

from .config import load_t2_config
from .results import analyze_results
from .runner import T2Experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Task 2 SkyDiscover comparison experiment."
    )
    parser.add_argument(
        "command", choices=("dry-run", "run", "analyze"), nargs="?", default="dry-run"
    )
    parser.add_argument(
        "--config", default="examples/t2/config.yaml", help="T2 YAML configuration"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing run directories"
    )
    return parser


def main(argv=None) -> int:
    if argv is None and os.name == "nt" and not sys.flags.utf8_mode:
        return subprocess.call(
            [
                sys.executable,
                "-X",
                "utf8",
                "-m",
                "alphaapollo.experiments.t2",
                *sys.argv[1:],
            ]
        )
    args = build_parser().parse_args(argv)
    config = load_t2_config(args.config)
    experiment = T2Experiment(config)
    if args.command == "dry-run":
        payload = experiment.dry_run()
        config.output_root.mkdir(parents=True, exist_ok=True)
        (config.output_root / "dry_run.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.command == "run":
        summaries = experiment.run_all(overwrite=args.overwrite)
        analysis = analyze_results(config)
        print(
            json.dumps(
                {
                    "runs": [asdict(summary) for summary in summaries],
                    "analysis": analysis,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if all(summary.status == "completed" for summary in summaries) else 1
    print(json.dumps(analyze_results(config), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
