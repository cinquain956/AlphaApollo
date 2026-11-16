from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .config import load_t3_config
from .runner import T3Experiment, choose_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Replay T2 programs on Task 3 neighbour instances.')
    parser.add_argument('command', choices=('select', 'run'), nargs='?', default='select')
    parser.add_argument('--config', default='examples/t3/config.yaml')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(argv=None) -> int:
    if argv is None and os.name == 'nt' and not sys.flags.utf8_mode:
        return subprocess.call([sys.executable, '-X', 'utf8', '-m', 'alphaapollo.experiments.t3', *sys.argv[1:]])
    args = build_parser().parse_args(argv)
    config = load_t3_config(args.config)
    experiment = T3Experiment(config)
    if args.command == 'select':
        choices = [
            choose_trajectory(config, algorithm, problem).__dict__
            for algorithm in config.algorithms
            for problem in config.problems
        ]
        print(json.dumps({'choices': choices}, indent=2, ensure_ascii=False))
        return 0
    result = experiment.run_all(overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
