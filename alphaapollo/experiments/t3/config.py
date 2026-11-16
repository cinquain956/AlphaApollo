from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TargetSettings:
    name: str
    label: str
    problem: str
    instance: Dict[str, Any]
    known_best_objective: float
    source_note: str = ''


@dataclass(frozen=True)
class T3Config:
    t2_output_root: Path = PROJECT_ROOT / 'outputs' / 't2'
    output_root: Path = PROJECT_ROOT / 'outputs' / 't3'
    algorithms: List[str] = field(default_factory=lambda: ['openevolve_native', 'evox'])
    problems: List[str] = field(default_factory=lambda: ['circle_packing', 'mmd'])
    selected_seeds: Dict[str, Optional[int]] = field(default_factory=dict)
    execution_timeout: float = 30.0
    iterations: int = 32
    targets: Dict[str, List[TargetSettings]] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.t2_output_root.is_dir():
            raise FileNotFoundError(f'T2 output root not found: {self.t2_output_root}')
        for problem in self.problems:
            if problem not in self.targets:
                raise ValueError(f'missing T3 targets for {problem}')
            names = [target.name for target in self.targets[problem]]
            if len(names) != 5 or len(set(names)) != 5:
                raise ValueError(f'{problem} must define source plus four unique neighbours')
            if not any(name.endswith('_source') for name in names):
                raise ValueError(f'{problem} targets must include a *_source target')
            for target in self.targets[problem]:
                if target.problem != problem:
                    raise ValueError(f'target {target.name} has wrong problem {target.problem}')
                if target.known_best_objective <= 0:
                    raise ValueError(f'target {target.name} known best must be positive')


def _resolve_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_t3_config(path: str | Path = 'examples/t3/config.yaml') -> T3Config:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    data = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    experiment = data.get('experiment', {})
    targets: Dict[str, List[TargetSettings]] = {}
    for problem, items in (data.get('targets', {}) or {}).items():
        targets[str(problem)] = [
            TargetSettings(
                name=str(item['name']),
                label=str(item.get('label', item['name'])),
                problem=str(problem),
                instance=dict(item['instance']),
                known_best_objective=float(item['known_best_objective']),
                source_note=str(item.get('source_note', '')),
            )
            for item in items
        ]
    selected = {
        str(key): (None if value is None else int(value))
        for key, value in (experiment.get('selected_seeds', {}) or {}).items()
    }
    config = T3Config(
        t2_output_root=_resolve_path(str(experiment.get('t2_output_root', 'outputs/t2'))),
        output_root=_resolve_path(str(experiment.get('output_root', 'outputs/t3'))),
        algorithms=[str(item) for item in experiment.get('algorithms', ['openevolve_native', 'evox'])],
        problems=[str(item) for item in experiment.get('problems', ['circle_packing', 'mmd'])],
        selected_seeds=selected,
        execution_timeout=float(experiment.get('execution_timeout', 30.0)),
        iterations=int(experiment.get('iterations', 32)),
        targets=targets,
    )
    config.validate()
    return config
