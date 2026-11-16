from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from alphaapollo.program_actions import build_program_action_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass
SUPPORTED_ALGORITHMS = ("openevolve_native", "evox")
SUPPORTED_PROBLEMS = ("circle_packing", "mmd")


@dataclass(frozen=True)
class ModelSettings:
    name: str = "qwen3.8-27b"
    api_base: str = ""
    api_base_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout: int = 600
    retries: int = 2

    def resolved_api_base(self) -> str:
        return self.api_base or os.environ.get(self.api_base_env, "")

    def resolved_api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass(frozen=True)
class ProblemSettings:
    evaluator: Path
    initial_program: Path
    instance: Dict[str, Any]
    system_prompt: str
    execution_timeout: float = 30.0


@dataclass(frozen=True)
class T2Config:
    iterations: int = 32
    seeds: List[int] = field(default_factory=lambda: [11, 22, 33])
    algorithms: List[str] = field(
        default_factory=lambda: ["openevolve_native", "evox"]
    )
    problems: List[str] = field(
        default_factory=lambda: ["circle_packing", "mmd"]
    )
    output_root: Path = PROJECT_ROOT / "outputs" / "t2"
    max_workers: int = 3
    checkpoint_interval: int = 1
    evox_switch_interval: int = 8
    model: ModelSettings = field(default_factory=ModelSettings)
    problem_settings: Dict[str, ProblemSettings] = field(default_factory=dict)

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if len(self.seeds) != 3:
            raise ValueError("T2 requires exactly three random seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("T2 random seeds must be distinct")
        unknown_algorithms = set(self.algorithms) - set(SUPPORTED_ALGORITHMS)
        if unknown_algorithms:
            raise ValueError(f"unsupported algorithms: {sorted(unknown_algorithms)}")
        unknown_problems = set(self.problems) - set(SUPPORTED_PROBLEMS)
        if unknown_problems:
            raise ValueError(f"unsupported problems: {sorted(unknown_problems)}")
        missing = set(self.problems) - set(self.problem_settings)
        if missing:
            raise ValueError(f"missing problem configuration: {sorted(missing)}")
        for name in self.problems:
            problem = self.problem_settings[name]
            if not problem.evaluator.is_file():
                raise FileNotFoundError(f"evaluator not found: {problem.evaluator}")
            if not problem.initial_program.is_file():
                raise FileNotFoundError(
                    f"initial program not found: {problem.initial_program}"
                )

    def require_api_credentials(self) -> None:
        if not self.model.resolved_api_base():
            raise RuntimeError(
                f"missing API base; set {self.model.api_base_env} or model.api_base"
            )
        if not self.model.resolved_api_key():
            raise RuntimeError(f"missing API key; set {self.model.api_key_env}")

    def matrix_size(self) -> int:
        return len(self.algorithms) * len(self.problems) * len(self.seeds)


def _resolve_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_t2_config(path: str | Path) -> T2Config:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = data.get("experiment", {})
    model_data = data.get("model", {})
    model = ModelSettings(
        name=str(model_data.get("name", "qwen3.8-27b")),
        api_base=str(model_data.get("api_base", "")),
        api_base_env=str(model_data.get("api_base_env", "OPENAI_BASE_URL")),
        api_key_env=str(model_data.get("api_key_env", "OPENAI_API_KEY")),
        temperature=float(model_data.get("temperature", 0.7)),
        max_tokens=int(model_data.get("max_tokens", 8192)),
        timeout=int(model_data.get("timeout", 600)),
        retries=int(model_data.get("retries", 2)),
    )

    problem_settings: Dict[str, ProblemSettings] = {}
    for name, values in (data.get("problems", {}) or {}).items():
        problem_settings[str(name)] = ProblemSettings(
            evaluator=_resolve_path(str(values["evaluator"])),
            initial_program=_resolve_path(str(values["initial_program"])),
            instance=dict(values.get("instance", {})),
            system_prompt=(
                str(values.get("system_prompt", ""))
                or build_program_action_prompt(
                    str(values.get("task_instruction", "")), final_answer=False
                )
            ),
            execution_timeout=float(values.get("execution_timeout", 30.0)),
        )

    config = T2Config(
        iterations=int(experiment.get("iterations", 32)),
        seeds=[int(seed) for seed in experiment.get("seeds", [11, 22, 33])],
        algorithms=[
            str(value)
            for value in experiment.get(
                "algorithms", ["openevolve_native", "evox"]
            )
        ],
        problems=[
            str(value)
            for value in experiment.get("problems", ["circle_packing", "mmd"])
        ],
        output_root=_resolve_path(
            str(experiment.get("output_root", "outputs/t2"))
        ),
        max_workers=int(experiment.get("max_workers", 3)),
        checkpoint_interval=int(experiment.get("checkpoint_interval", 1)),
        evox_switch_interval=int(experiment.get("evox_switch_interval", 8)),
        model=model,
        problem_settings=problem_settings,
    )
    config.validate()
    return config


