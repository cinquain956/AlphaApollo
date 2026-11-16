import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from .base import GeometryChecker
from .circle_packing import CirclePackingChecker
from .mmd import MMDChecker
from .prompts import build_program_action_prompt


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    checker_type: type
    entrypoint_builder: Callable[[GeometryChecker], Tuple[str, ...]]
    normalize_output: Callable[[Any], Any]
    statement_builder: Callable[[GeometryChecker, Dict[str, Any]], str]

    def build_checker(self, params: Dict[str, Any]) -> GeometryChecker:
        signature = inspect.signature(self.checker_type)
        supported = {key: value for key, value in params.items() if key in signature.parameters}
        return self.checker_type(**supported)

    def entrypoints(self, checker: GeometryChecker) -> Tuple[str, ...]:
        return self.entrypoint_builder(checker)

    def problem_statement(self, checker: GeometryChecker, instance: Dict[str, Any]) -> str:
        return self.statement_builder(checker, instance)


def _normalize_circle_output(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return {'circles': value}
    if isinstance(value, (list, tuple)) and len(value) in (2, 3):
        centers, radii = value[:2]
        return {'centers': centers, 'radii': radii}
    return value


def _normalize_mmd_output(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    return {'points': value}


def _circle_statement(checker: CirclePackingChecker, instance: Dict[str, Any]) -> str:
    container = checker.container.as_metrics()
    task = (
        f'Write a complete Python program that packs exactly {checker.n} non-overlapping '
        f'circles and maximizes the sum of radii. Instance: {instance}. Container: {container}. '
        'Define solve(instance), read n and all container parameters from instance, and return '
        'either {"centers": [[x,y],...], "radii": [...]} or '
        '{"circles": [{"x":x,"y":y,"r":r},...]}. For SkyDiscover compatibility, '
        'run_packing() is also accepted.'
    )
    return build_program_action_prompt(task)


def _mmd_statement(checker: MMDChecker, instance: Dict[str, Any]) -> str:
    task = (
        f'Write a complete Python program that places exactly {checker.n} points in '
        f'{checker.dimensions}-dimensional space and maximizes minimum pairwise distance '
        f'divided by maximum pairwise distance. Instance: {instance}. Define solve(instance), '
        'read n and dimensions from instance, and return the points as a list, array, or '
        '{"points": [...]}. The SkyDiscover-style function '
        f'min_max_dist_dim{checker.dimensions}_{checker.n}() is also accepted.'
    )
    return build_program_action_prompt(task)


PROBLEM_SPECS = {
    'circle_packing': ProblemSpec(
        name='circle_packing',
        checker_type=CirclePackingChecker,
        entrypoint_builder=lambda checker: ('solve', 'run_packing'),
        normalize_output=_normalize_circle_output,
        statement_builder=_circle_statement,
    ),
    'mmd': ProblemSpec(
        name='mmd',
        checker_type=MMDChecker,
        entrypoint_builder=lambda checker: (
            'solve',
            f'min_max_dist_dim{checker.dimensions}_{checker.n}',
        ),
        normalize_output=_normalize_mmd_output,
        statement_builder=_mmd_statement,
    ),
}


def get_problem_spec(name: str) -> ProblemSpec:
    normalized = str(name).lower()
    try:
        return PROBLEM_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f'unsupported geometry problem: {name}') from exc



