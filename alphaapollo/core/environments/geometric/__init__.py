"""Deterministic geometry environments and program evaluators."""

__all__ = [
    'CirclePackingChecker',
    'CirclePackingEnv',
    'MMDChecker',
    'MMDEnv',
    'PythonProgramExecutor',
    'ProblemSpec',
    'build_geometric_envs',
    'geometric_projection',
    'get_problem_spec',
]


def __getattr__(name):
    if name == 'CirclePackingChecker':
        from .circle_packing import CirclePackingChecker

        return CirclePackingChecker
    if name == 'MMDChecker':
        from .mmd import MMDChecker

        return MMDChecker
    if name in {'CirclePackingEnv', 'MMDEnv'}:
        from .env import CirclePackingEnv, MMDEnv

        return {'CirclePackingEnv': CirclePackingEnv, 'MMDEnv': MMDEnv}[name]
    if name == 'PythonProgramExecutor':
        from .executor import PythonProgramExecutor

        return PythonProgramExecutor
    if name in {'ProblemSpec', 'get_problem_spec'}:
        from .spec import ProblemSpec, get_problem_spec

        return {'ProblemSpec': ProblemSpec, 'get_problem_spec': get_problem_spec}[name]
    if name == 'build_geometric_envs':
        from .envs import build_geometric_envs

        return build_geometric_envs
    if name == 'geometric_projection':
        from .projection import geometric_projection

        return geometric_projection
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
