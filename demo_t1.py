#!/usr/bin/env python3
"""One-command Task 1 demonstration for program-based geometry environments."""

import importlib.util
import os

if importlib.util.find_spec('ray') is None:
    os.environ.setdefault('ALPHAAPOLLO_LIGHTWEIGHT_IMPORT', '1')


from alphaapollo.core.environments.geometric.circle_packing import CirclePackingChecker
from alphaapollo.core.environments.geometric.executor import PythonProgramExecutor
from alphaapollo.core.environments.geometric.mmd import MMDChecker
from alphaapollo.core.environments.geometric.projection import geometric_projection
from alphaapollo.core.environments.geometric.spec import get_problem_spec


CIRCLE_PROGRAM = '''
def solve(instance):
    n = instance["n"]
    columns = 5
    rows = 6
    centers = []
    for index in range(n):
        row, column = divmod(index, columns)
        centers.append([(column + 0.5) / columns, (row + 0.5) / rows])
    return {"centers": centers, "radii": [0.07] * n}
'''

MMD_PROGRAM = '''
def solve(instance):
    n = instance["n"]
    dimensions = instance["dimensions"]
    return [[(index + 1) ** (axis + 1) for axis in range(dimensions)] for index in range(n)]
'''


def evaluate(problem, program, instance, checker):
    projected, valid = geometric_projection([f'<python_code>{program}</python_code>'])
    if not valid[0]:
        raise RuntimeError('projection rejected the demonstration program')
    spec = get_problem_spec(problem)
    execution = PythonProgramExecutor(timeout_seconds=10).execute(
        projected[0],
        instance,
        spec.entrypoints(checker),
    )
    if not execution.ok:
        raise RuntimeError(execution.as_dict())
    payload = spec.normalize_output(execution.payload)
    result = checker.check(payload)
    return {
        'problem': problem,
        'entrypoint': execution.entrypoint,
        'valid': result['valid'],
        'objective': result['objective'],
        'score': result['score'],
        'violations': result['violations'],
        'runtime_seconds': execution.runtime_seconds,
    }


def main():
    circle_instance = {
        'n': 26,
        'container_type': 'rectangle',
        'width': 1.0,
        'height': 1.0,
    }
    mmd_instance = {'n': 14, 'dimensions': 3}
    results = [
        evaluate(
            'circle_packing',
            CIRCLE_PROGRAM,
            circle_instance,
            CirclePackingChecker(**circle_instance),
        ),
        evaluate('mmd', MMD_PROGRAM, mmd_instance, MMDChecker(**mmd_instance)),
    ]
    for result in results:
        print(
            f"{result['problem']}: valid={result['valid']}, "
            f"objective={result['objective']:.8g}, score={result['score']:.8g}, "
            f"entrypoint={result['entrypoint']}, runtime={result['runtime_seconds']:.4f}s"
        )
        if result['violations']:
            print(f"  violations={result['violations']}")


if __name__ == '__main__':
    main()




