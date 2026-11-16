import math
from typing import Any, Dict, List

from .base import CheckResult, GeometryChecker, finite_float, parse_payload, violation


MMD_BENCHMARK = 1.0 / 4.165849767


class MMDChecker(GeometryChecker):
    '''Validate an n-point configuration and score its distance ratio.'''

    def __init__(
        self,
        n: int = 14,
        dimensions: int = 3,
        benchmark: float = MMD_BENCHMARK,
    ) -> None:
        self.n = int(n)
        self.dimensions = int(dimensions)
        self.benchmark = float(benchmark)
        if self.n < 2:
            raise ValueError('n must be at least 2')
        if self.dimensions < 1:
            raise ValueError('dimensions must be positive')
        if self.benchmark <= 0:
            raise ValueError('benchmark must be positive')

    def check(self, action: Any) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        try:
            payload = parse_payload(action)
        except ValueError as exc:
            return CheckResult(
                valid=False,
                score=0.0,
                objective=0.0,
                violations=[violation('parse_error', str(exc))],
            ).as_dict()

        points = payload.get('points') if isinstance(payload, dict) else payload
        if not isinstance(points, list):
            return CheckResult(
                valid=False,
                score=0.0,
                objective=0.0,
                violations=[violation('format', 'points must be a list')],
            ).as_dict()

        if len(points) != self.n:
            violations.append(
                violation(
                    'count',
                    f'expected {self.n} points, got {len(points)}',
                    expected=self.n,
                    actual=len(points),
                )
            )

        parsed = []
        for index, point in enumerate(points):
            if not isinstance(point, (list, tuple)):
                violations.append(
                    violation('point_format', f'point {index} must be a coordinate list', index=index)
                )
                continue
            if len(point) != self.dimensions:
                violations.append(
                    violation(
                        'dimension',
                        f'point {index} must have {self.dimensions} coordinates, got {len(point)}',
                        index=index,
                        expected=self.dimensions,
                        actual=len(point),
                    )
                )
                continue
            try:
                coordinates = tuple(
                    finite_float(value, f'point {index} coordinate {axis}')
                    for axis, value in enumerate(point)
                )
            except ValueError as exc:
                violations.append(violation('numeric', str(exc), index=index))
                continue
            parsed.append((index, coordinates))

        min_distance = math.inf
        max_distance = 0.0
        min_pair = None
        max_pair = None
        duplicate_pairs = []
        for left in range(len(parsed)):
            index_i, point_i = parsed[left]
            for right in range(left + 1, len(parsed)):
                index_j, point_j = parsed[right]
                distance = math.dist(point_i, point_j)
                if distance < min_distance:
                    min_distance = distance
                    min_pair = [index_i, index_j]
                if distance > max_distance:
                    max_distance = distance
                    max_pair = [index_i, index_j]
                if distance == 0.0:
                    duplicate_pairs.append([index_i, index_j])

        if duplicate_pairs:
            warnings.append(
                violation(
                    'duplicate_points',
                    'duplicate points are legal but force the objective to zero',
                    pairs=duplicate_pairs,
                )
            )

        valid = not violations
        if not valid or len(parsed) < 2 or max_distance == 0.0:
            ratio = 0.0
            reference_objective = 0.0
            score = 0.0
        else:
            ratio = min_distance / max_distance
            reference_objective = ratio * ratio
            score = reference_objective / self.benchmark

        return CheckResult(
            valid=valid,
            score=score,
            objective=ratio,
            violations=violations,
            warnings=warnings,
            metrics={
                'point_count': len(points),
                'parsed_point_count': len(parsed),
                'dimensions': self.dimensions,
                'min_distance': 0.0 if math.isinf(min_distance) else min_distance,
                'max_distance': max_distance,
                'min_pair': min_pair,
                'max_pair': max_pair,
                'distance_ratio': ratio,
                'reference_objective': reference_objective,
                'benchmark': self.benchmark,
            },
        ).as_dict()

