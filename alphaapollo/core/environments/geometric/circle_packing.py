import math
from typing import Any, Dict, List

from .base import CheckResult, GeometryChecker, finite_float, parse_payload, violation
from .containers import build_circle_container


CIRCLE_PACKING_BENCHMARK = 2.635
REFERENCE_TOLERANCE = 1e-6


class CirclePackingChecker(GeometryChecker):
    '''Validate circles packed in a configurable container.'''

    def __init__(
        self,
        n: int = 26,
        container_type: str = 'rectangle',
        width: float = 1.0,
        height: float = 1.0,
        side_length: float = 1.0,
        tolerance: float = REFERENCE_TOLERANCE,
        benchmark: float = CIRCLE_PACKING_BENCHMARK,
    ) -> None:
        self.n = int(n)
        self.container_type = str(container_type).lower()
        self.width = float(width)
        self.height = float(height)
        self.side_length = float(side_length)
        self.tolerance = float(tolerance)
        self.benchmark = float(benchmark)
        if self.n < 1:
            raise ValueError('n must be positive')
        if self.tolerance < 0:
            raise ValueError('tolerance must be non-negative')
        if self.benchmark <= 0:
            raise ValueError('benchmark must be positive')
        self.container = build_circle_container(
            container_type=self.container_type,
            width=self.width,
            height=self.height,
            side_length=self.side_length,
        )

    @staticmethod
    def _extract_circles(payload: Any) -> Any:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return None
        if 'circles' in payload:
            return payload['circles']
        centers = payload.get('centers')
        radii = payload.get('radii')
        if isinstance(centers, list) and isinstance(radii, list):
            if len(centers) != len(radii):
                return None
            return [
                {'x': center[0], 'y': center[1], 'r': radius}
                if isinstance(center, (list, tuple)) and len(center) == 2
                else center
                for center, radius in zip(centers, radii)
            ]
        return None

    def check(self, action: Any) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        try:
            payload = parse_payload(action)
        except ValueError as exc:
            return CheckResult(
                valid=False,
                score=0.0,
                objective=0.0,
                violations=[violation('parse_error', str(exc))],
            ).as_dict()

        circles = self._extract_circles(payload)
        if not isinstance(circles, list):
            return CheckResult(
                valid=False,
                score=0.0,
                objective=0.0,
                violations=[violation('format', 'expected a circles list or centers/radii lists')],
            ).as_dict()

        if len(circles) != self.n:
            violations.append(
                violation(
                    'count',
                    f'expected {self.n} circles, got {len(circles)}',
                    expected=self.n,
                    actual=len(circles),
                )
            )

        parsed = []
        for index, circle in enumerate(circles):
            if not isinstance(circle, dict):
                violations.append(
                    violation('circle_format', f'circle {index} must contain x, y, and r', index=index)
                )
                continue
            try:
                x = finite_float(circle.get('x'), f'circle {index} x')
                y = finite_float(circle.get('y'), f'circle {index} y')
                radius = finite_float(circle.get('r'), f'circle {index} radius')
            except ValueError as exc:
                violations.append(violation('numeric', str(exc), index=index))
                continue

            if radius < 0:
                violations.append(
                    violation('negative_radius', f'circle {index} has negative radius', index=index, radius=radius)
                )
                continue

            violations.extend(
                self.container.validate_circle(index, x, y, radius, self.tolerance)
            )
            parsed.append((index, x, y, radius))

        for left in range(len(parsed)):
            index_i, x_i, y_i, radius_i = parsed[left]
            for right in range(left + 1, len(parsed)):
                index_j, x_j, y_j, radius_j = parsed[right]
                distance = math.hypot(x_i - x_j, y_i - y_j)
                required = radius_i + radius_j
                if distance < required - self.tolerance:
                    violations.append(
                        violation(
                            'overlap',
                            f'circles {index_i} and {index_j} overlap',
                            indices=[index_i, index_j],
                            distance=distance,
                            required_distance=required,
                            penetration=required - distance,
                        )
                    )

        valid = not violations
        objective = sum(circle[3] for circle in parsed) if valid else 0.0
        return CheckResult(
            valid=valid,
            score=objective,
            objective=objective,
            violations=violations,
            metrics={
                'circle_count': len(circles),
                'parsed_circle_count': len(parsed),
                'radius_sum': objective,
                'reference_best': self.benchmark,
                'normalized_score': objective / self.benchmark if valid else 0.0,
                **self.container.as_metrics(),
            },
        ).as_dict()
