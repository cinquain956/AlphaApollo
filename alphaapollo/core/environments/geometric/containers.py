import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .base import violation


class CircleContainer(ABC):
    @abstractmethod
    def validate_circle(
        self,
        index: int,
        x: float,
        y: float,
        radius: float,
        tolerance: float,
    ) -> List[Dict[str, Any]]:
        """Return boundary violations for one circle."""

    @abstractmethod
    def as_metrics(self) -> Dict[str, Any]:
        """Return serializable container metadata."""


class RectangleContainer(CircleContainer):
    def __init__(self, width: float = 1.0, height: float = 1.0) -> None:
        self.width = float(width)
        self.height = float(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError('width and height must be positive')

    def validate_circle(self, index, x, y, radius, tolerance):
        checks = (
            ('left', x - radius < -tolerance, radius - x),
            ('right', x + radius > self.width + tolerance, x + radius - self.width),
            ('bottom', y - radius < -tolerance, radius - y),
            ('top', y + radius > self.height + tolerance, y + radius - self.height),
        )
        return [
            violation(
                'boundary',
                f'circle {index} crosses the {boundary} boundary',
                index=index,
                boundary=boundary,
                excess=max(0.0, excess),
            )
            for boundary, outside, excess in checks
            if outside
        ]

    def as_metrics(self):
        return {
            'container_type': 'rectangle',
            'width': self.width,
            'height': self.height,
        }


class EquilateralTriangleContainer(CircleContainer):
    """Equilateral triangle with vertices (0,0), (side,0), (side/2,h)."""

    def __init__(self, side_length: float = 1.0) -> None:
        self.side_length = float(side_length)
        if self.side_length <= 0:
            raise ValueError('side_length must be positive')
        self.height = math.sqrt(3.0) * self.side_length / 2.0

    def validate_circle(self, index, x, y, radius, tolerance):
        distances = {
            'bottom': y,
            'left': (math.sqrt(3.0) * x - y) / 2.0,
            'right': (math.sqrt(3.0) * (self.side_length - x) - y) / 2.0,
        }
        result = []
        for boundary, signed_distance in distances.items():
            if signed_distance < radius - tolerance:
                result.append(
                    violation(
                        'boundary',
                        f'circle {index} crosses the {boundary} boundary',
                        index=index,
                        boundary=boundary,
                        signed_distance=signed_distance,
                        required_distance=radius,
                        excess=radius - signed_distance,
                    )
                )
        return result

    def as_metrics(self):
        return {
            'container_type': 'equilateral_triangle',
            'side_length': self.side_length,
            'height': self.height,
        }


def build_circle_container(
    container_type: str = 'rectangle',
    width: float = 1.0,
    height: float = 1.0,
    side_length: float = 1.0,
) -> CircleContainer:
    normalized = str(container_type).lower()
    if normalized in {'rectangle', 'rect', 'square'}:
        return RectangleContainer(width=width, height=height)
    if normalized in {'equilateral_triangle', 'triangle'}:
        return EquilateralTriangleContainer(side_length=side_length)
    raise ValueError(f'unsupported circle container: {container_type}')
