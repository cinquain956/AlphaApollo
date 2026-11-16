import math


def solve(instance):
    """Parameterised deterministic point-set baseline."""
    n = int(instance["n"])
    dimensions = int(instance["dimensions"])
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(n):
        z = 1.0 - 2.0 * (index + 0.5) / n
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden_angle * index
        base = [radius * math.cos(angle), radius * math.sin(angle), z]
        if dimensions <= 3:
            point = base[:dimensions]
        else:
            point = base + [0.0] * (dimensions - 3)
        points.append(point)
    return {"points": points}
