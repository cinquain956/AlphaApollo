import math


def solve(instance):
    """Parameterised rectangular-grid baseline for any positive n."""
    n = int(instance["n"])
    width = float(instance.get("width", 1.0))
    height = float(instance.get("height", 1.0))
    columns = max(1, math.ceil(math.sqrt(n * width / height)))
    rows = max(1, math.ceil(n / columns))
    radius = min(width / (2.0 * columns), height / (2.0 * rows))
    centers = []
    for index in range(n):
        row, column = divmod(index, columns)
        centers.append(
            [
                (column + 0.5) * width / columns,
                (row + 0.5) * height / rows,
            ]
        )
    return {"centers": centers, "radii": [radius] * n}
