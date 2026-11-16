# Task 1: Program-based Circle Packing and MMD environments

Task 1 adds two deterministic geometry environments to AlphaApollo:

- `circle_packing`: default source instance `n=26` in a unit square.
- `mmd`: default source instance `n=14`, `dimensions=3`.

The environment uses AlphaApollo-style XML actions while evaluating complete Python programs rather than final JSON answers. This is required by Task 2, which evolves and stores programs, and Task 3, which re-runs those programs on neighboring instances.

## Evaluation pipeline

```text
LLM output (<python_code> candidate or <answer> final program)
  -> geometric_projection (extract and syntax-check one Python program)
  -> PythonProgramExecutor (temporary subprocess, timeout, exception capture)
  -> ProblemSpec output adapter
  -> deterministic Checker
  -> feedback, reward, and trajectory_record
```

The subprocess boundary prevents a candidate from corrupting the environment process, but it is not a security sandbox. Production execution of untrusted programs should use the same external sandbox used by the wider AlphaApollo pipeline.

## Candidate program interface

The preferred candidate action is `<python_code>...</python_code>`. The final candidate may use `<answer>...</answer>`. Plain Python remains accepted for backward compatibility. The program interface for both problems is:

```python
def solve(instance):
    ...
    return result
```

`instance` contains replaceable parameters such as `n`, `dimensions`, `width`, `height`, or `container_type`. Programs should read these values rather than hardcoding the source instance.

For compatibility with SkyDiscover, the following entrypoints are also accepted:

- Circle Packing: `run_packing()`; return `(centers, radii)`.
- MMD: `min_max_dist_dim{dimensions}_{n}()`; return the point array.

Circle results may be `{"centers": ..., "radii": ...}` or `{"circles": ...}`. MMD results may be a point list/array or `{"points": ...}`.

## Separation of responsibilities

- `spec.py`: problem registry, entrypoints, prompts, output adapters.
- `executor.py` and `executor_worker.py`: generic program execution.
- `env.py`: problem-independent reset, execution, feedback, and trajectory metadata.
- `circle_packing.py` and `mmd.py`: deterministic domain validation and scoring.
- `containers.py`: rectangle and equilateral-triangle circle constraints.

Adding another problem or container no longer requires adding problem-specific branches to the environment class.

## Scoring

Circle Packing returns the raw sum of radii as both objective and score, matching the SkyDiscover evaluator. The known source-instance best value (`2.635`) is retained only as `metrics.normalized_score` for analysis.

MMD exposes `min_distance / max_distance` as the human-readable objective. Its evaluator score matches SkyDiscover:

```text
score = (min_distance / max_distance)^2 / (1 / 4.165849767)
```

Task 3 should perform instance-specific normalization in the experiment/analysis layer rather than embedding neighboring-instance optima in the Checker.

## Feedback

Invalid candidates receive structured violations. Circle Packing reports count errors, non-finite coordinates, negative radii, boundary names, overlapping indices, distances, and penetration. MMD reports exact count, dimension, and coordinate errors. Execution failures and timeouts are reported as `execution_error` violations.

Each step exposes a serializable `trajectory_record` containing:

```text
problem, iteration, action_kind, raw_action, source_code, instance_params,
execution status/runtime/entrypoint, candidate_output,
valid, objective, score, violations, warnings, metrics
```

This record is the persistence boundary for Task 2 and the replay input for Task 3.

## Configuration

```yaml
env:
  env_name: circle_packing
  max_steps: 32
  circle_packing:
    n: 26
    container_type: rectangle
    width: 1.0
    height: 1.0
    tolerance: 1.0e-6
    benchmark: 2.635
    execution_timeout: 30
```

MMD:

```yaml
env:
  env_name: mmd
  max_steps: 32
  mmd:
    n: 14
    dimensions: 3
    benchmark: 0.2400470626477107
    execution_timeout: 30
```

Reset items can replace instance parameters:

```python
{"instance_params": {"n": 16, "dimensions": 2}}
```

## Verification

Run the standalone demo:

```bash
python demo_t1.py
```

Run local deterministic tests:

```bash
python -m pytest tests/test_geometric_envs.py -q
python tests/generate_geometric_visualizations.py
```

True differential tests load the official evaluator from an external SkyDiscover checkout; no SkyDiscover source is copied into this repository:

```bash
set SKYDISCOVER_REPO=D:\path\to\skydiscover
python -m pytest tests/test_geometric_differential.py -q
```

The differential suite executes the same candidate program through both implementations and compares validity and score. Pin the SkyDiscover commit before producing Task 2 experiment results.


