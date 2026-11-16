# Task 3: neighbour-instance transfer

T3 selects one completed formal T2 trajectory for each algorithm/problem pair and replays every source-executable retained program on the source instance plus four neighbours. It does not call the LLM, evolve on a neighbour, modify source text, assemble patches, or inject a target answer.

| Algorithm | Problem | Seed | Retained | Replayed | Skipped |
|---|---|---:|---:|---:|---:|
| OpenEvolve | Circle Packing | 33 | 33 | 29 | 4 |
| OpenEvolve | MMD | 22 | 33 | 33 | 0 |
| EvoX | Circle Packing | 22 | 33 | 31 | 2 |
| EvoX | MMD | 33 | 33 | 32 | 1 |

All four inputs have `xml_compatible_t2` provenance. For every replayed program, `source_sha256_t2`, `source_sha256_before`, and `source_sha256_after` are identical. Each replay record contains exactly five target results, and retained count equals replayed plus skipped count.

## Transfer results

The table reports normalized best-so-far score from the initial program to the final T2 iteration.

| Trajectory | Source | Scale neighbour 1 | Scale neighbour 2 | Structural neighbour 1 | Structural neighbour 2 |
|---|---:|---:|---:|---:|---:|
| OpenEvolve Circle | 0.822264 -> 0.936980 | square n=24: 0.948500 -> 0.948501 | square n=28: 0.852288 -> 0.933883 | triangle n=24: 0 -> 0 | triangle n=28: 0 -> 0 |
| EvoX Circle | 0.822264 -> 0.959914 | square n=24: 0.948500 -> 0.957333 | square n=28: 0.852288 -> 0.973630 | triangle n=24: 0 -> 0 | triangle n=28: 0 -> 0 |
| OpenEvolve MMD | 0.847046 -> 0.965981 | 3D n=12: 0.853692 -> 1.000000 | 3D n=16: 0.827786 -> 0.966561 | 2D n=12: 0.461325 -> 0.752986 | 2D n=16: 0.234847 -> 0.786155 |
| EvoX MMD | 0.847046 -> 0.938268 | 3D n=12: 0.853692 -> 1.000000 | 3D n=16: 0.827786 -> 0.963959 | 2D n=12: 0.461325 -> 0.752986 | 2D n=16: 0.234847 -> 0.700399 |

Circle programs transfer well across nearby values of `n` in the same square container, but none of the selected trajectories produces a valid equilateral-triangle packing. This is evidence of container-specific geometry assumptions. MMD programs transfer across both point count and dimension, with positive gains on every neighbour in both selected trajectories.

## Transport and checking

Scale transfer changes only `instance["n"]`. Structural transfer changes `container_type` and `side_length` for Circle Packing or `dimensions` for MMD. The original `solve(instance)` source is executed unchanged in an isolated process and checked by the deterministic T1 checker for the target instance. Wrong counts, wrong dimensions, invalid boundaries, overlaps, and execution failures receive zero for that target.

The equilateral-triangle checker uses signed perpendicular distance to all three sides. The 2D MMD targets reuse the dimension-parameterized exhaustive pairwise-distance checker. Approximate known-best constants are explicitly marked in `outputs/t3/manifest.json`; they affect vertical normalization but not validity or within-target ordering.

## Artifacts

- `outputs/t3/manifest.json`: selected trajectories, target definitions, and provenance.
- `outputs/t3/report.md`: complete per-target early/middle/late result tables.
- `outputs/t3/trajectories/*/replay.jsonl`: all five target evaluations and three source hashes per replayed program.
- `outputs/t3/trajectories/*/skipped_source_programs.jsonl`: programs that could not execute on the source instance.
- `outputs/t3/trajectories/*/curves.csv`: per-iteration and cumulative transfer curves.
- `outputs/t3/trajectories/*/transfer.png`: four transfer plots.
