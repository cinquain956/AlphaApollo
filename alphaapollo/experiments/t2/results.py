from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, List, Tuple

from .config import T2Config


def _load_curve(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "iteration": int(row["iteration"]),
                "best_so_far": float(row["best_so_far"]),
            }
            for row in csv.DictReader(handle)
        ]


def collect_completed_runs(config: T2Config) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = config.output_root / "runs"
    if not root.is_dir():
        return rows
    for summary_path in root.glob("*/*/seed_*/run_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            continue
        curve_path = summary_path.parent / "best_so_far.csv"
        if not curve_path.is_file():
            continue
        summary["curve"] = _load_curve(curve_path)
        rows.append(summary)
    rows.sort(key=lambda item: (item["algorithm"], item["problem"], item["seed"]))
    return rows


def _write_raw_data(config: T2Config, runs: List[Dict[str, Any]]) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    with (config.output_root / "raw_best_so_far.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["algorithm", "problem", "seed", "iteration", "best_so_far"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for point in run["curve"]:
                writer.writerow(
                    {
                        "algorithm": run["algorithm"],
                        "problem": run["problem"],
                        "seed": run["seed"],
                        **point,
                    }
                )

    with (config.output_root / "llm_call_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "algorithm",
            "problem",
            "seed",
            "total",
            "solution",
            "strategy",
            "probe",
            "failed",
            "invalid_xml",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            counts = run.get("llm_calls") or {}
            writer.writerow(
                {
                    "algorithm": run["algorithm"],
                    "problem": run["problem"],
                    "seed": run["seed"],
                    **{key: int(counts.get(key, 0)) for key in fields[3:]},
                }
            )


def _aggregate(
    config: T2Config, runs: List[Dict[str, Any]]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["algorithm"], run["problem"])].append(run)

    aggregates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, group in grouped.items():
        values_by_iteration = [[] for _ in range(config.iterations + 1)]
        for run in group:
            by_iteration = {
                int(point["iteration"]): float(point["best_so_far"])
                for point in run["curve"]
            }
            for iteration in range(config.iterations + 1):
                if iteration in by_iteration:
                    values_by_iteration[iteration].append(by_iteration[iteration])
        means = [fmean(values) if values else 0.0 for values in values_by_iteration]
        stds = [pstdev(values) if len(values) > 1 else 0.0 for values in values_by_iteration]
        aggregates[key] = {
            "seed_count": len(group),
            "seeds": [int(run["seed"]) for run in group],
            "iterations": list(range(config.iterations + 1)),
            "mean": means,
            "std": stds,
        }
    return aggregates


def _plot(config: T2Config, aggregates: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = config.output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for algorithm in config.algorithms:
        for problem in config.problems:
            data = aggregates.get((algorithm, problem))
            if not data:
                continue
            x = data["iterations"]
            mean = data["mean"]
            std = data["std"]
            lower = [m - s for m, s in zip(mean, std)]
            upper = [m + s for m, s in zip(mean, std)]
            fig, axis = plt.subplots(figsize=(7.2, 4.4))
            axis.plot(x, mean, linewidth=2, label=f"mean ({data['seed_count']} seeds)")
            axis.fill_between(x, lower, upper, alpha=0.22, label="+/- 1 std")
            axis.set_xlim(0, config.iterations)
            axis.set_xlabel("Cumulative evolution iterations")
            axis.set_ylabel("Best-so-far score")
            axis.set_title(f"{algorithm} / {problem}")
            axis.grid(alpha=0.25)
            axis.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{algorithm}__{problem}.png", dpi=180)
            plt.close(fig)


def _report(
    config: T2Config,
    runs: List[Dict[str, Any]],
    aggregates: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    lines = [
        "# Task 2 experiment report",
        "",
        f"Configured iterations per run: {config.iterations}.",
        f"Completed runs: {len(runs)} / {config.matrix_size()}.",
        "",
        "## Evolution results",
        "",
        "| Algorithm | Problem | Seeds | Initial mean | Final mean | Improvement | Evolving observed |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for algorithm in config.algorithms:
        for problem in config.problems:
            data = aggregates.get((algorithm, problem))
            if not data:
                lines.append(f"| {algorithm} | {problem} | 0 | ? | ? | ? | not run |")
                continue
            initial = float(data["mean"][0])
            final = float(data["mean"][-1])
            improvement = final - initial
            observed = "yes" if improvement > 1e-12 else "no"
            lines.append(
                f"| {algorithm} | {problem} | {data['seed_count']} | "
                f"{initial:.8g} | {final:.8g} | {improvement:.8g} | {observed} |"
            )

    lines.extend(
        [
            "",
            "## LLM calls",
            "",
            "`solution` calls can produce candidate programs. `strategy` calls are EvoX "
            "search-policy/variation-operator calls and do not directly produce a geometry candidate. "
            "`probe` calls are endpoint availability checks.",
            "",
            "| Algorithm | Solution calls | Strategy calls | Probe calls | Failed calls | Invalid XML |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for algorithm in config.algorithms:
        selected = [run for run in runs if run["algorithm"] == algorithm]
        totals = {key: 0 for key in ("solution", "strategy", "probe", "failed", "invalid_xml")}
        for run in selected:
            counts = run.get("llm_calls") or {}
            for key in totals:
                totals[key] += int(counts.get(key, 0))
        lines.append(
            f"| {algorithm} | {totals['solution']} | {totals['strategy']} | "
            f"{totals['probe']} | {totals['failed']} | {totals['invalid_xml']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The report marks evolving only when the three-seed mean final best-so-far exceeds "
            "the initial mean. Compare methods using both score and call counts: EvoX may spend "
            "extra strategy calls that do not generate candidates, so equal iteration counts do "
            "not imply equal API cost.",
            "",
            "Raw per-seed curves are stored in `raw_best_so_far.csv`; per-run programs remain "
            "in each run's `programs.jsonl` for Task 3 replay.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_results(config: T2Config) -> Dict[str, Any]:
    runs = collect_completed_runs(config)
    _write_raw_data(config, runs)
    aggregates = _aggregate(config, runs)
    _plot(config, aggregates)
    report = _report(config, runs, aggregates)
    (config.output_root / "report.md").write_text(report, encoding="utf-8")
    serializable = {
        f"{algorithm}__{problem}": value
        for (algorithm, problem), value in aggregates.items()
    }
    (config.output_root / "aggregate.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "completed_runs": len(runs),
        "expected_runs": config.matrix_size(),
        "plots": len(aggregates),
        "output_root": str(config.output_root),
    }
