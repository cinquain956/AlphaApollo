from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict

if importlib.util.find_spec("ray") is None:
    os.environ.setdefault("ALPHAAPOLLO_LIGHTWEIGHT_IMPORT", "1")

from alphaapollo.core.environments.geometric.executor import PythonProgramExecutor
from alphaapollo.core.environments.geometric.spec import get_problem_spec


def _feedback(result: Dict[str, Any], execution: Dict[str, Any]) -> str:
    if not execution.get("ok"):
        return (
            "Program execution failed: "
            + str(execution.get("error_type") or execution.get("status"))
            + ": "
            + str(execution.get("error_message") or "unknown error")
        )
    if result.get("valid"):
        warning_text = "; ".join(
            str(item.get("message", item)) for item in result.get("warnings", [])
        )
        message = (
            f"Valid candidate. Objective={float(result['objective']):.12g}; "
            f"score={float(result['score']):.12g}."
        )
        return message + (f" Warnings: {warning_text}" if warning_text else "")
    details = []
    for item in result.get("violations", []):
        details.append(
            f"{item.get('type', 'violation')}: "
            f"{item.get('message', json.dumps(item, ensure_ascii=False))}"
        )
    return "Invalid candidate. " + " | ".join(details)


def evaluate_program_path(
    program_path: str,
    problem: str,
    instance: Dict[str, Any],
    execution_timeout: float = 30.0,
):
    """Evaluate one SkyDiscover program with the deterministic T1 checker."""
    from skydiscover.optimize.evaluation.evaluation_result import EvaluationResult

    source = Path(program_path).read_text(encoding="utf-8")
    spec = get_problem_spec(problem)
    checker = spec.build_checker(instance)
    execution_result = PythonProgramExecutor(execution_timeout).execute(
        source,
        instance,
        spec.entrypoints(checker),
    )
    execution = execution_result.as_dict()
    if execution_result.ok:
        candidate_output = spec.normalize_output(execution_result.payload)
        result = checker.check(candidate_output)
    else:
        candidate_output = None
        result = {
            "valid": False,
            "score": 0.0,
            "objective": 0.0,
            "violations": [
                {
                    "type": "execution_error",
                    "message": execution_result.error_message
                    or execution_result.status,
                }
            ],
            "warnings": [],
            "metrics": {},
        }

    violation_counts = Counter(
        str(item.get("type", "unknown")) for item in result.get("violations", [])
    )
    metrics: Dict[str, float] = {
        "combined_score": float(result["score"]) if result.get("valid") else 0.0,
        "objective": float(result.get("objective", 0.0)),
        "valid": float(bool(result.get("valid"))),
        "runtime_seconds": float(execution.get("runtime_seconds", 0.0)),
        "violation_count": float(len(result.get("violations", []))),
        "warning_count": float(len(result.get("warnings", []))),
    }
    for key, value in (result.get("metrics") or {}).items():
        if isinstance(value, bool):
            metrics[str(key)] = float(value)
        elif isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    for kind, count in violation_counts.items():
        metrics[f"violations_{kind}"] = float(count)

    artifacts = {
        "feedback": _feedback(result, execution),
        "checker_result": json.dumps(result, ensure_ascii=False, allow_nan=False),
        "execution": json.dumps(execution, ensure_ascii=False, allow_nan=False),
        "candidate_output": json.dumps(
            candidate_output, ensure_ascii=False, allow_nan=False
        )
        if candidate_output is not None
        else "null",
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
