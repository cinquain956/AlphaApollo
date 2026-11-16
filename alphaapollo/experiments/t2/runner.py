from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .config import ProblemSettings, T2Config
from .llm_tracking import LLMCallRecorder


@dataclass(frozen=True)
class RunSpec:
    algorithm: str
    problem: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.algorithm}__{self.problem}__seed_{self.seed}"


@dataclass
class RunSummary:
    run_id: str
    algorithm: str
    problem: str
    seed: int
    iterations: int
    status: str
    output_dir: str
    program_count: int = 0
    initial_score: float = 0.0
    final_best_score: float = 0.0
    llm_calls: Dict[str, int] | None = None
    successful_solution_calls: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


def experiment_matrix(config: T2Config) -> List[RunSpec]:
    return [
        RunSpec(algorithm, problem, seed)
        for algorithm in config.algorithms
        for problem in config.problems
        for seed in config.seeds
    ]


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _jsonl_dump(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _score(program: Dict[str, Any]) -> float:
    metrics = program.get("metrics") or {}
    value = metrics.get("combined_score", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def collect_program_records(run_dir: Path) -> List[Dict[str, Any]]:
    """Collect solution programs from all checkpoints without including EvoX strategies."""
    candidates: List[Path] = []
    database_dir = run_dir / "database" / "programs"
    if database_dir.is_dir():
        candidates.extend(database_dir.glob("*.json"))
    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.is_dir():
        candidates.extend(checkpoint_root.glob("checkpoint_*/*/*.json"))
        candidates.extend(checkpoint_root.glob("checkpoint_*/programs/*.json"))

    by_id: Dict[str, Dict[str, Any]] = {}
    for path in candidates:
        if "search" in path.relative_to(run_dir).parts:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        solution = str(record.get("solution", ""))
        solution_hash = hashlib.sha256(solution.strip().encode("utf-8")).hexdigest()
        program_id = str(record.get("id") or solution_hash or path.stem)
        if "solution" not in record:
            continue
        record["id"] = program_id
        record["source_code"] = solution
        record["source_sha256"] = solution_hash
        record["score"] = _score(record)
        record["source_path"] = str(path.relative_to(run_dir))
        previous = by_id.get(program_id)
        if previous is None or len(record) > len(previous):
            by_id[program_id] = record

    records = list(by_id.values())
    records.sort(
        key=lambda item: (
            int(item.get("iteration_found", 0)),
            float(item.get("timestamp", 0.0)),
            str(item.get("id", "")),
        )
    )
    return records


def _call_counts(calls: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "total": 0,
        "solution": 0,
        "strategy": 0,
        "probe": 0,
        "failed": 0,
        "invalid_xml": 0,
    }
    for call in calls:
        category = str(call.get("category", "solution"))
        status = str(call.get("status", ""))
        counts["total"] += 1
        counts[category] = counts.get(category, 0) + 1
        if status != "ok":
            counts["failed"] += 1
        if status == "invalid_xml":
            counts["invalid_xml"] += 1
    return counts


def build_trajectory_records(
    database_records: List[Dict[str, Any]],
    calls: List[Dict[str, Any]],
    initial_source: str = "",
) -> List[Dict[str, Any]]:
    """Build one initial record plus one record per successful solution call.

    SkyDiscover may copy high-scoring programs between islands as migrants and
    reset ``iteration_found`` to zero. The LLM recorder is the authoritative
    iteration clock: its successful solution calls correspond one-for-one with
    completed evolution rounds. Database records supply evaluator metrics only.
    """
    successful_calls = [
        call
        for call in sorted(calls, key=lambda item: int(item.get("index", 0)))
        if call.get("category") == "solution"
        and call.get("status") == "ok"
        and str(call.get("normalized_source", "")).strip()
    ]
    if not successful_calls:
        records = [dict(record) for record in database_records]
        for record in records:
            record.setdefault("raw_action", "")
            record.setdefault(
                "action_kind",
                "initial_program" if record.get("parent_id") is None else "unknown",
            )
        return records

    by_hash: Dict[str, List[Dict[str, Any]]] = {}
    for record in database_records:
        by_hash.setdefault(str(record.get("source_sha256", "")), []).append(record)

    def best_match(source_hash: str, target_time: float) -> Dict[str, Any] | None:
        matches = by_hash.get(source_hash, [])
        if not matches:
            return None
        return min(
            matches,
            key=lambda item: (
                bool((item.get("metadata") or {}).get("migrant", False)),
                abs(float(item.get("timestamp", 0.0)) - target_time),
                str(item.get("id", "")),
            ),
        )

    records: List[Dict[str, Any]] = []
    canonical_initial = initial_source.strip()
    if canonical_initial:
        initial_hash = hashlib.sha256(canonical_initial.encode("utf-8")).hexdigest()
        initial_match = best_match(initial_hash, 0.0)
        initial = dict(initial_match or {})
        initial["id"] = str(initial.get("id") or f"initial__{initial_hash[:12]}")
        initial["source_code"] = initial_source
        initial["source_sha256"] = initial_hash
        initial["score"] = _score(initial)
    else:
        candidates = [
            item for item in database_records if item.get("parent_id") is None
        ]
        initial = dict(
            min(
                candidates or database_records,
                key=lambda item: (
                    float(item.get("timestamp", 0.0)),
                    str(item.get("id", "")),
                ),
            )
        ) if (candidates or database_records) else {
            "id": "initial",
            "source_code": "",
            "source_sha256": hashlib.sha256(b"").hexdigest(),
            "score": 0.0,
        }
    initial["framework_iteration_found"] = int(initial.get("iteration_found", 0))
    initial["iteration_found"] = 0
    initial["raw_action"] = ""
    initial["action_kind"] = "initial_program"
    records.append(initial)

    for iteration, call in enumerate(successful_calls, start=1):
        source = str(call["normalized_source"]).strip()
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        target_time = float(call.get("started_at", 0.0)) + float(
            call.get("duration_seconds", 0.0)
        )
        match = best_match(source_hash, target_time)
        record = dict(match or {})
        record["framework_program_id"] = str(record.get("id", ""))
        record["framework_iteration_found"] = int(
            record.get("iteration_found", iteration)
        )
        record["id"] = (
            f"iteration_{iteration:03d}__call_{int(call.get('index', 0)):04d}"
            f"__{source_hash[:12]}"
        )
        record["iteration_found"] = iteration
        record["source_code"] = source
        record["solution"] = source
        record["source_sha256"] = source_hash
        record["score"] = _score(record)
        record["llm_call_index"] = int(call.get("index", 0))
        record["raw_action"] = str(call.get("raw_response", ""))
        record["action_kind"] = str(call.get("action_kind", ""))
        records.append(record)
    return records


def build_best_so_far(
    records: List[Dict[str, Any]], iterations: int
) -> List[Dict[str, Any]]:
    scores_by_iteration: Dict[int, List[float]] = {}
    for record in records:
        iteration = int(record.get("iteration_found", 0))
        if iteration < 0:
            continue
        scores_by_iteration.setdefault(iteration, []).append(float(record["score"]))

    rows: List[Dict[str, Any]] = []
    best = 0.0
    for iteration in range(iterations + 1):
        current_scores = scores_by_iteration.get(iteration, [])
        if current_scores:
            best = max(best, max(current_scores))
        rows.append(
            {
                "iteration": iteration,
                "best_so_far": best,
                "programs_generated": len(current_scores),
            }
        )
    return rows


def _write_run_artifacts(
    run_dir: Path,
    iterations: int,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    _jsonl_dump(run_dir / "programs.jsonl", records)
    curve = build_best_so_far(records, iterations)
    with (run_dir / "best_so_far.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "best_so_far", "programs_generated"],
        )
        writer.writeheader()
        writer.writerows(curve)
    return {"program_count": len(records), "curve": curve}


def export_run_artifacts(
    run_dir: Path,
    iterations: int,
    recorder: LLMCallRecorder,
    initial_source: str = "",
) -> Dict[str, Any]:
    calls = recorder.records()
    database_records = collect_program_records(run_dir)
    records = build_trajectory_records(database_records, calls, initial_source)
    artifacts = _write_run_artifacts(run_dir, iterations, records)
    _jsonl_dump(run_dir / "llm_calls.jsonl", calls)
    artifacts["llm_calls"] = recorder.counts()
    return artifacts


def rebuild_run_artifacts(
    run_dir: Path,
    iterations: int,
    initial_program: Path,
) -> Dict[str, Any]:
    """Rebuild completed-run exports using persisted calls and database metrics."""
    calls_path = run_dir / "llm_calls.jsonl"
    calls: List[Dict[str, Any]] = []
    if calls_path.is_file():
        with calls_path.open("r", encoding="utf-8") as handle:
            calls = [json.loads(line) for line in handle if line.strip()]
    database_records = collect_program_records(run_dir)
    initial_source = initial_program.read_text(encoding="utf-8")
    records = build_trajectory_records(database_records, calls, initial_source)
    artifacts = _write_run_artifacts(run_dir, iterations, records)
    artifacts["llm_calls"] = _call_counts(calls)
    return artifacts


def _build_skydiscover_config(
    config: T2Config,
    spec: RunSpec,
    problem: ProblemSettings,
    run_dir: Path,
    recorder: LLMCallRecorder,
):
    from skydiscover.optimize.config import (
        Config,
        ContextBuilderConfig,
        EvaluatorConfig,
        EvoxDatabaseConfig,
        LLMConfig,
        LLMModelConfig,
        OpenEvolveNativeDatabaseConfig,
        SearchConfig,
    )

    model_cfg = LLMModelConfig(
        name=config.model.name,
        api_base=config.model.resolved_api_base(),
        api_key=config.model.resolved_api_key(),
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        timeout=config.model.timeout,
        retries=config.model.retries,
        init_client=recorder.client_factory(),
    )
    llm_cfg = LLMConfig(
        api_base=config.model.resolved_api_base(),
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        timeout=config.model.timeout,
        retries=config.model.retries,
        models=[model_cfg],
    )

    database_path = str(run_dir / "database")
    if spec.algorithm == "openevolve_native":
        database = OpenEvolveNativeDatabaseConfig(
            db_path=database_path,
            random_seed=spec.seed,
        )
    elif spec.algorithm == "evox":
        database = EvoxDatabaseConfig(
            db_path=database_path,
            random_seed=spec.seed,
            auto_generate_variation_operators=True,
        )
    else:
        raise ValueError(f"unsupported algorithm: {spec.algorithm}")

    return Config(
        max_iterations=config.iterations,
        checkpoint_interval=config.checkpoint_interval,
        log_level="INFO",
        language="python",
        file_suffix=".py",
        llm=llm_cfg,
        context_builder=ContextBuilderConfig(
            # The outer controller always evolves downstream solution programs.
            # EvoX creates its own inner controller with the dedicated `evox`
            # context builder for search-algorithm evolution.
            template="default",
            system_message=problem.system_prompt,
        ),
        search=SearchConfig(
            type=spec.algorithm,
            database=database,
            num_context_programs=4,
            output_dir=str(run_dir / "search"),
            switch_interval=(
                config.evox_switch_interval if spec.algorithm == "evox" else None
            ),
            share_llm=True,
        ),
        evaluator=EvaluatorConfig(
            evaluation_file=str(problem.evaluator),
            timeout=max(60, int(problem.execution_timeout) + 10),
            max_retries=0,
            cascade_evaluation=False,
            inject_evaluator_context=False,
        ),
        # T2 prompts require complete executable programs; keep the framework
        # in full-rewrite mode instead of SEARCH/REPLACE patches.
        diff_based_generation=False,
        max_parallel_iterations=1,
        system_prompt_override=problem.system_prompt,
    )


def _run_spec_worker(config: T2Config, spec: RunSpec, overwrite: bool) -> RunSummary:
    """Run one isolated matrix entry in its own process."""
    return T2Experiment(config).run_one(spec, overwrite=overwrite)


class T2Experiment:
    def __init__(
        self,
        config: T2Config,
        discovery_runner: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config
        self.config.validate()
        self.discovery_runner = discovery_runner

    def _runner(self) -> Callable[..., Any]:
        if self.discovery_runner is not None:
            return self.discovery_runner
        from skydiscover import run_discovery

        return run_discovery

    def run_dir(self, spec: RunSpec) -> Path:
        return (
            self.config.output_root
            / "runs"
            / spec.algorithm
            / spec.problem
            / f"seed_{spec.seed}"
        )

    def run_one(self, spec: RunSpec, overwrite: bool = False) -> RunSummary:
        problem = self.config.problem_settings[spec.problem]
        run_dir = self.run_dir(spec)
        if run_dir.exists() and overwrite:
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = run_dir / "run_summary.json"
        if existing.is_file() and not overwrite:
            data = json.loads(existing.read_text(encoding="utf-8"))
            return RunSummary(**data)

        started = time.time()
        recorder = LLMCallRecorder()
        metadata = {
            "run_id": spec.run_id,
            "algorithm": spec.algorithm,
            "problem": spec.problem,
            "seed": spec.seed,
            "iterations": self.config.iterations,
            "model": self.config.model.name,
            "api_base_env": self.config.model.api_base_env,
            "api_key_env": self.config.model.api_key_env,
            "instance": problem.instance,
            "evaluator": str(problem.evaluator),
            "initial_program": str(problem.initial_program),
            "skydiscover_version": _skydiscover_version(),
        }
        _json_dump(run_dir / "run_config.json", metadata)

        try:
            sky_config = _build_skydiscover_config(
                self.config, spec, problem, run_dir, recorder
            )
            result = self._runner()(
                evaluator=str(problem.evaluator),
                initial_program=str(problem.initial_program),
                iterations=self.config.iterations,
                search=spec.algorithm,
                config=sky_config,
                output_dir=str(run_dir),
                cleanup=False,
            )
            artifacts = export_run_artifacts(
                run_dir,
                self.config.iterations,
                recorder,
                problem.initial_program.read_text(encoding="utf-8"),
            )
            curve = artifacts["curve"]
            call_counts = dict(artifacts["llm_calls"])
            solution_records = [
                record for record in recorder.records()
                if record.get("category") == "solution"
            ]
            successful_solution_calls = sum(
                record.get("status") == "ok" for record in solution_records
            )
            incomplete_generation = (
                self.discovery_runner is None
                and successful_solution_calls < self.config.iterations
            )
            summary = RunSummary(
                run_id=spec.run_id,
                algorithm=spec.algorithm,
                problem=spec.problem,
                seed=spec.seed,
                iterations=self.config.iterations,
                status="failed" if incomplete_generation else "completed",
                output_dir=str(run_dir),
                program_count=int(artifacts["program_count"]),
                initial_score=float(curve[0]["best_so_far"]),
                final_best_score=float(curve[-1]["best_so_far"]),
                llm_calls=call_counts,
                successful_solution_calls=successful_solution_calls,
                elapsed_seconds=time.time() - started,
                error=(
                    f"Only {successful_solution_calls}/{self.config.iterations} evolution rounds produced a valid XML program."
                    if incomplete_generation
                    else ""
                ),
            )
            if not summary.program_count and getattr(result, "best_solution", ""):
                summary.error = "SkyDiscover returned a best program but no checkpoint programs were found"
        except Exception as exc:
            _jsonl_dump(run_dir / "llm_calls.jsonl", recorder.records())
            summary = RunSummary(
                run_id=spec.run_id,
                algorithm=spec.algorithm,
                problem=spec.problem,
                seed=spec.seed,
                iterations=self.config.iterations,
                status="failed",
                output_dir=str(run_dir),
                llm_calls=recorder.counts(),
                elapsed_seconds=time.time() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        _json_dump(run_dir / "run_summary.json", asdict(summary))
        return summary

    def run_all(self, overwrite: bool = False) -> List[RunSummary]:
        self.config.require_api_credentials()
        specs = experiment_matrix(self.config)
        # Each matrix entry owns a separate output/database directory. Separate
        # processes keep evaluator signal handlers on a main thread and isolate
        # EvoX state while allowing the taskbook-recommended API concurrency.
        if self.discovery_runner is None and self.config.max_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(self.config.max_workers, len(specs))
            ) as executor:
                futures = [
                    executor.submit(_run_spec_worker, self.config, spec, overwrite)
                    for spec in specs
                ]
                summaries = [future.result() for future in futures]
        else:
            summaries = [self.run_one(spec, overwrite=overwrite) for spec in specs]
        summaries.sort(key=lambda item: (item.algorithm, item.problem, item.seed))
        _json_dump(
            self.config.output_root / "manifest.json",
            {
                "expected_runs": len(specs),
                "completed_runs": sum(item.status == "completed" for item in summaries),
                "failed_runs": sum(item.status == "failed" for item in summaries),
                "runs": [asdict(item) for item in summaries],
            },
        )
        return summaries

    def dry_run(self) -> Dict[str, Any]:
        from .evaluation import evaluate_program_path

        baselines = {}
        for name in self.config.problems:
            problem = self.config.problem_settings[name]
            result = evaluate_program_path(
                str(problem.initial_program),
                name,
                problem.instance,
                problem.execution_timeout,
            )
            baselines[name] = {
                "score": float(result.metrics["combined_score"]),
                "valid": bool(result.metrics["valid"]),
                "feedback": str(result.artifacts.get("feedback", "")),
            }
        return {
            "matrix_size": self.config.matrix_size(),
            "iterations": self.config.iterations,
            "algorithms": list(self.config.algorithms),
            "problems": list(self.config.problems),
            "seeds": list(self.config.seeds),
            "model": self.config.model.name,
            "api_base_configured": bool(self.config.model.resolved_api_base()),
            "api_key_configured": bool(self.config.model.resolved_api_key()),
            "baselines": baselines,
        }


def _skydiscover_version() -> str:
    try:
        import skydiscover

        return str(getattr(skydiscover, "__version__", "unknown"))
    except ImportError:
        return "not-installed"
