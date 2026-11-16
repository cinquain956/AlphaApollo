from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from alphaapollo.core.environments.geometric.executor import PythonProgramExecutor
from alphaapollo.core.environments.geometric.spec import get_problem_spec

from .config import T3Config, TargetSettings


@dataclass(frozen=True)
class TrajectoryChoice:
    algorithm: str
    problem: str
    seed: int
    run_dir: str
    retained_programs: int
    final_best_score: float
    selection_reason: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _jsonl_dump(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False) + '\n')


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda item: (int(item.get('iteration_found', 0)), float(item.get('timestamp', 0.0)), str(item.get('id', ''))))
    return records


def source_sha256(source: str) -> str:
    """Use the same canonical source hash as the T2 trajectory exporter."""
    return hashlib.sha256(source.strip().encode('utf-8')).hexdigest()


def _run_candidates(config: T3Config, algorithm: str, problem: str) -> List[Tuple[int, Path, Dict[str, Any]]]:
    root = config.t2_output_root / 'runs' / algorithm / problem
    candidates = []
    if not root.is_dir():
        return candidates
    for run_dir in root.glob('seed_*'):
        summary_path = run_dir / 'run_summary.json'
        programs_path = run_dir / 'programs.jsonl'
        if not summary_path.is_file() or not programs_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('status') != 'completed':
            continue
        seed = int(summary.get('seed', run_dir.name.replace('seed_', '')))
        candidates.append((seed, run_dir, summary))
    return candidates


def choose_trajectory(config: T3Config, algorithm: str, problem: str) -> TrajectoryChoice:
    key = f'{algorithm}__{problem}'
    requested = config.selected_seeds.get(key)
    candidates = _run_candidates(config, algorithm, problem)
    if requested is not None:
        candidates = [item for item in candidates if item[0] == requested]
        if not candidates:
            raise FileNotFoundError(f'configured T2 trajectory not found: {key} seed {requested}')
        reason = 'explicitly configured seed'
    else:
        reason = 'largest retained completed trajectory; final score breaks ties'
    if not candidates:
        raise FileNotFoundError(f'no completed T2 trajectory found for {key}')
    seed, run_dir, summary = max(
        candidates,
        key=lambda item: (
            int(item[2].get('program_count', 0)),
            float(item[2].get('final_best_score', 0.0)),
            -item[0],
        ),
    )
    return TrajectoryChoice(
        algorithm=algorithm,
        problem=problem,
        seed=seed,
        run_dir=str(run_dir),
        retained_programs=int(summary.get('program_count', 0)),
        final_best_score=float(summary.get('final_best_score', 0.0)),
        selection_reason=reason,
    )


def evaluate_source(source: str, target: TargetSettings, timeout: float) -> Dict[str, Any]:
    spec = get_problem_spec(target.problem)
    checker_params = dict(target.instance)
    if target.problem == 'circle_packing':
        checker_params['benchmark'] = target.known_best_objective
    else:
        checker_params['benchmark'] = target.known_best_objective ** 2
    checker = spec.build_checker(checker_params)
    execution = PythonProgramExecutor(timeout).execute(
        source,
        dict(target.instance),
        spec.entrypoints(checker),
    )
    if not execution.ok:
        return {
            'executable': False,
            'valid': False,
            'objective': 0.0,
            'normalized_score': 0.0,
            'execution': execution.as_dict(),
            'violations': [{'type': 'execution_error', 'message': execution.error_message or execution.status}],
        }
    output = spec.normalize_output(execution.payload)
    result = checker.check(output)
    objective = float(result.get('objective', 0.0)) if result.get('valid') else 0.0
    return {
        'executable': True,
        'valid': bool(result.get('valid')),
        'objective': objective,
        'normalized_score': objective / target.known_best_objective if result.get('valid') else 0.0,
        'execution': execution.as_dict(),
        'violations': result.get('violations', []),
        'warnings': result.get('warnings', []),
        'checker_metrics': result.get('metrics', {}),
    }


def build_curves(records: List[Dict[str, Any]], targets: List[TargetSettings], iterations: int) -> List[Dict[str, Any]]:
    by_target_iteration: Dict[Tuple[str, int], List[float]] = {}
    for record in records:
        if not record.get('source_executable'):
            continue
        iteration = int(record['iteration'])
        for target in targets:
            result = record['targets'][target.name]
            by_target_iteration.setdefault((target.name, iteration), []).append(float(result['normalized_score']))
    rows = []
    for target in targets:
        best = 0.0
        for iteration in range(iterations + 1):
            values = by_target_iteration.get((target.name, iteration), [])
            iteration_best = max(values) if values else None
            if iteration_best is not None:
                best = max(best, iteration_best)
            rows.append(
                {
                    'target': target.name,
                    'label': target.label,
                    'iteration': iteration,
                    'iteration_best': '' if iteration_best is None else iteration_best,
                    'best_so_far': best,
                    'programs_evaluated': len(values),
                }
            )
    return rows


def _write_curve(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['target', 'label', 'iteration', 'iteration_best', 'best_so_far', 'programs_evaluated'])
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, title: str, rows: List[Dict[str, Any]], iterations: int) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    labels = []
    for row in rows:
        if row['label'] not in labels:
            labels.append(row['label'])
    for label in labels:
        selected = [row for row in rows if row['label'] == label]
        axis.plot([row['iteration'] for row in selected], [row['best_so_far'] for row in selected], linewidth=2, label=label)
    axis.set_xlim(0, iterations)
    axis.set_xlabel('T2 evolution iteration')
    axis.set_ylabel('Transferred normalized objective')
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


class T3Experiment:
    def __init__(self, config: T3Config) -> None:
        self.config = config
        self.config.validate()

    def replay_one(self, choice: TrajectoryChoice, overwrite: bool = False) -> Dict[str, Any]:
        output_dir = self.config.output_root / 'trajectories' / f'{choice.algorithm}__{choice.problem}'
        existing_summary = output_dir / 'summary.json'
        if existing_summary.is_file() and not overwrite:
            return json.loads(existing_summary.read_text(encoding='utf-8'))
        if output_dir.exists() and overwrite:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        t2_records = _load_jsonl(Path(choice.run_dir) / 'programs.jsonl')
        targets = self.config.targets[choice.problem]
        source_target = next(target for target in targets if target.name.endswith('_source'))
        neighbour_targets = [target for target in targets if target.name != source_target.name]
        replay_records = []
        skipped = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(neighbour_targets))
        ) as executor:
            for program in t2_records:
                source = str(program.get('source_code') or program.get('solution') or '')
                source_hash = source_sha256(source)
                t2_source_hash = str(program.get('source_sha256', '')).strip()
                if t2_source_hash and source_hash != t2_source_hash:
                    raise ValueError(
                        f"T2 source hash mismatch for program {program.get('id', '')}: "
                        f'{t2_source_hash} != {source_hash}'
                    )
                source_result = evaluate_source(source, source_target, self.config.execution_timeout)
                if not source_result['executable']:
                    skipped.append(
                        {
                            'program_id': str(program.get('id', '')),
                            'iteration': int(program.get('iteration_found', 0)),
                            'source_sha256': source_hash,
                            'reason': source_result['execution'].get('error_message') or source_result['execution'].get('status'),
                        }
                    )
                    continue
                future_targets = {
                    executor.submit(
                        evaluate_source,
                        source,
                        target,
                        self.config.execution_timeout,
                    ): target
                    for target in neighbour_targets
                }
                target_results = {source_target.name: source_result}
                for future, target in future_targets.items():
                    target_results[target.name] = future.result()
                replay_records.append(
                    {
                        'program_id': str(program.get('id', '')),
                        'iteration': int(program.get('iteration_found', 0)),
                        'source_score_t2': float(program.get('score', 0.0)),
                        'source_sha256_t2': t2_source_hash or source_hash,
                        'source_sha256_before': source_hash,
                        'source_sha256_after': source_sha256(source),
                        'transport': 'instance_only_no_source_edit',
                        'source_executable': True,
                        'targets': target_results,
                    }
                )
        rows = build_curves(replay_records, targets, self.config.iterations)
        _jsonl_dump(output_dir / 'replay.jsonl', replay_records)
        _jsonl_dump(output_dir / 'skipped_source_programs.jsonl', skipped)
        _write_curve(output_dir / 'curves.csv', rows)
        _plot(
            output_dir / 'transfer.png',
            f'{choice.algorithm} / {choice.problem} / seed {choice.seed}',
            rows,
            self.config.iterations,
        )
        has_raw_xml = any(bool(str(program.get('raw_action', '')).strip()) for program in t2_records)
        summary = {
            'choice': asdict(choice),
            'input_has_raw_xml_actions': has_raw_xml,
            'input_provenance': ('xml_compatible_t2' if has_raw_xml else 'legacy_pre_xml_t2'),
            'source_programs_retained': len(t2_records),
            'source_programs_executable': len(replay_records),
            'source_programs_skipped': len(skipped),
            'targets': [asdict(target) for target in targets],
            'transport_policy': 'The original source text is unchanged. Only solve(instance) input and checker parameters change; no neighbour evolution, patching, algorithm rewrite, or answer injection occurs.',
            'output_dir': str(output_dir),
        }
        _json_dump(output_dir / 'summary.json', summary)
        return summary

    def run_all(self, overwrite: bool = False) -> Dict[str, Any]:
        choices = [
            choose_trajectory(self.config, algorithm, problem)
            for algorithm in self.config.algorithms
            for problem in self.config.problems
        ]
        summaries = [self.replay_one(choice, overwrite=overwrite) for choice in choices]
        manifest = {
            'trajectories': summaries,
            'plots': len(summaries),
            'transport_policy': 'instance_only_no_source_edit',
        }
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        _json_dump(self.config.output_root / 'manifest.json', manifest)
        self._write_report(manifest)
        return manifest

    def _write_report(self, manifest: Dict[str, Any]) -> None:
        lines = [
            '# Task 3 transfer report',
            '',
            'Each trajectory is one documented T2 seed. Every source-executable retained program is replayed on the source instance and four neighbours. Candidate source hashes before and after transport are identical.',
            '',
            '| Algorithm | Problem | Seed | Retained | Replayed | Skipped | T2 provenance |',
            '|---|---|---:|---:|---:|---:|---|',
        ]
        for item in manifest['trajectories']:
            choice = item['choice']
            lines.append(
                f"| {choice['algorithm']} | {choice['problem']} | {choice['seed']} | "
                f"{item['source_programs_retained']} | {item['source_programs_executable']} | "
                f"{item['source_programs_skipped']} | {item.get('input_provenance', 'unknown')} |"
            )

        lines.extend(['', '## Transfer curves', ''])
        for item in manifest['trajectories']:
            choice = item['choice']
            curve_path = Path(item['output_dir']) / 'curves.csv'
            with curve_path.open('r', encoding='utf-8', newline='') as handle:
                curve_rows = list(csv.DictReader(handle))
            grouped: Dict[str, List[Dict[str, str]]] = {}
            for row in curve_rows:
                grouped.setdefault(row['label'], []).append(row)
            lines.extend(
                [
                    f"### {choice['algorithm']} / {choice['problem']} / seed {choice['seed']}",
                    '',
                    '| Instance | Initial best | Final best | Gain | Best found at/before iteration | Early max | Middle max | Late max |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|',
                ]
            )
            for label, rows in grouped.items():
                values = [float(row['best_so_far']) for row in rows]
                initial, final = values[0], values[-1]
                best_iteration = next(
                    int(row['iteration']) for row in rows
                    if abs(float(row['best_so_far']) - final) <= 1e-15
                )
                stage_maxima = []
                for low, high in ((0, 10), (11, 21), (22, self.config.iterations)):
                    stage_values = [
                        float(row['iteration_best'])
                        for row in rows
                        if low <= int(row['iteration']) <= high and row['iteration_best'] != ''
                    ]
                    stage_maxima.append(max(stage_values) if stage_values else 0.0)
                lines.append(
                    f"| {label} | {initial:.6g} | {final:.6g} | {final - initial:.6g} | "
                    f"{best_iteration} | {stage_maxima[0]:.6g} | {stage_maxima[1]:.6g} | {stage_maxima[2]:.6g} |"
                )
            neighbour_gains = []
            for label, rows in grouped.items():
                if '(source)' in label:
                    continue
                neighbour_gains.append(float(rows[-1]['best_so_far']) - float(rows[0]['best_so_far']))
            lines.extend(
                [
                    '',
                    (
                        'Transferred evolving is observed on at least one neighbour.'
                        if any(gain > 1e-12 for gain in neighbour_gains)
                        else 'No neighbour improves beyond its initial transferred program in this trajectory.'
                    ),
                    '',
                ]
            )

        lines.extend(
            [
                '## Transport design',
                '',
                'Scale changes replace `instance["n"]`. Structural changes replace `container_type`/`side_length` for Circle Packing or `dimensions` for MMD. The replay runner does not edit Python source. A program that ignores the parameter, returns the wrong shape, or violates the new container receives zero; an execution failure is also recorded. This avoids embedding neighbour answers or silently repairing an algorithm.',
                '',
                'The equilateral-triangle checker uses signed perpendicular distance to all three sides. The 2D MMD checker reuses the dimension-parameterized MMD checker and its exhaustive pairwise-distance calculation. Unit tests cover the triangle incenter/tangency, boundary crossing, overlap, valid 2D ratios, count errors, and dimension errors.',
                '',
                '## Interpretation',
                '',
                'The cumulative curves answer whether useful improvements survive transport. The early/middle/late columns use the best newly generated program inside each stage, so they expose stage-specific regressions hidden by best-so-far. A later source improvement with zero neighbour gain indicates source specialization; simultaneous source and neighbour gains indicate transferable algorithmic structure.',
                '',
                'Approximate known-best constants are explicitly marked in each target record. They affect vertical normalization slightly but do not change validity or within-target trajectory ordering.',
            ]
        )
        (self.config.output_root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
