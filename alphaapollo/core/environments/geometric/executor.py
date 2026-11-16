import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class ExecutionResult:
    ok: bool
    status: str
    payload: Any = None
    entrypoint: Optional[str] = None
    runtime_seconds: float = 0.0
    returncode: Optional[int] = None
    captured_output: str = ''
    stderr: str = ''
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PythonProgramExecutor:
    """Execute one candidate program in a temporary subprocess."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive')
        self.worker_path = Path(__file__).with_name('executor_worker.py').resolve()

    def execute(
        self,
        source_code: str,
        instance: Dict[str, Any],
        entrypoints: Iterable[str],
    ) -> ExecutionResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='alphaapollo_geometry_') as temp_dir:
            temp_path = Path(temp_dir)
            candidate_path = temp_path / 'candidate.py'
            request_path = temp_path / 'request.json'
            candidate_path.write_text(source_code, encoding='utf-8')
            request_path.write_text(
                json.dumps(
                    {
                        'candidate_path': str(candidate_path),
                        'instance': instance,
                        'entrypoints': list(entrypoints),
                    },
                    allow_nan=False,
                ),
                encoding='utf-8',
            )
            try:
                completed = subprocess.run(
                    [sys.executable, str(self.worker_path), str(request_path)],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    ok=False,
                    status='timeout',
                    runtime_seconds=time.perf_counter() - started,
                    captured_output=exc.stdout or '',
                    stderr=exc.stderr or '',
                    error_type='TimeoutExpired',
                    error_message=f'candidate exceeded {self.timeout_seconds:g} seconds',
                )

        runtime = time.perf_counter() - started
        if completed.returncode != 0:
            return ExecutionResult(
                ok=False,
                status='process_error',
                runtime_seconds=runtime,
                returncode=completed.returncode,
                stderr=completed.stderr,
                error_type='ProcessError',
                error_message='candidate worker exited with a non-zero status',
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return ExecutionResult(
                ok=False,
                status='protocol_error',
                runtime_seconds=runtime,
                returncode=completed.returncode,
                captured_output=completed.stdout,
                stderr=completed.stderr,
                error_type='JSONDecodeError',
                error_message=f'candidate worker returned invalid protocol data: {exc}',
            )
        return ExecutionResult(
            ok=bool(response.get('ok')),
            status='ok' if response.get('ok') else 'execution_error',
            payload=response.get('payload'),
            entrypoint=response.get('entrypoint'),
            runtime_seconds=runtime,
            returncode=completed.returncode,
            captured_output=response.get('captured_output', ''),
            stderr=completed.stderr,
            error_type=response.get('error_type'),
            error_message=response.get('error_message'),
            traceback=response.get('traceback'),
        )
