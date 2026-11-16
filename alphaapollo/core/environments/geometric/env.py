from typing import Any, Dict, List, Optional, TypedDict


class BaseTextEnvStepOutput(TypedDict):
    observations: List[Dict[str, Any]]
    reward: float
    done: bool
    metadata: Dict[str, Any]
    postprocessed_action: Optional[str]

from .base import CheckResult, violation
from .executor import PythonProgramExecutor
from .projection import parse_program_action
from .spec import get_problem_spec


def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, 'items'):
        return dict(value.items())
    return {}


class GeometricTextEnv:
    '''Problem-agnostic program execution environment for geometry checkers.'''

    def __init__(
        self,
        problem_type: str,
        env_config: Any = None,
        parameter_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.problem_type = str(problem_type).lower()
        self.spec = get_problem_spec(self.problem_type)
        section = getattr(env_config, self.problem_type, None) if env_config is not None else None
        raw_params = _to_dict(section)
        raw_params.pop('memory_type', None)
        self.execution_timeout = float(raw_params.pop('execution_timeout', 30.0))
        if parameter_overrides:
            overrides = dict(parameter_overrides)
            self.execution_timeout = float(
                overrides.pop('execution_timeout', self.execution_timeout)
            )
            raw_params.update(overrides)
        self.default_params = raw_params
        self.default_max_steps = int(getattr(env_config, 'max_steps', 1)) if env_config is not None else 1
        self.executor = PythonProgramExecutor(self.execution_timeout)
        self.checker = self.spec.build_checker(self.default_params)
        self.instance_params = dict(self.default_params)
        self.max_steps = self.default_max_steps
        self.turns = 0
        self.done = False
        self.data_source = self.problem_type

    def reset(self, extras: Optional[Dict[str, Any]] = None) -> None:
        extras = extras or {}
        params = dict(self.default_params)
        params.update(_to_dict(extras.get('instance_params')))
        self.checker = self.spec.build_checker(params)
        self.instance_params = params
        timeout = float(extras.get('execution_timeout', self.execution_timeout))
        self.executor = PythonProgramExecutor(timeout)
        self.max_steps = int(extras.get('max_steps', self.default_max_steps))
        if self.max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.turns = 0
        self.done = False
        self.data_source = extras.get('data_source', self.problem_type)

    def problem_statement(self) -> str:
        return self.spec.problem_statement(self.checker, self.instance_params)

    @staticmethod
    def _feedback(result: Dict[str, Any]) -> str:
        if result['valid']:
            objective = result['objective']
            score = result['score']
            text = f'Valid candidate. Objective: {objective:.12g}. Score: {score:.12g}.'
            if result.get('warnings'):
                messages = '; '.join(item['message'] for item in result['warnings'])
                text += f' Warnings: {messages}.'
            return text
        messages = '\n'.join(
            '- {kind}: {message}'.format(kind=item['type'], message=item['message'])
            for item in result['violations']
        )
        return f'Invalid candidate. Fix these violations:\n{messages}'

    def _execution_failure(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        message = execution.get('error_message') or execution.get('status') or 'execution failed'
        return CheckResult(
            valid=False,
            score=0.0,
            objective=0.0,
            violations=[
                violation(
                    'execution_error',
                    message,
                    status=execution.get('status'),
                    error_type=execution.get('error_type'),
                )
            ],
            metrics={},
        ).as_dict()

    def step(self, action: Any, text_actions: Any = None) -> BaseTextEnvStepOutput:
        self.turns += 1
        source_code = action if isinstance(action, str) else ''
        raw_action = text_actions if isinstance(text_actions, str) else source_code
        try:
            _, action_kind = parse_program_action(raw_action)
        except (SyntaxError, TypeError, ValueError):
            action_kind = 'invalid'
        empty_action = not source_code.strip()
        if empty_action:
            execution = {
                'ok': False,
                'status': 'empty_action',
                'error_type': 'EmptyAction',
                'error_message': 'candidate program is empty',
            }
            candidate_output = None
            result = self._execution_failure(execution)
        else:
            execution_result = self.executor.execute(
                source_code,
                self.instance_params,
                self.spec.entrypoints(self.checker),
            )
            execution = execution_result.as_dict()
            if execution_result.ok:
                candidate_output = self.spec.normalize_output(execution_result.payload)
                result = self.checker.check(candidate_output)
            else:
                candidate_output = None
                result = self._execution_failure(execution)

        self.done = (
            empty_action
            or action_kind == 'answer'
            or self.turns >= self.max_steps
        )
        feedback = self._feedback(result)
        trajectory_record = {
            'schema_version': 1,
            'problem': self.problem_type,
            'iteration': self.turns,
            'action_kind': action_kind,
            'raw_action': raw_action,
            'source_code': source_code,
            'instance_params': dict(self.instance_params),
            'execution': execution,
            'candidate_output': candidate_output,
            **result,
        }
        observation = {
            'role': 'user',
            'content': feedback,
            'text_actions': text_actions if text_actions is not None else action,
        }
        metadata = {
            'data_source': self.data_source,
            'problem': self.problem_type,
            'turn': self.turns,
            'action_kind': action_kind,
            'raw_action': raw_action,
            'done_reason': (
                'empty_action'
                if empty_action
                else ('answer' if action_kind == 'answer' else ('max_steps' if self.done else None))
            ),
            'source_code': source_code,
            'instance_params': dict(self.instance_params),
            'execution': execution,
            'candidate_output': candidate_output,
            'trajectory_record': trajectory_record,
            **result,
        }
        return BaseTextEnvStepOutput(
            observations=[observation],
            reward=float(result['score']) if result['valid'] else 0.0,
            done=self.done,
            metadata=metadata,
            postprocessed_action=source_code,
        )

    def close(self) -> None:
        return None

class CirclePackingEnv(GeometricTextEnv):
    def __init__(self, env_config: Any = None, **params: Any) -> None:
        super().__init__('circle_packing', env_config, params)


class MMDEnv(GeometricTextEnv):
    def __init__(self, env_config: Any = None, **params: Any) -> None:
        super().__init__('mmd', env_config, params)



