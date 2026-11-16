import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class CheckResult:
    valid: bool
    score: float
    objective: float
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            'valid': self.valid,
            'score': self.score,
            'objective': self.objective,
            'violations': self.violations,
            'warnings': self.warnings,
            'metrics': self.metrics,
        }


class GeometryChecker(ABC):
    @abstractmethod
    def check(self, action: Any) -> Dict[str, Any]:
        '''Validate and score one candidate solution.'''


def parse_payload(action: Any) -> Any:
    '''Parse a JSON candidate from a dict/list or model text.'''
    if isinstance(action, (dict, list)):
        return action
    if not isinstance(action, str):
        raise ValueError('candidate must be JSON text, an object, or an array')

    text = action.strip()
    answer = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    if answer:
        text = answer.group(1).strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    positions = [pos for pos in (text.find('{'), text.find('[')) if pos >= 0]
    if not positions:
        raise ValueError('candidate does not contain a JSON object or array')
    start = min(positions)
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid JSON candidate: {exc}') from exc
    return payload


def finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be numeric') from exc
    if not math.isfinite(number):
        raise ValueError(f'{label} must be finite')
    return number


def violation(kind: str, message: str, **details: Any) -> Dict[str, Any]:
    return {'type': kind, 'message': message, **details}
