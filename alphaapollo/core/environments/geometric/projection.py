import ast
import re
from typing import List, Tuple


_ENTRYPOINT_PREFIXES = ('solve', 'run_packing', 'min_max_dist_dim')
_ACTION_TAGS = ('python_code', 'answer')


def _strip_fence(candidate: str) -> str:
    fenced = re.fullmatch(
        r'```(?:python|py)?\s*(.*?)\s*```',
        candidate.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1).strip() if fenced else candidate.strip()


def parse_program_action(action: str, *, require_xml: bool = False) -> Tuple[str, str]:
    """Extract exactly one complete program from an AlphaApollo-style action."""
    if not isinstance(action, str):
        raise ValueError('candidate action must be text')
    text = action.strip()
    if not text:
        raise ValueError('candidate program is empty')

    opening_matches = {
        tag: list(re.finditer(fr'<{tag}>', text, re.IGNORECASE))
        for tag in _ACTION_TAGS
    }
    closing_matches = {
        tag: list(re.finditer(fr'</{tag}>', text, re.IGNORECASE))
        for tag in _ACTION_TAGS
    }
    total_openings = sum(len(matches) for matches in opening_matches.values())
    total_closings = sum(len(matches) for matches in closing_matches.values())

    if total_openings or total_closings:
        if total_openings != 1 or total_closings != 1:
            raise ValueError('submit exactly one complete python_code or answer action')

        action_kind = next(tag for tag, matches in opening_matches.items() if matches)
        closing_kind = next(tag for tag, matches in closing_matches.items() if matches)
        if action_kind != closing_kind:
            raise ValueError('action opening and closing tags do not match')

        block = re.search(
            fr'<{action_kind}>(.*?)</{action_kind}>',
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if block is None:
            raise ValueError(f'invalid <{action_kind}> action')
        source = _strip_fence(block.group(1))
        if not source:
            raise ValueError('candidate program is empty')
        return source, action_kind

    if require_xml:
        raise ValueError('solution response must contain exactly one <python_code> or <answer> action')
    source = _strip_fence(text)
    if not source:
        raise ValueError('candidate program is empty')
    return source, 'plain_program'


def validate_program_source(source: str) -> None:
    """Syntax-check a complete candidate and require a supported top-level entrypoint."""
    tree = ast.parse(source, filename='<candidate>', mode='exec')
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not any(
        name == prefix or name.startswith(prefix)
        for name in function_names
        for prefix in _ENTRYPOINT_PREFIXES
    ):
        raise ValueError(
            'candidate must define solve, run_packing, or a min_max_dist_dim* function'
        )


def geometric_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Extract and syntax-check one complete candidate program per model action."""
    projected: List[str] = []
    valid: List[int] = []
    for action in actions:
        try:
            source, _ = parse_program_action(action)
            validate_program_source(source)
        except (SyntaxError, TypeError, ValueError):
            projected.append('')
            valid.append(0)
            continue
        projected.append(source)
        valid.append(1)
    return projected, valid
