import contextlib
import importlib.util
import inspect
import io
import json
import sys
import traceback
from pathlib import Path


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f'object of type {type(value).__name__} is not JSON serializable')


def _call_entrypoint(function, instance):
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    if not parameters:
        return function()

    kwargs = {}
    missing = []
    for parameter in parameters:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.name in instance:
            kwargs[parameter.name] = instance[parameter.name]
        elif parameter.default is parameter.empty:
            missing.append(parameter)

    if not missing:
        return function(**kwargs)
    if len(parameters) == 1 and len(missing) == 1:
        return function(instance)
    names = ', '.join(parameter.name for parameter in missing)
    raise TypeError(f'cannot supply required entrypoint parameters: {names}')


def main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding='utf-8'))
    candidate_path = Path(request['candidate_path'])
    captured = io.StringIO()
    response = {}

    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            spec = importlib.util.spec_from_file_location('candidate_program', candidate_path)
            if spec is None or spec.loader is None:
                raise ImportError('could not load candidate program')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            entrypoint = None
            for name in request['entrypoints']:
                candidate = getattr(module, name, None)
                if callable(candidate):
                    entrypoint = name
                    function = candidate
                    break
            if entrypoint is None:
                expected = ', '.join(request['entrypoints'])
                raise AttributeError(f'candidate must define one of: {expected}')
            payload = _call_entrypoint(function, request['instance'])

        response = {
            'ok': True,
            'entrypoint': entrypoint,
            'payload': payload,
            'captured_output': captured.getvalue(),
        }
    except BaseException as exc:
        response = {
            'ok': False,
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'traceback': traceback.format_exc(),
            'captured_output': captured.getvalue(),
        }

    sys.stdout.write(json.dumps(response, default=_json_default, allow_nan=True))


if __name__ == '__main__':
    main()
