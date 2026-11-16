from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from alphaapollo.core.environments.geometric.projection import (
    parse_program_action,
    validate_program_source,
)


_STRATEGY_USER_MARKERS = (
    "rewrite the program to improve search algorithm score",
    "# your rewritten search algorithm here",
)

_STRATEGY_SYSTEM_MARKERS = (
    "evolving a search algorithm for program optimization",
    "summarize the population state with number-backed observations",
    "summarize the downstream problem context for a search algorithm designer",
    "summarizing previous search algorithm attempts",
    "generate problem-specific variation operators",
    "suggesting diverse algorithmic approaches",
    "different approaches sections",
    "refinement strategies guidance block",
)


@dataclass(frozen=True)
class LLMCallRecord:
    index: int
    category: str
    model: str
    started_at: float
    duration_seconds: float
    status: str
    prompt_chars: int
    response_chars: int
    prompt_sha256: str
    prompt_text: str = ""
    response_sha256: str = ""
    action_kind: str = ""
    normalized_source_chars: int = 0
    raw_response: str = ""
    normalized_source: str = ""
    error_type: str = ""
    error_message: str = ""
    parse_error: str = ""


def normalize_solution_action(text: str) -> Tuple[str, str]:
    """Strictly convert one AlphaApollo XML program action for SkyDiscover."""
    source, action_kind = parse_program_action(text, require_xml=True)
    validate_program_source(source)
    return source, action_kind


def _set_first_choice_content(response: Any, content: str) -> None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("LLM response contains no choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("LLM response choice contains no message")
    try:
        message.content = content
    except Exception:
        object.__setattr__(message, "content", content)


class LLMCallRecorder:
    """Thread-safe accounting and XML normalization for actual API attempts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[LLMCallRecord] = []

    @staticmethod
    def classify(params: Dict[str, Any]) -> str:
        messages = params.get("messages", []) or []
        contents = [str(message.get("content", "")) for message in messages]
        if len(contents) <= 2 and any(text.strip().lower() == "ping" for text in contents):
            return "probe"
        # EvoX shares one client across downstream solution generation and
        # several meta-search helpers. Inspect message roles instead of the
        # concatenated prompt because variation guidance is embedded verbatim
        # inside legitimate downstream solution prompts.
        system_prompt = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if str(message.get("role", "")).lower() == "system"
        ).lower()
        user_messages = [
            str(message.get("content", ""))
            for message in messages
            if str(message.get("role", "")).lower() == "user"
        ]
        last_user = (user_messages[-1] if user_messages else "").lower()
        if any(marker in system_prompt for marker in _STRATEGY_SYSTEM_MARKERS):
            return "strategy"
        if any(marker in last_user for marker in _STRATEGY_USER_MARKERS):
            return "strategy"
        return "solution"

    def record(
        self,
        *,
        category: str,
        model: str,
        started_at: float,
        duration_seconds: float,
        status: str,
        prompt: str,
        response_text: str = "",
        action_kind: str = "",
        normalized_source: str = "",
        error_type: str = "",
        error_message: str = "",
        parse_error: str = "",
    ) -> None:
        with self._lock:
            self._records.append(
                LLMCallRecord(
                    index=len(self._records) + 1,
                    category=category,
                    model=model,
                    started_at=started_at,
                    duration_seconds=duration_seconds,
                    status=status,
                    prompt_chars=len(prompt),
                    response_chars=len(response_text),
                    prompt_sha256=hashlib.sha256(
                        prompt.encode("utf-8", errors="replace")
                    ).hexdigest(),
                    prompt_text=prompt,
                    response_sha256=(
                        hashlib.sha256(
                            response_text.encode("utf-8", errors="replace")
                        ).hexdigest()
                        if response_text
                        else ""
                    ),
                    action_kind=action_kind,
                    normalized_source_chars=len(normalized_source),
                    raw_response=response_text if category == "solution" else "",
                    normalized_source=normalized_source if category == "solution" else "",
                    error_type=error_type,
                    error_message=error_message,
                    parse_error=parse_error,
                )
            )

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(record) for record in self._records]

    def counts(self) -> Dict[str, int]:
        counts = {
            "total": 0,
            "solution": 0,
            "strategy": 0,
            "probe": 0,
            "failed": 0,
            "invalid_xml": 0,
        }
        for record in self.records():
            counts["total"] += 1
            counts[record["category"]] = counts.get(record["category"], 0) + 1
            if record["status"] != "ok":
                counts["failed"] += 1
            if record["status"] == "invalid_xml":
                counts["invalid_xml"] += 1
        return counts

    def client_factory(self):
        recorder = self

        def create_client(model_cfg):
            from skydiscover.optimize.llm.openai import OpenAILLM

            client = OpenAILLM(model_cfg)
            completions = client.client.chat.completions
            original_create = completions.create

            def create_with_tracking(**params):
                extra_body = dict(params.get("extra_body") or {})
                extra_body.setdefault("enable_thinking", False)
                params["extra_body"] = extra_body

                started = time.time()
                prompt = "\n".join(
                    str(message.get("content", ""))
                    for message in (params.get("messages", []) or [])
                )
                category = recorder.classify(params)
                try:
                    response = original_create(**params)
                except Exception as exc:
                    recorder.record(
                        category=category,
                        model=str(model_cfg.name),
                        started_at=started,
                        duration_seconds=time.time() - started,
                        status="error",
                        prompt=prompt,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    raise

                text = ""
                choices = getattr(response, "choices", None) or []
                if choices:
                    text = str(getattr(choices[0].message, "content", "") or "")

                action_kind = ""
                normalized_source = ""
                if category == "solution":
                    try:
                        normalized_source, action_kind = normalize_solution_action(text)
                        _set_first_choice_content(
                            response, f"```python\n{normalized_source}\n```"
                        )
                    except (SyntaxError, TypeError, ValueError) as exc:
                        recorder.record(
                            category=category,
                            model=str(model_cfg.name),
                            started_at=started,
                            duration_seconds=time.time() - started,
                            status="invalid_xml",
                            prompt=prompt,
                            response_text=text,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            parse_error=str(exc),
                        )
                        raise ValueError(
                            f"invalid AlphaApollo solution action: {exc}"
                        ) from exc

                recorder.record(
                    category=category,
                    model=str(model_cfg.name),
                    started_at=started,
                    duration_seconds=time.time() - started,
                    status="ok",
                    prompt=prompt,
                    response_text=text,
                    action_kind=action_kind,
                    normalized_source=normalized_source,
                )
                return response

            completions.create = create_with_tracking
            return client

        return create_client
