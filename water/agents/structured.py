__all__ = [
    "create_structured_task",
]

"""
Structured Output for Water Agent Tasks.

Provides a factory function that creates tasks returning validated Pydantic
model instances.  The LLM is prompted to respond in JSON matching a given
schema; the response is parsed, validated, and retried with error feedback
on failure.
"""

import json
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from water.agents.llm import (
    LLMProvider,
    MockProvider,
    _resolve_provider,
    estimate_token_count,
    _normalize_usage,
)
from water.core.task import Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract a JSON object from *text*.

    Looks for JSON inside markdown fences (```json ... ``` or ``` ... ```)
    first, then falls back to the raw text.  Returns the extracted string
    (still needs ``json.loads``).
    """
    # Try markdown code fences first
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to raw text (strip surrounding whitespace)
    return text.strip()


def _schema_to_prompt(model_cls: Type[BaseModel]) -> str:
    """Generate JSON schema instructions from a Pydantic model class.

    Returns a string suitable for appending to a system prompt so the LLM
    knows exactly what JSON structure to produce.
    """
    schema = model_cls.model_json_schema()

    # Build a concise field description
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines: List[str] = []
    for name, prop in properties.items():
        typ = prop.get("type", "any")
        desc = prop.get("description", "")
        req = " (required)" if name in required else " (optional)"
        line = f'  - "{name}": {typ}{req}'
        if desc:
            line += f" — {desc}"
        lines.append(line)

    field_block = "\n".join(lines) if lines else "  (no fields)"

    return (
        "\n\nYou MUST respond with valid JSON matching the following schema.\n"
        "Do NOT include any text outside the JSON object.\n\n"
        f"Fields:\n{field_block}\n\n"
        f"JSON Schema:\n{json.dumps(schema, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_structured_task(
    id: Optional[str] = None,
    description: Optional[str] = None,
    prompt_template: str = "",
    model: str = "default",
    provider: str = "openai",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    output_parser: Optional[Callable] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    retry_count: int = 0,
    timeout: Optional[float] = None,
    custom_fn: Optional[Callable] = None,
    provider_instance: Optional[LLMProvider] = None,
    model_cls: Optional[Type[BaseModel]] = None,
    max_retries: int = 3,
    mode: str = "json_prompt",
) -> Task:
    """Create a Task that wraps an LLM call and validates output against a Pydantic model.

    The returned :class:`Task` integrates seamlessly with Flow, retries,
    caching, etc.  On each execution the LLM is instructed to respond in
    JSON matching *model_cls*.  The response is parsed, validated, and — if
    validation fails — the LLM is re-prompted with error details up to
    *max_retries* times.

    Args:
        id: Task identifier.  Auto-generated if not provided.
        description: Human-readable description.
        prompt_template: String with ``{variable}`` placeholders formatted
            with the task's input data.
        model: Model identifier passed to the provider.
        provider: One of ``"openai"``, ``"anthropic"``, ``"custom"``,
            ``"mock"``.
        api_key: API key for the provider (or use env vars).
        system_prompt: Optional system message prepended to the conversation.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        output_parser: Optional extra callable applied **after** Pydantic
            validation.  Receives the validated model instance and should
            return a dict.
        input_schema: Pydantic model for task input validation.
        output_schema: Pydantic model for task output validation.  Defaults
            to *model_cls* if not provided.
        retry_count: Number of retries on failure (passed through to Task).
        timeout: Timeout in seconds for the task execution.
        custom_fn: Async callable for the ``"custom"`` provider.
        provider_instance: Pre-built :class:`LLMProvider` instance.
        model_cls: The Pydantic model class the LLM response must conform to.
        max_retries: Maximum number of parse/validation retries with error
            feedback before raising.
        mode: Extraction mode — currently only ``"json_prompt"`` is supported.

    Returns:
        A :class:`Task` instance ready to be added to a Flow.

    Raises:
        ValueError: If *model_cls* is not provided or *mode* is unsupported.
    """
    if model_cls is None:
        raise ValueError("create_structured_task requires a model_cls (Pydantic BaseModel)")

    if mode != "json_prompt":
        raise ValueError(f"Unsupported mode {mode!r}. Currently only 'json_prompt' is supported.")

    task_id = id or f"structured_{uuid.uuid4().hex[:8]}"

    if not input_schema:
        input_schema = type(
            f"{task_id}_Input",
            (BaseModel,),
            {"__annotations__": {"prompt": str}},
        )
    if not output_schema:
        output_schema = model_cls

    llm_provider = _resolve_provider(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        custom_fn=custom_fn,
        provider_instance=provider_instance,
    )

    # Build the augmented system prompt once
    schema_instructions = _schema_to_prompt(model_cls)
    full_system_prompt = (system_prompt or "") + schema_instructions

    async def execute(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        input_data = params.get("input_data", params)

        # 1. Format prompt template with input variables
        if prompt_template:
            try:
                user_content = prompt_template.format(**input_data)
            except KeyError as exc:
                raise ValueError(
                    f"StructuredTask: prompt template variable {exc} not found "
                    f"in input data. Available keys: {list(input_data.keys())}"
                )
        else:
            user_content = str(input_data.get("prompt", ""))

        # 2. Build initial messages
        messages: List[Dict[str, str]] = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        messages.append({"role": "user", "content": user_content})

        last_error: Optional[str] = None

        for attempt in range(1 + max_retries):
            # On retry, append error feedback as a user message
            if last_error is not None:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response failed validation:\n{last_error}\n\n"
                        "Please fix and respond with valid JSON matching the schema."
                    ),
                })

            # 3. Estimate tokens and log
            total_input_text = "".join(m["content"] for m in messages)
            estimated_tokens = estimate_token_count(total_input_text)
            logger.info(
                "Structured LLM call [%s] attempt %d/%d: ~%d input tokens",
                task_id, attempt + 1, 1 + max_retries, estimated_tokens,
            )

            # 4. Call the provider
            response = await llm_provider.complete(messages)
            response_text = response.get("text", "")

            # Build cost-tracking metadata
            cost_meta: Dict[str, Any] = {}
            usage = _normalize_usage(response.get("usage"))
            if usage:
                cost_meta["usage"] = usage
                cost_meta["model"] = getattr(llm_provider, "model", "unknown")

            # 5. Extract & validate
            try:
                raw_json = _extract_json(response_text)
                data = json.loads(raw_json)
                validated = model_cls.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Structured output [%s] attempt %d failed: %s",
                    task_id, attempt + 1, last_error,
                )
                # Add the assistant response to conversation so the LLM
                # can see what it produced.
                messages.append({"role": "assistant", "content": response_text})
                if attempt < max_retries:
                    continue
                raise ValueError(
                    f"Structured output validation failed after {max_retries + 1} "
                    f"attempts. Last error: {last_error}"
                ) from exc

            # 6. Success — return as dict
            result = validated.model_dump()
            if output_parser:
                parsed = output_parser(validated)
                if isinstance(parsed, dict):
                    return {**parsed, **cost_meta}
                return {"response": parsed, **cost_meta}

            return {**result, **cost_meta}

        # Should not be reached, but just in case
        raise ValueError(  # pragma: no cover
            f"Structured output validation failed for task {task_id}"
        )

    return Task(
        id=task_id,
        description=description or f"Structured agent task: {task_id}",
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
        retry_count=retry_count,
        timeout=timeout,
        validate_schema=False,
    )
