from __future__ import annotations

from functools import partial
import re

from openai import OpenAI
from scicode.utils.log import get_logger

logger = get_logger("models")


def generate_openai_compatible_response(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
    timeout: float = 3600.0,
    repetition_penalty: float | None = None,
) -> str:
    """Call any OpenAI-compatible API (OpenAI, Bailian, local vLLM, etc.)"""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    create_kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    if repetition_penalty is not None:
        create_kwargs["extra_body"] = {"repetition_penalty": repetition_penalty}
    completion = client.chat.completions.create(**create_kwargs)
    return completion.choices[0].message.content


def generate_dummy_response(prompt: str, **kwargs) -> str:
    """Used for testing as a substitute for actual models"""
    return "Blah blah\n```python\nprint('Hello, World!')\n```\n"


def get_model_function(
    model: str,
    *,
    api_key: str,
    base_url: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
    timeout: float = 3600.0,
    repetition_penalty: float | None = None,
    **kwargs,
):
    """Return a callable (prompt: str) -> str for the given model."""
    if model == "dummy":
        return partial(generate_dummy_response, model=model)

    return partial(
        generate_openai_compatible_response,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        repetition_penalty=repetition_penalty,
        **kwargs,
    )


def extract_python_script(response: str) -> str:
    if '```' in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if '```python' in response
            else response.split('```')[1].split('```')[0]
        )
    else:
        print("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(
        r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE
    )
    return python_script
