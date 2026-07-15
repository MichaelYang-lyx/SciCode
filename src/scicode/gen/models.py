from __future__ import annotations

from functools import partial
import re

from openai import OpenAI
from scicode.utils.log import get_logger
from scicode.gen.token_records import record_response

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
    stream: bool = False,
    sample_id: str | None = None,
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
    if 'gpt' in model:
        create_kwargs["extra_body"]={'reasoning_effort': 'xhigh'}
    if repetition_penalty is not None:
        create_kwargs["extra_body"] = {"repetition_penalty": repetition_penalty}
    if stream:
        # 流式：逐 chunk 累加 delta.content / delta.reasoning_content 后拼接返回
        create_kwargs["stream"] = True
        create_kwargs["stream_options"] = {"include_usage": True}
        completion = client.chat.completions.create(**create_kwargs)
        pieces: list[str] = []
        reasoning_pieces: list[str] = []
        stream_usage = None
        for chunk in completion:
            if getattr(chunk, "usage", None):
                stream_usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                pieces.append(piece)
            r_piece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if r_piece:
                reasoning_pieces.append(r_piece)
        text = "".join(pieces)
        reasoning_text = "".join(reasoning_pieces) or None
        # think 计数优先用单独的 reasoning_content；否则由 record_response 从 <think> 拆
        record_response(stream_usage, text=text, reasoning_text=reasoning_text, sample_id=sample_id)
        return text
    completion = client.chat.completions.create(**create_kwargs)
    msg = completion.choices[0].message
    text = msg.content
    reasoning_text = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    record_response(getattr(completion, "usage", None), text=text, reasoning_text=reasoning_text, sample_id=sample_id)
    return text


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
    stream: bool = False,
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
        stream=stream,
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
