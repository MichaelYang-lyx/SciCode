"""校验同一个 SDK 客户端在流式与普通请求之间的隔离。"""
import asyncio
from types import SimpleNamespace

from scicode.gen.streaming import enable_streaming, streaming_preference


class Stream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get_final_completion(self):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])


class Resource:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append("nonstream")
        return "normal"

    def stream(self, **kwargs):
        self.calls.append("stream")
        return Stream()


def test_true_then_false_on_same_client():
    async def run():
        resource = Resource()
        model = SimpleNamespace(api=SimpleNamespace(client=SimpleNamespace(
            chat=SimpleNamespace(completions=resource))))
        enable_streaming(model)
        with streaming_preference(True):
            await resource.create()
        with streaming_preference(False):
            assert await resource.create() == "normal"
        from contextvars import Context
        assert await Context().run(asyncio.create_task, resource.create()) == "normal"
        assert resource.calls == ["stream", "nonstream", "nonstream"]

    asyncio.run(run())


def test_concurrent_preferences_are_isolated():
    async def run():
        resource = Resource()
        model = SimpleNamespace(api=SimpleNamespace(client=SimpleNamespace(
            chat=SimpleNamespace(completions=resource))))
        enable_streaming(model)

        async def call(value):
            with streaming_preference(value):
                await asyncio.sleep(0)
                await resource.create()

        await asyncio.gather(call(True), call(False))
        assert sorted(resource.calls) == ["nonstream", "stream"]

    asyncio.run(run())


def test_openai_requests_follow_per_call_preference():
    import json
    import httpx
    from openai import AsyncOpenAI

    async def run():
        observed = []

        async def handle(request):
            payload = json.loads(request.content)
            observed.append(payload.get("stream", False))
            message = {"id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                       "model": "candidate", "choices": [{"index": 0,
                       "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
            if not payload.get("stream"):
                return httpx.Response(200, json=message)
            chunks = [
                {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
                 "model": "candidate", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
                 "model": "candidate", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            body = "".join("data: " + json.dumps(item) + "\n\n" for item in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

        async with AsyncOpenAI(api_key="fixture", base_url="https://model.invalid/v1",
                               max_retries=0, http_client=httpx.AsyncClient(
                                   transport=httpx.MockTransport(handle))) as client:
            enable_streaming(SimpleNamespace(api=SimpleNamespace(client=client)))
            for value in (True, False, True):
                with streaming_preference(value):
                    result = await client.chat.completions.create(
                        model="candidate", messages=[{"role": "user", "content": "test"}])
                    assert result.choices[0].message.content == "ok"
        assert observed == [True, False, True]

    asyncio.run(run())
