import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('verified_pipeline', ROOT / 'pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class Response:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return io.BytesIO(json.dumps(self.payload).encode())
    def __exit__(self, *args):
        return False


class RequestTests(unittest.TestCase):
    def config(self, protocol='openai_chat'):
        return {'model': {'name': 'test-model', 'base_url': 'https://example.test/v1',
                          'protocol': protocol, 'api_key_env': 'FAKE_TEST_KEY'},
                'generation': {'max_tokens': 123, 'temperature': None, 'top_p': None,
                               'top_k': None, 'reasoning_effort': None, 'stream': False,
                               'api_timeout_seconds': 10, 'extra_body': {}}}

    def test_openai_request_and_invalid_short_response(self):
        seen = []
        def request(req, **kwargs):
            seen.append((req.full_url, json.loads(req.data)))
            return Response({'choices': [{'message': {'content': '!'}, 'finish_reason': 'stop'}]})
        with mock.patch.dict(os.environ, {'FAKE_TEST_KEY': 'dummy'}), mock.patch.object(pipeline.urllib.request, 'urlopen', request):
            with self.assertRaisesRegex(pipeline.GenerationIncomplete, 'one-character'):
                pipeline.generate(self.config(), 'hello')
        self.assertEqual(seen[0][0], 'https://example.test/v1/chat/completions')
        self.assertEqual(seen[0][1]['max_tokens'], 123)
        self.assertEqual(seen[0][1]['messages'][0]['content'], 'hello')

    def test_anthropic_request(self):
        seen = []
        def request(req, **kwargs):
            seen.append((req.full_url, json.loads(req.data)))
            return Response({'content': [{'type': 'text', 'text': 'code'}], 'stop_reason': 'end_turn'})
        with mock.patch.dict(os.environ, {'FAKE_TEST_KEY': 'dummy'}), mock.patch.object(pipeline.urllib.request, 'urlopen', request):
            result = pipeline.generate(self.config('anthropic'), 'hello')
        self.assertEqual(result['content'], 'code')
        self.assertEqual(seen[0][0], 'https://example.test/v1/messages')
        self.assertEqual(seen[0][1]['max_tokens'], 123)

    def test_resume_generation_compatibility(self):
        a = self.config()['generation']
        b = dict(a, workers_per_round=4, api_timeout_seconds=3600)
        a['workers_per_round'] = 1
        self.assertTrue(pipeline.generation_compatible(a, b))
        self.assertFalse(pipeline.generation_compatible(a, dict(b, temperature=0.5)))


if __name__ == '__main__':
    unittest.main()
