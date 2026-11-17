#!/usr/bin/env python3
"""Prepare an explicit, secret-free experiment and run the verified pipeline."""
import argparse
import json
import os
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='SciCode-Verified checkout with v2 data')
    parser.add_argument('--env-2024', required=True, help='2024 grading Python executable')
    parser.add_argument('--env-2025', required=True, help='2025 grading Python executable')
    parser.add_argument('--output', required=True, help='new experiment directory')
    parser.add_argument('--model', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--protocol', choices=['openai_chat', 'anthropic'], default='openai_chat')
    parser.add_argument('--api-key-env', default='SCICODE_VERIFIED_API_KEY')
    parser.add_argument('--max-tokens', type=int, default=65536)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--grading-workers', type=int, default=1)
    parser.add_argument('--cpu-slots', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--max-active-rounds', type=int, default=1)
    parser.add_argument('--api-timeout', type=int, default=1800)
    parser.add_argument('--grading-timeout', type=int, default=1800)
    parser.add_argument('--only', default='', help='comma-separated problem IDs')
    parser.add_argument('--without-background', action='store_true')
    parser.add_argument('--non-stream', action='store_true')
    args = parser.parse_args()
    if not os.environ.get(args.api_key_env):
        parser.error('API key environment variable is missing')
    if min(args.max_tokens, args.workers, args.grading_workers, args.cpu_slots,
           args.rounds, args.max_active_rounds, args.api_timeout, args.grading_timeout) < 1:
        parser.error('counts and timeouts must be positive')
    out = pathlib.Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    output = out / 'outputs'
    output.mkdir()
    config = {
        'source': str(pathlib.Path(args.source).expanduser().resolve()),
        'output': str(output),
        'model': {'name': args.model, 'base_url': args.base_url, 'protocol': args.protocol,
                  'api_key_env': args.api_key_env},
        'generation': {'max_tokens': args.max_tokens, 'temperature': None, 'top_p': None,
                       'top_k': None, 'reasoning_effort': None, 'token_field': 'max_tokens',
                       'strict_effort': True, 'stop_on_api_error': False, 'extra_body': {},
                       'workers_per_round': args.workers, 'api_timeout_seconds': args.api_timeout,
                       'stream': not args.non_stream},
        'grading': {'envs': {'2024': str(pathlib.Path(args.env_2024).expanduser().absolute()),
                             '2025': str(pathlib.Path(args.env_2025).expanduser().absolute())},
                    'timeout_seconds': args.grading_timeout,
                    'workers_per_round': args.grading_workers,
                    'global_cpu_slots': args.cpu_slots, 'slot_dir': str(out / 'cpu_slots')},
        'benchmark': {'version': 'v2', 'rounds': args.rounds,
                      'with_background': not args.without_background, 'only': args.only},
        'execution_policy': {'max_active_rounds': args.max_active_rounds},
    }
    experiment = out / 'experiment.json'
    experiment.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    runner = pathlib.Path(__file__).with_name('pipeline.py')
    return subprocess.call([sys.executable, str(runner), '--experiment', str(experiment)])


if __name__ == '__main__':
    sys.exit(main())
