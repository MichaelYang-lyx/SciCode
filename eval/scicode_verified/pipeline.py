"""SciCode-Verified v2：显式配置、隔离评分和三轮独立采样。"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import http.client
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class InfrastructureError(RuntimeError):
    """评测基础设施异常，不能计为模型答错。"""


class GenerationIncomplete(RuntimeError):
    """生成被截断或流未正常结束，保留已收到的原始响应。"""

    def __init__(self, message: str, response: dict[str, Any]):
        super().__init__(message)
        self.response = response


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".atomic-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def record_failure(config: dict[str, Any], round_no: int, scope: str,
                   identifier: str, exc: BaseException) -> None:
    """Keep the original worker error independently of cancellation errors."""
    message = str(exc)
    if message.startswith("Evaluation stopped"):
        return
    secret = os.environ.get(config["model"]["api_key_env"], "")
    if secret:
        message = message.replace(secret, "[redacted]")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", identifier)
    atomic_json(Path(config["output"]) / "errors" / f"round_{round_no}" /
                f"{scope}_{safe_id}.json", {"error_type": type(exc).__name__,
                                            "error_message": message})


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".atomic-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    stable = {k: v for k, v in config.items() if k not in {"preflight", "runtime"} and not k.startswith("_")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generation_compatible(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare request parameters, excluding the controller error policy."""
    return (isinstance(old, dict) and isinstance(new, dict)
            and {k: v for k, v in old.items() if k not in {"max_tokens", "stop_on_api_error", "api_timeout_seconds", "workers_per_round"}}
            == {k: v for k, v in new.items() if k not in {"max_tokens", "stop_on_api_error", "api_timeout_seconds", "workers_per_round"}}
            and ((old.get("max_tokens") is None and new.get("max_tokens") is None)
                 or (isinstance(old.get("max_tokens"), int)
                     and isinstance(new.get("max_tokens"), int)
                     and old["max_tokens"] <= new["max_tokens"])))


def verify_config(config: dict[str, Any]) -> None:
    current = json.loads(Path(config["_experiment_path"]).read_text())
    if config_digest(current) != config["_config_digest"]:
        raise InfrastructureError("Experiment configuration changed after start")


def cpu_capacity() -> int:
    count = len(os.sched_getaffinity(0))
    for entry in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = entry.split(":", 2)
        if controllers == "":
            root = Path("/sys/fs/cgroup")
            node = root / relative.lstrip("/")
            while node == root or root in node.parents:
                try:
                    quota, period = (node / "cpu.max").read_text().split()
                    if quota != "max":
                        count = min(count, max(1, int(quota) // int(period)))
                except (OSError, ValueError):
                    pass
                if node == root:
                    break
                node = node.parent
        elif "cpu" in controllers.split(","):
            root = Path("/sys/fs/cgroup/cpu")
            node = root / relative.lstrip("/")
            while node == root or root in node.parents:
                try:
                    quota = int((node / "cpu.cfs_quota_us").read_text())
                    period = int((node / "cpu.cfs_period_us").read_text())
                    if quota > 0:
                        count = min(count, max(1, quota // period))
                except (OSError, ValueError, ZeroDivisionError):
                    pass
                if node == root:
                    break
                node = node.parent
    return count


def load_upstream(source: Path) -> Any:
    path = source / "eval_clean" / "run_deepseek_eval.py"
    spec = importlib.util.spec_from_file_location("scicode_verified_upstream", path)
    if not spec or not spec.loader:
        raise InfrastructureError(f"Cannot load upstream runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def preserve_imports(response: str) -> str:
        if not response:
            return ""
        if "```python" in response:
            return response.split("```python", 1)[1].split("```", 1)[0]
        if "```" in response:
            return response.split("```", 1)[1].split("```", 1)[0]
        return response
    module.extract_code = preserve_imports
    return module


def preflight(config: dict[str, Any], upstream: Any) -> list[dict[str, Any]]:
    source = Path(config["source"])
    data_dir = source / "scicode_verified"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("version") != "v2" or manifest.get("n_problems") != 64:
        raise InfrastructureError("Expected the SciCode-Verified v2 manifest with 64 problems")
    for file, key in ((data_dir / "problems_test.jsonl", "problems_test_jsonl_md5"),
                      (data_dir / "test_data_cleaned.h5", "h5_md5")):
        if not file.is_file() or md5_file(file) != manifest.get(key):
            raise InfrastructureError(f"Verified data missing or manifest checksum mismatch: {file.name}")
    problems = [json.loads(line) for line in (data_dir / "problems_test.jsonl").read_text().splitlines() if line.strip()]
    steps = sum(len(problem["sub_steps"]) for problem in problems) - len(upstream.SKIP)
    if len(problems) != 64 or steps != 287:
        raise InfrastructureError(f"Unexpected data coverage: {len(problems)} problems, {steps} steps")
    only = {x.strip() for x in config["benchmark"].get("only", "").split(",") if x.strip()}
    if only:
        problems = [p for p in problems if str(p["problem_id"]) in only]
        if {str(p["problem_id"]) for p in problems} != only:
            raise InfrastructureError("Requested smoke problem IDs are not all in the Verified dataset")
    envs = config["grading"]["envs"]
    if set(envs) != {"2024", "2025"}:
        raise InfrastructureError("Exactly the 2024 and 2025 grading environments are required")
    for label, py in envs.items():
        if not Path(py).is_file():
            raise InfrastructureError(f"Missing grading interpreter {label}: {py}")
        result = subprocess.run([py, "-c", "import numpy,scipy,h5py,sympy"],
                                capture_output=True, text=True, timeout=60,
                                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        if result.returncode:
            raise InfrastructureError(f"Grading environment {label} cannot import required libraries")
    if not shutil.which("bwrap") or not shutil.which("taskset"):
        raise InfrastructureError("bwrap and taskset are required")
    if cpu_capacity() < config["grading"]["global_cpu_slots"]:
        raise InfrastructureError("CPU slot count exceeds the process affinity or cgroup quota")
    output = Path(config["output"])
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    tested_envs = {}
    host_net = os.readlink("/proc/self/ns/net")
    for label, py in envs.items():
        with tempfile.TemporaryDirectory(prefix="preflight-", dir=runtime) as temp:
            script = Path(temp) / "preflight.py"
            script.write_text("import os, sys, json, numpy, scipy, h5py, sympy\n"
                              f"assert os.readlink('/proc/self/ns/net') != {host_net!r}\n"
                              "assert 'SCICODE_VERIFIED_API_KEY' not in os.environ\n"
                              f"assert not os.path.exists({str(Path.home() / '.ssh')!r})\n"
                              "sys.path.insert(0, '/vendor')\n"
                              "from scicode.parse.parse import process_hdf5_to_tuple\n"
                              "with h5py.File('/test_data.h5', 'r') as h: assert len(h.keys()) > 0\n"
                              "print(json.dumps({'numpy':numpy.__version__, 'scipy':scipy.__version__, "
                              "'h5py':h5py.__version__, 'sympy':sympy.__version__}))\n")
            with cpu_slot(config, threading.Event()) as cpu:
                result = subprocess.run(sandbox_command(py, script, source, cpu),
                                        capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise InfrastructureError(f"Sandbox or data preflight failed for {label}: {result.stderr[-500:]}")
            tested_envs[label] = json.loads(result.stdout.strip().splitlines()[-1])
    config["preflight"] = {"manifest_version": "v2", "problems": 64, "scored_steps": 287,
                           "grading_env_versions": tested_envs, "cpu_capacity": cpu_capacity(),
                           "network_isolated": True, "api_key_hidden": True,
                           "source_hashes": {str(path.relative_to(source)): sha256_file(path) for path in [
                               source / "eval_clean/run_deepseek_eval.py",
                               source / "eval_clean/vendor/scicode/parse/parse.py",
                               source / "scicode_verified/manifest.json"]}}
    return problems


def _endpoint(base: str, protocol: str) -> str:
    base = base.rstrip("/")
    if protocol == "anthropic":
        return (base if base.endswith("/v1") else base + "/v1") + "/messages"
    return base + "/chat/completions"


def generate(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    model, gen = config["model"], config["generation"]
    api_key = os.environ.get(model["api_key_env"], "")
    extra = dict(gen.get("extra_body") or {})
    forbidden = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "stream", "stream_options"} & extra.keys()
    if forbidden:
        raise InfrastructureError(f"extra_body contains reserved request fields: {sorted(forbidden)}")
    effort = gen.get("reasoning_effort")
    if effort is not None:
        if "reasoning_effort" in extra and extra["reasoning_effort"] != effort:
            raise InfrastructureError("Conflicting reasoning_effort settings")
        extra["reasoning_effort"] = effort
    for key in ("top_p", "top_k"):
        if gen.get(key) is not None:
            if key in extra and extra[key] != gen[key]:
                raise InfrastructureError(f"Conflicting {key} settings")
            extra[key] = gen[key]
    if model["protocol"] == "anthropic" and "reasoning_effort" in extra:
        effort = extra.pop("reasoning_effort")
        output_config = dict(extra.get("output_config") or {})
        if "effort" in output_config and output_config["effort"] != effort:
            raise InfrastructureError("Conflicting Anthropic effort settings")
        output_config["effort"] = effort
        extra["output_config"] = output_config
    token_field = gen.get("token_field", "max_tokens")
    body: dict[str, Any] = {"model": model["name"], "messages": [{"role": "user", "content": prompt}],
                            "stream": gen["stream"], **extra}
    if gen["max_tokens"] is not None or model["protocol"] == "anthropic":
        body[token_field] = gen["max_tokens"]
    if gen["temperature"] is not None:
        body["temperature"] = gen["temperature"]
    headers = {"Content-Type": "application/json"}
    if model["protocol"] == "anthropic":
        headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        if body.get("thinking", {}).get("type") in {"adaptive", "enabled"}:
            body.pop("temperature", None)
    else:
        headers["Authorization"] = f"Bearer {api_key}"
        if gen["stream"]:
            body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(_endpoint(model["base_url"], model["protocol"]),
                                 data=json.dumps(body).encode(), headers=headers, method="POST")
    chunks, thought, finish, usage = [], [], None, None
    stream_diagnostics = []
    deadline = time.monotonic() + gen["api_timeout_seconds"]
    try:
        with urllib.request.urlopen(req, timeout=gen["api_timeout_seconds"]) as response:
            if not gen["stream"]:
                data = json.load(response)
                if model["protocol"] == "anthropic":
                    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                    thinking = "".join(b.get("thinking", "") for b in data.get("content", []) if b.get("type") == "thinking")
                    finish = data.get("stop_reason")
                else:
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    text, thinking, finish = message.get("content") or "", message.get("reasoning_content") or "", choice.get("finish_reason")
                usage = data.get("usage")
            else:
                terminal = False
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Model request time limit exceeded")
                    sock = response.fp.raw._sock
                    sock.settimeout(remaining)
                    raw = response.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    value = line[5:].strip()
                    if value == "[DONE]":
                        terminal = True
                        continue
                    event = json.loads(value)
                    choices = event.get("choices") or []
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        stream_diagnostics.append({"content_chars": len(content), "content_is_exclamation": content.strip() == "!", "reasoning_chars": len(delta.get("reasoning_content") or delta.get("reasoning") or ""), "finish_reason": choice.get("finish_reason")})
                        if len(stream_diagnostics) > 12:
                            del stream_diagnostics[0]
                    if event.get("error") or event.get("type") == "error":
                        raise GenerationIncomplete("Model stream reported an error", {"content": "".join(chunks), "reasoning_content": "".join(thought), "finish_reason": finish, "usage": usage})
                    if model["protocol"] == "anthropic":
                        kind = event.get("type")
                        if kind == "message_start":
                            usage = (event.get("message") or {}).get("usage") or {}
                        elif kind == "content_block_start":
                            block = event.get("content_block") or {}
                            chunks.append(block.get("text") or "")
                            thought.append(block.get("thinking") or "")
                        elif kind == "content_block_delta":
                            delta = event.get("delta") or {}
                            chunks.append(delta.get("text") or "")
                            thought.append(delta.get("thinking") or "")
                        elif kind == "message_delta":
                            finish = (event.get("delta") or {}).get("stop_reason")
                            usage = {**(usage or {}), **(event.get("usage") or {})}
                        elif kind == "message_stop":
                            terminal = True
                    else:
                        if event.get("usage") is not None:
                            usage = event["usage"]
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            chunks.append(delta.get("content") or "")
                            thought.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                            finish = choice.get("finish_reason") or finish
                text, thinking = "".join(chunks), "".join(thought)
                if not terminal:
                    raise GenerationIncomplete("Model stream did not include a terminal event",
                                               {"content": text, "reasoning_content": thinking or None,
                                                "finish_reason": finish, "usage": usage})
    except urllib.error.HTTPError as exc:
        detail = {}
        try:
            payload = json.loads(exc.read(65536).decode("utf-8", errors="replace"))
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                detail = {k: error[k] for k in ("type", "code", "param") if k in error}
        except (ValueError, OSError):
            pass
        raise GenerationIncomplete(f"Model API request failed: HTTP {exc.code}; details={detail}",
                                   {"content": "".join(chunks), "reasoning_content": "".join(thought),
                                    "finish_reason": finish, "usage": usage}) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, http.client.HTTPException) as exc:
        raise GenerationIncomplete(f"Model API request failed: {type(exc).__name__}",
                                   {"content": "".join(chunks), "reasoning_content": "".join(thought),
                                    "finish_reason": finish, "usage": usage}) from exc
    if text.strip() == "!":
        raise GenerationIncomplete("Model returned an invalid one-character response",
                                   {"content": text, "reasoning_content": thinking or None,
                                    "finish_reason": finish, "usage": usage,
                                    "stream_diagnostics": stream_diagnostics})
    valid_finish = {"end_turn", "stop_sequence"} if model["protocol"] == "anthropic" else {"stop"}
    if finish not in valid_finish or not text:
        raise GenerationIncomplete(f"Model response incomplete or empty (finish_reason={finish!r})",
                                   {"content": text, "reasoning_content": thinking or None,
                                    "finish_reason": finish, "usage": usage})
    return {"content": text, "reasoning_content": thinking or None, "finish_reason": finish, "usage": usage}


@contextmanager
def cpu_slot(config: dict[str, Any], stop: threading.Event):
    grading = config["grading"]
    directory = Path(grading["slot_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))[:grading["global_cpu_slots"]]
    while not stop.is_set():
        for cpu in cpus:
            f = (directory / f"cpu-{cpu}.lock").open("a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield cpu
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(0.2)
    raise InfrastructureError("Evaluation stopped while waiting for a CPU slot")


def sandbox_command(py: str, script: Path, source: Path, cpu: int) -> list[str]:
    venv = Path(py).parent.parent
    # uv 的 venv 可执行文件经版本别名链接到真实解释器，须同时挂载别名。
    interpreter_root = Path(os.path.realpath(py)).parent.parent.parent
    if str(interpreter_root) in {"/", "/home", str(Path.home()), str(Path.home().parent), "/opt"}:
        raise InfrastructureError("Grading interpreter layout would expose an overly broad directory")
    mounts = [(venv, venv), (interpreter_root, interpreter_root),
              (source / "eval_clean" / "vendor", Path("/vendor")),
              (source / "scicode_verified" / "test_data_cleaned.h5", Path("/test_data.h5")),
              (script, Path("/test.py"))]
    ancestors: set[str] = set()
    for _, dst in mounts[:2]:
        for parent in dst.parents:
            if str(parent) not in {"/", "/usr", "/bin", "/lib", "/lib64"}:
                ancestors.add(str(parent))
    cmd = ["env", "-i", "taskset", "-c", str(cpu), "bwrap", "--die-with-parent", "--unshare-all", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
           "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--dir", "/etc", "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for directory in sorted(ancestors, key=lambda x: (x.count("/"), x)):
        cmd.extend(["--dir", directory])
    for src, dst in mounts:
        cmd.extend(["--ro-bind", str(src), str(dst)])
    for key, value in {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": "/tmp/mpl"}.items():
        cmd.extend(["--setenv", key, value])
    cmd.extend(["--chdir", "/tmp", py, "/test.py"])
    return cmd


def grade_step(config: dict[str, Any], code: str, cases: list[str], step_id: str,
               directory: Path, stop: threading.Event) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f".{step_id}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / f"{step_id}.score.json").exists():
            raise InfrastructureError(f"Duplicate score execution for step {step_id}")
        return _grade_step_locked(config, code, cases, step_id, directory, stop)


def _grade_step_locked(config: dict[str, Any], code: str, cases: list[str], step_id: str,
                       directory: Path, stop: threading.Event) -> dict[str, Any]:
    if stop.is_set():
        raise InfrastructureError("Evaluation stopped before grading")
    lines = [code, "import sys; sys.path.insert(0, '/vendor')",
             "from scicode.parse.parse import process_hdf5_to_tuple",
             f"targets = process_hdf5_to_tuple({step_id!r}, {len(cases)}, '/test_data.h5')"]
    for index, case in enumerate(cases):
        lines.extend([f"target = targets[{index}]", case])
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grade-", dir=directory) as temp:
        script = Path(temp) / "test.py"
        script.write_text("\n".join(lines), encoding="utf-8")
        env_results, within_300 = {}, {}
        for label, python in config["grading"]["envs"].items():
            with cpu_slot(config, stop) as cpu:
                start = time.monotonic()
                cmd = sandbox_command(python, script, Path(config["source"]), cpu)
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, start_new_session=True)
                try:
                    stdout, stderr = proc.communicate(timeout=config["grading"]["timeout_seconds"])
                    timed_out = False
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    stdout, stderr = proc.communicate()
                    timed_out = True
                elapsed = round(time.monotonic() - start, 3)
                if not timed_out and (stderr.startswith("bwrap:") or proc.returncode in {-9, 137}
                                      or "MemoryError" in stderr):
                    raise InfrastructureError(f"Grading infrastructure failure in {label}")
                result = {"passed": not timed_out and proc.returncode == 0, "timeout": timed_out,
                          "duration_seconds": elapsed, "cpu": cpu, "returncode": proc.returncode,
                          "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-4000:]}
                atomic_json(directory / f"{step_id}.execution.{label}.json", result)
                env_results[label] = result["passed"]
                within_300[label] = result["passed"] and elapsed <= 300
    outcome = {"step_id": step_id, "passed": any(env_results.values()), "per_env": env_results}
    if config["grading"]["timeout_seconds"] == 1800:
        outcome["passed_300_derived"] = any(within_300.values())
    atomic_json(directory / f"{step_id}.score.json", outcome)
    return outcome


def run_problem(config: dict[str, Any], upstream: Any, problem: dict[str, Any], round_no: int,
                grader: concurrent.futures.ThreadPoolExecutor, stop: threading.Event) -> dict[str, Any]:
    pid = str(problem["problem_id"])
    bg = config["benchmark"]["with_background"]
    model_dir = config["model"]["name"].replace("/", "__")
    directory = Path(config["output"]) / f"repeat_{round_no}" / f"cleaned_{model_dir}" / ("with_bg" if bg else "no_bg") / pid
    directory.mkdir(parents=True, exist_ok=True)
    previous, cumulative, futures, usages = [], problem["required_dependencies"], [], []
    resume_root = config.get("_resume_source")
    resume_model_dir = config.get("_resume_model", config["model"]["name"]).replace("/", "__")
    old_directory = (Path(resume_root) / f"repeat_{round_no}" / f"cleaned_{resume_model_dir}" /
                     ("with_bg" if bg else "no_bg") / pid) if resume_root else None
    for index, step in enumerate(problem["sub_steps"]):
        if stop.is_set():
            raise InfrastructureError("Evaluation stopped after an earlier error")
        verify_config(config)
        step_id = str(step["step_number"])
        if (pid, index) in upstream.SKIP:
            gold = (upstream.SCICODE_DATA / f"{step_id}.txt").read_text()
            code = re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", gold, flags=re.MULTILINE)
            atomic_text(directory / f"{step_id}.code.py", code)
        else:
            prompt = upstream.build_prompt(problem, index, previous, bg)
            old_raw = old_directory / f"{step_id}.raw.json" if old_directory else None
            raw = json.loads(old_raw.read_text()) if old_raw and old_raw.is_file() else None
            reusable = bool(raw and not raw.get("generation_error") and raw.get("extracted_code") and
                            raw.get("finish_reason") in ("stop", "end_turn"))
            if reusable:
                if (raw.get("step_id") != step_id or raw.get("model") not in config.get("_allowed_resume_models", [config["model"]["name"]]) or
                    raw.get("with_background") != bg or not generation_compatible(raw.get("generation"), config["generation"]) or
                    raw.get("prompt") != prompt or upstream.extract_code(raw["response"]) != raw["extracted_code"]):
                    raise InfrastructureError(f"Incompatible saved generation for step {step_id}")
                response = {"content": raw["response"], "reasoning_content": raw.get("reasoning_content"),
                            "finish_reason": raw["finish_reason"], "usage": raw.get("usage")}
            else:
                if old_raw and old_raw == directory / f"{step_id}.raw.json" and raw and raw.get("generation_error"):
                    atomic_json(directory / f"{step_id}.attempt-1.json", raw)
                try:
                    response = generate(config, prompt)
                except GenerationIncomplete as exc:
                    previous_raw = directory / f"{step_id}.raw.json"
                    if previous_raw.exists():
                        atomic_json(directory / f"{step_id}.attempt-1.json", json.loads(previous_raw.read_text()))
                    atomic_json(directory / f"{step_id}.raw.json", {
                        "step_id": step_id, "model": config["model"]["name"], "with_background": bg,
                        "generation": config["generation"], "prompt": prompt,
                        "response": exc.response.get("content"),
                        "reasoning_content": exc.response.get("reasoning_content"),
                        "finish_reason": exc.response.get("finish_reason"),
                        "usage": exc.response.get("usage"), "generation_error": str(exc),
                        "stream_diagnostics": exc.response.get("stream_diagnostics"),
                    })
                    record_failure(config, round_no, "generation", step_id, exc)
                    return {"problem_id": pid, "futures": futures, "usages": usages,
                            "incomplete_step": step_id, "error": str(exc)}
            code = upstream.extract_code(response["content"])
            if not code.strip():
                raise InfrastructureError(f"Code extraction yielded no code for step {step_id}")
            if not reusable:
                raw = {"step_id": step_id, "model": config["model"]["name"], "with_background": bg,
                       "generation": config["generation"],
                       "prompt": prompt, "response": response["content"],
                       "reasoning_content": response["reasoning_content"],
                       "finish_reason": response["finish_reason"], "usage": response["usage"], "extracted_code": code}
            atomic_json(directory / f"{step_id}.raw.json", raw)
            atomic_text(directory / f"{step_id}.code.py", code)
            usages.append(response["usage"])
        previous.append(code)
        cumulative += "\n\n" + code
        if (pid, index) not in upstream.SKIP:
            old_score = old_directory / f"{step_id}.score.json" if old_directory else None
            if old_score and old_score.is_file():
                if not reusable:
                    raise InfrastructureError(f"Saved score lacks compatible generation for step {step_id}")
                if (old_directory / f"{step_id}.code.py").read_text() != code:
                    raise InfrastructureError(f"Saved code differs from generation for step {step_id}")
                saved = json.loads(old_score.read_text())
                envs = {label: json.loads((old_directory / f"{step_id}.execution.{label}.json").read_text())
                        for label in ("2024", "2025")}
                if (saved.get("step_id") != step_id or
                    saved.get("per_env") != {k: v["passed"] for k, v in envs.items()} or
                    saved.get("passed") != any(v["passed"] for v in envs.values())):
                    raise InfrastructureError(f"Inconsistent saved score for step {step_id}")
                for label, record in envs.items():
                    atomic_json(directory / f"{step_id}.execution.{label}.json", record)
                atomic_json(directory / f"{step_id}.score.json", saved)
                future = concurrent.futures.Future()
                future.set_result(saved)
            else:
                future = grader.submit(grade_step, config, cumulative, step["test_cases"], step_id, directory, stop)
            def signal_failure(f: concurrent.futures.Future, failed_step: str = step_id) -> None:
                if not f.cancelled() and f.exception() is not None:
                    record_failure(config, round_no, "grading", failed_step, f.exception())
                    stop.set()
            future.add_done_callback(signal_failure)
            futures.append(future)
    return {"problem_id": pid, "futures": futures, "usages": usages}


def run_round(config: dict[str, Any], upstream: Any, problems: list[dict[str, Any]],
              number: int, stop: threading.Event) -> dict[str, Any]:
    model_dir = config["model"]["name"].replace("/", "__")
    bg_dir = "with_bg" if config["benchmark"]["with_background"] else "no_bg"
    round_dir = Path(config["output"]) / f"repeat_{number}" / f"cleaned_{model_dir}" / bg_dir
    atomic_json(round_dir / "request-config.json", {"model": config["model"], "generation": config["generation"],
                                                     "benchmark": config["benchmark"], "config_digest": config["_config_digest"]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["grading"]["workers_per_round"]) as graders:
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["generation"]["workers_per_round"]) as generators:
            pending = {generators.submit(run_problem, config, upstream, p, number, graders, stop): str(p["problem_id"]) for p in problems}
            generated = []
            for future in concurrent.futures.as_completed(pending):
                try:
                    generated.append(future.result())
                except Exception as exc:
                    record_failure(config, number, "generation", pending[future], exc)
                    stop.set()
                    raise
        results = []
        for item in generated:
            steps = [future.result() for future in item["futures"]]
            results.append({"problem_id": item["problem_id"], "steps": steps, "n_steps_scored": len(steps),
                            "n_steps_ok": sum(step["passed"] for step in steps),
                            "problem_correct": bool(steps) and all(step["passed"] for step in steps),
                            "usages": item["usages"]})
    scored = sum(p["n_steps_scored"] for p in results)
    incomplete = [{"problem_id": x["problem_id"], "step": x["incomplete_step"], "error": x["error"]}
                  for x in generated if x.get("incomplete_step")]
    if incomplete:
        result = {"round": number, "status": "incomplete", "incomplete": incomplete,
                  "problems": len(results), "steps_total": scored,
                  "steps_correct": sum(p["n_steps_ok"] for p in results), "results": results}
        atomic_json(round_dir / "results.json", {"summary": {k: v for k, v in result.items() if k != "results"}, "results": results})
        return result
    expected = {str(p["problem_id"]): {str(step["step_number"])
                for i, step in enumerate(p["sub_steps"])
                if (str(p["problem_id"]), i) not in upstream.SKIP} for p in problems}
    actual = {p["problem_id"]: [str(step["step_id"]) for step in p["steps"]] for p in results}
    if (len(results) != len(problems) or not scored or set(actual) != set(expected)
            or any(len(actual[pid]) != len(expected[pid]) or set(actual[pid]) != expected[pid]
                   for pid in expected)):
        raise InfrastructureError("Round did not complete all problems and scored steps")
    result = {"round": number, "problems": len(results),
            "problems_correct": sum(p["problem_correct"] for p in results),
            "steps_total": scored, "steps_correct": sum(p["n_steps_ok"] for p in results),
            "results": results}
    atomic_json(round_dir / "results.json", {"summary": {k: v for k, v in result.items() if k != "results"}, "results": results})
    return result


def make_summary(config: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rounds)
    expected_rounds = config["benchmark"]["rounds"]
    if n != expected_rounds or {r["round"] for r in rounds} != set(range(1, expected_rounds + 1)):
        raise InfrastructureError("Cannot summarize: missing or duplicate rounds")
    official = not config["benchmark"].get("only")
    for r in rounds:
        if (r["problems"] != len(r["results"]) or not r["steps_total"]
                or sum(p["n_steps_scored"] for p in r["results"]) != r["steps_total"]
                or sum(p["n_steps_ok"] for p in r["results"]) != r["steps_correct"]
                or sum(bool(p["problem_correct"]) for p in r["results"]) != r["problems_correct"]
                or not 0 <= r["steps_correct"] <= r["steps_total"]
                or not 0 <= r["problems_correct"] <= r["problems"]):
            raise InfrastructureError("Cannot summarize: inconsistent round results")
        if official and (r["steps_total"] != 287 or r["problems"] != 64):
            raise InfrastructureError("Cannot summarize incomplete benchmark: expected 287 steps and 64 problems per round")
    sub = sum(r["steps_correct"] / r["steps_total"] for r in rounds) / n
    main = sum(r["problems_correct"] / r["problems"] for r in rounds) / n
    usages = [u for r in rounds for p in r["results"] for u in p["usages"] if isinstance(u, dict)]
    token_stats = None
    if usages:
        prompt = sum(int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
                     + int(u.get("cache_read_input_tokens", 0) or 0)
                     + int(u.get("cache_creation_input_tokens", 0) or 0) for u in usages)
        completion = sum(int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0) for u in usages)
        total = sum(int(u.get("total_tokens", 0) or 0) for u in usages) or prompt + completion
        token_stats = {"agent_prompt_tokens": prompt, "agent_completion_tokens": completion,
                       "total_tokens": total, "avg_tokens_per_sample": round(total / sum(r["problems"] for r in rounds), 1)}
        details = [(u.get("completion_tokens_details") or {}).get("reasoning_tokens") for u in usages]
        if any(value is not None for value in details):
            token_stats["reasoning_tokens"] = sum(int(value or 0) for value in details)
    trials = sum(r["problems"] for r in rounds)
    summary = {"status": "completed", "metrics": {"log_status": "success", "completed_samples": rounds[0]["problems"],
            "total_samples": rounds[0]["problems"], "total_trials": trials,
            "total_correct": sum(r["steps_correct"] for r in rounds),
            "total_steps": sum(r["steps_total"] for r in rounds),
            "sub_problem_correctness": round(100 * sub, 2), "sub_problem_correctness_raw": sub,
            "problem_correctness": round(100 * main, 2), "problem_correctness_raw": main},
            "token_stats": token_stats, "error_stats": {"total": trials, "failed": 0, "success_rate": 1.0,
                                                            "error_types": {}, "log_error": None},
            "details": {"dataset_version": "v2", "with_background": config["benchmark"]["with_background"],
                        "grading_policy": "2024_OR_2025", "official_full_coverage": official,
                        "continuation": config.get("continuation"), "usage_records": len(usages),
                        "generation_requests": sum(len(p["usages"]) for r in rounds for p in r["results"]),
                        "rounds": [{**{k: v for k, v in r.items() if k != "results"},
                                    "sub_problem_correctness": round(100 * r["steps_correct"] / r["steps_total"], 2),
                                    "problem_correctness": round(100 * r["problems_correct"] / r["problems"], 2)}
                                   for r in rounds]}}
    if config["grading"]["timeout_seconds"] == 1800:
        derived = []
        for r in rounds:
            count = sum(step.get("passed_300_derived", False) for p in r["results"] for step in p["steps"])
            problems = sum(all(step.get("passed_300_derived", False) for step in p["steps"]) for p in r["results"])
            derived.append({"round": r["round"], "steps_correct": count, "steps_total": r["steps_total"],
                            "problems_correct": problems, "problems": r["problems"]})
        summary["details"]["derived_300_seconds"] = {"source": "same_execution_records", "rounds": derived}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--resume-source")
    parser.add_argument("--allow-output-budget-increase", action="store_true")
    parser.add_argument("--allow-error-policy-change", action="store_true")
    args = parser.parse_args()
    path = Path(args.experiment)
    config = json.loads(path.read_text())
    old_preflight = None
    if args.resume_source:
        source = Path(args.resume_source).resolve()
        old_path = source / "experiment.json"
        if not old_path.is_file():
            old_path = source.parent / "experiment.json"
        old = json.loads(old_path.read_text())
        for key in ("source", "grading", "benchmark"):
            if old.get(key) != config.get(key):
                raise InfrastructureError(f"Resume configuration differs: {key}")
        old_model, new_model = old["model"], config["model"]
        if old_model != new_model:
            raise InfrastructureError("Resume model configuration differs")
        config["_resume_model"] = old_model["name"]
        if old.get("generation") != config.get("generation"):
            old_gen, new_gen = old["generation"], config["generation"]
            if not generation_compatible(old_gen, new_gen):
                raise InfrastructureError("Resume generation configuration differs")
            if old_gen["max_tokens"] != new_gen["max_tokens"] and not args.allow_output_budget_increase:
                raise InfrastructureError("Resume budget change not authorized")
            if old_gen.get("stop_on_api_error") != new_gen.get("stop_on_api_error") and not args.allow_error_policy_change:
                raise InfrastructureError("Resume error policy change not authorized")
            info = config.get("continuation") or {}
            if (info.get("source") != str(source) or info.get("old_max_tokens") != old["generation"]["max_tokens"]
                    or info.get("new_max_tokens") != config["generation"]["max_tokens"] or not info.get("reason")):
                raise InfrastructureError("Resume budget change provenance missing or inconsistent")
        if old.get("preflight", {}).get("config_digest") != config_digest(old):
            raise InfrastructureError("Original experiment digest does not match")
        old_preflight = old["preflight"]
        config["_resume_source"] = str(source)
    config["_experiment_path"] = str(path.resolve())
    config["_config_digest"] = config_digest(config)
    output = Path(config["output"])
    marker = output / "controller.started"
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    stop = threading.Event()
    try:
        upstream = load_upstream(Path(config["source"]))
        problems = preflight(config, upstream)
        if old_preflight:
            for key in ("manifest_version", "grading_env_versions", "source_hashes"):
                if old_preflight.get(key) != config["preflight"].get(key):
                    raise InfrastructureError(f"Resume environment or source differs: {key}")
        config["preflight"]["pipeline_sha256"] = sha256_file(Path(__file__))
        config["preflight"]["extraction_policy"] = "preserve_model_imports"
        config["preflight"]["config_digest"] = config["_config_digest"]
        atomic_json(path, {k: v for k, v in config.items() if not k.startswith("_")})
        verify_config(config)
        print(f"Verified v2: {len(problems)} problems; {config['benchmark']['rounds']} rounds", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(config["benchmark"]["rounds"], config.get("execution_policy", {}).get("max_active_rounds", config["benchmark"]["rounds"]))) as executor:
            futures = [executor.submit(run_round, config, upstream, problems, n, stop)
                       for n in range(1, config["benchmark"]["rounds"] + 1)]
            rounds = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    rounds.append(future.result())
                except Exception:
                    stop.set()
                    raise
        rounds.sort(key=lambda r: r["round"])
        if any(r.get("status") == "incomplete" for r in rounds):
            print("Initial pass complete; retrying incomplete problems once", flush=True)
            retry_config = {**config, "_resume_source": str(output), "_resume_model": config["model"]["name"]}
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(config["benchmark"]["rounds"], config.get("execution_policy", {}).get("max_active_rounds", config["benchmark"]["rounds"]))) as executor:
                retry_futures = [executor.submit(run_round, retry_config, upstream, problems, n, stop)
                                 for n in range(1, config["benchmark"]["rounds"] + 1)]
                rounds = [f.result() for f in concurrent.futures.as_completed(retry_futures)]
            rounds.sort(key=lambda r: r["round"])
        if any(r.get("status") == "incomplete" for r in rounds):
            incomplete = [{"round": r["round"], "failures": r.get("incomplete", [])} for r in rounds if r.get("status") == "incomplete"]
            atomic_json(output / "final-summary.json", {"status": "incomplete", "metrics": {},
                        "error_stats": {"incomplete_rounds": len(incomplete)}, "details": {"incomplete": incomplete}})
            print("Incomplete after one retry; no official score", flush=True)
            raise SystemExit(1)
        summary = make_summary(config, rounds)
        atomic_json(output / "final-summary.json", summary)
        print(json.dumps(summary["metrics"], ensure_ascii=False), flush=True)
    except Exception as exc:
        stop.set()
        atomic_json(output / "generation-stop.json", {"error_type": type(exc).__name__, "error_message": str(exc)})
        raise


if __name__ == "__main__":
    main()
