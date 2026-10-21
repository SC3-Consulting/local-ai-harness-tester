"""Find the largest stable context and prompt size for a model, then the fastest setting inside it.

Each probe is one real request through the existing runtime adapter, run with the harness's own
``execute_trial`` so timeouts, subprocess handling and response checks behave like a normal
benchmark. Probes are logged to ``probes.jsonl`` as they happen; nothing is held in SQLite, so a
killed process leaves no lock behind. See AUTO_TUNER.md for the algorithm and operating notes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import urlopen

import harness_tester
import runtime_adapters

LOG = logging.getLogger("auto_tuner")
HERE = Path(__file__).resolve().parent
RUNTIMES = ("ollama", "llama.cpp")
PROMPT_HEAD = (
    "Summarise the following excerpt from Moby-Dick in 2-3 sentences. "
    'Reply with ONLY a raw JSON object of the form {"summary": "<text>"} '
    "and no markdown fences or other text.\n\nEXCERPT:\n"
)
SERVER_ERROR = re.compile(r"HTTP 5\d\d|closed|reset|refused|terminated|killed|EOF", re.IGNORECASE)
REPLY_PREVIEW_CHARS = 160


class TunerError(Exception):
    """Base class for errors that end a tuning run."""


class PreflightError(TunerError):
    """The set-up is wrong (model, endpoint, corpus), so searching would be meaningless."""


class InfrastructureError(TunerError):
    """The server is gone and could not be brought back; this is not a capacity result."""


class BudgetExceeded(TunerError):
    """A safety limit (probes, time, memory, disk) stopped the run early."""


class ServerStartError(TunerError):
    """A managed server did not become ready; counts as a failed probe at that context."""


# --------------------------------------------------------------------------- search logic
# These functions know nothing about LLMs: they only call ``probe(value) -> bool``.

Probe = Callable[[int], bool]


def snap(value: float, granularity: int, up: bool = False) -> int:
    """Round to a multiple of ``granularity`` (never below one step)."""
    steps = math.ceil(value / granularity) if up else math.floor(value / granularity)
    return max(granularity, steps * granularity)


def bisect_boundary(
    probe: Probe, passed: int, failed: int, granularity: int, tolerance: float, label: str = ""
) -> tuple[int, int]:
    """Close the gap between a known pass and a known fail by testing the midpoint."""
    while True:
        gap = failed - passed
        if gap <= max(granularity, tolerance * passed):
            break
        mid = snap(passed + gap / 2, granularity)
        if mid <= passed or mid >= failed:
            break
        LOG.info("%s bracket pass=%d fail=%d gap=%d -> try %d", label, passed, failed, gap, mid)
        if probe(mid):
            passed = mid
        else:
            failed = mid
    LOG.info("%s settled: pass=%d, first fail=%d", label, passed, failed)
    return passed, failed


def search_down(
    probe: Probe, high: int, low: int, granularity: int, tolerance: float, label: str = ""
) -> tuple[int | None, int | None]:
    """Try ``high``; on failure halve until something passes, then bisect.

    Returns (largest pass, smallest fail). The pass is None if even ``low`` fails; the fail is
    None if ``high`` itself passes.
    """
    value, failed = high, None
    while True:
        LOG.info("%s try %d", label, value)
        if probe(value):
            if failed is None:
                return value, None
            return bisect_boundary(probe, value, failed, granularity, tolerance, label)
        failed = value
        if value <= low:
            return None, failed
        value = max(low, snap(value / 2, granularity))
        if value >= failed:
            value = low


def search_up(
    probe: Probe, start: int, limit: int, granularity: int, tolerance: float, label: str = ""
) -> tuple[int | None, int | None]:
    """Try ``start``, double while passing (capped at ``limit``), then bisect on the first fail."""
    value, passed = min(start, limit), None
    while True:
        LOG.info("%s try %d", label, value)
        if probe(value):
            passed = value
            if value >= limit:
                return passed, None
            value = min(limit, value * 2)
            continue
        if passed is None:
            return None, value
        return bisect_boundary(probe, passed, value, granularity, tolerance, label)


def sweep_contexts(low: int, high: int, points: int, granularity: int) -> list[int]:
    """Spread ``points`` context sizes geometrically between ``low`` and ``high`` inclusive."""
    low, high = min(low, high), max(low, high)
    if points <= 1 or low == high:
        return [high] if points <= 1 else [low]
    ratio = (high / low) ** (1 / (points - 1))
    values = {snap(low * ratio**i, granularity, up=True) for i in range(points)}
    values.update({low, high})
    return sorted(min(max(v, low), high) for v in values)


# ------------------------------------------------------------------------------ settings


@dataclass
class TunerSettings:
    runtime: str = "ollama"
    model: str = ""
    endpoint: str | None = None
    adapter: str = str(Path(runtime_adapters.__file__).resolve())
    server_cmd: str | None = None
    server_start_timeout: float = 180.0
    min_ctx: int = 2048
    max_ctx: int | None = None
    hard_ctx_cap: int = 65536
    ctx_strategy: str = "down"
    ctx_granularity: int = 256
    fixed_ctx: int | None = None
    skip_context_search: bool = False
    probe_words: int = 100
    probe_max_tokens: int = 16
    start_words: int = 500
    max_tokens: int = 200
    word_granularity: int = 50
    tokens_per_word: float = 1.5
    overhead_tokens: int = 64
    tolerance: float = 0.05
    buffer: float = 0.10
    confirm_repeats: int = 2
    sweep_points: int = 4
    sweep_repeats: int = 3
    min_score: float = 0.0
    timeout: float = 300.0
    max_probes: int = 80
    max_minutes: float = 240.0
    min_free_mem_mb: int = 300
    min_free_disk_mb: int = 1024
    cooldown: float = 5.0
    recovery_wait: float = 60.0
    seed: int = 42
    measure_ttft: bool = True
    json_mode: bool = False
    corpus: str = str(HERE / "examples" / "moby_dick.txt")
    output_dir: str = "tuning_runs"

    @property
    def probe_tokens(self) -> int:
        """Reply limit for context probes: a JSON check needs room for a complete reply."""
        return self.probe_max_tokens if self.min_score <= 0 else self.max_tokens

    def est_tokens(self, words: int) -> int:
        return math.ceil(words * self.tokens_per_word) + self.overhead_tokens

    def chunk_limit(self, ctx: int) -> int:
        """Most words that fit in ``ctx`` once the reply and prompt overhead are reserved."""
        return int((ctx - self.max_tokens - self.overhead_tokens) / self.tokens_per_word)

    def validate(self) -> None:
        if self.runtime not in RUNTIMES:
            raise ValueError(f"runtime must be one of {', '.join(RUNTIMES)}")
        if not self.model.strip():
            raise ValueError("a model is required")
        if self.ctx_strategy not in ("down", "up"):
            raise ValueError("ctx_strategy must be 'down' or 'up'")
        if not 0 <= self.buffer < 1 or not 0 < self.tolerance < 1:
            raise ValueError("buffer must be in [0, 1) and tolerance in (0, 1)")
        for name in ("ctx_granularity", "word_granularity", "probe_words", "start_words"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("confirm_repeats", "sweep_points", "sweep_repeats", "max_probes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.timeout <= 0 or self.tokens_per_word <= 0 or self.max_minutes <= 0:
            raise ValueError("timeout, tokens_per_word and max_minutes must be positive")
        if self.skip_context_search and not self.fixed_ctx:
            raise ValueError("--skip-context-search needs --ctx (the size your server uses)")
        if self.runtime == "llama.cpp" and not self.server_cmd and not self.skip_context_search:
            raise ValueError(
                "llama.cpp fixes its context when the server starts: give --server-cmd with a "
                "{ctx} placeholder, or start the server yourself and use --skip-context-search"
            )
        floor = self.est_tokens(self.probe_words) + self.probe_tokens
        if self.min_ctx < floor:
            raise ValueError(f"min_ctx must be at least {floor} for the probe prompt")
        if self.max_ctx is not None and self.max_ctx < self.min_ctx:
            raise ValueError("max_ctx must not be below min_ctx")


@dataclass
class ProbeResult:
    n: int
    phase: str
    tag: str
    ctx: int
    words: int
    est_tokens: int
    max_tokens: int
    ok: bool
    reason: str
    elapsed: float
    output_tokens: int | None
    score: float | None
    seed: int
    offset: int
    error: str
    at: str
    ttft: float | None = None
    decode_tps: float | None = None
    reply_preview: str = ""
    prompt_tokens: int | None = None
    finish_reason: str = ""


# ------------------------------------------------------------------------- system helpers


def available_memory_mb() -> int | None:
    """MemAvailable from /proc/meminfo, or None where that file does not exist."""
    try:
        with open("/proc/meminfo", encoding="ascii") as source:
            for line in source:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def classify(
    trial: dict[str, Any], min_score: float, max_tokens: int | None = None
) -> tuple[bool, str]:
    """Turn a harness trial into (passed, reason)."""
    if trial["status"] != "succeeded":
        error = trial.get("error") or ""
        if "TimeoutExpired" in error or "timed out" in error:
            return False, "timeout"
        return False, "server_error" if SERVER_ERROR.search(error) else "error"
    response = trial.get("response") or {}
    output = response.get("output")
    if not isinstance(output, str) or not output.strip():
        return False, "empty_output"
    if (response.get("score") or 0.0) < min_score:
        # JSON cut off by the reply limit says nothing about the context, so it is not a failure.
        reply_cap = (
            max_tokens is not None
            and response.get("finish_reason") == "length"
            and (response.get("output_tokens") or 0) >= max_tokens
        )
        if reply_cap:
            return True, "reply_cap"
        return False, "low_score"
    return True, "ok"


def _count_or_none(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def measurement(value: Any) -> float | None:
    """A non-negative finite number from an adapter response, else None."""
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def decode_rate(output_tokens: Any, decode_seconds: float | None) -> float | None:
    """Tokens per second after the first token: (n - 1) tokens over first-to-last chunk time."""
    if type(output_tokens) is not int or output_tokens < 2 or not decode_seconds:
        return None
    return (output_tokens - 1) / decode_seconds


def build_prompt(corpus: list[str], offset: int, words: int) -> str:
    return PROMPT_HEAD + " ".join(corpus[offset : offset + words])


def discover_max_ctx(settings: TunerSettings, endpoint: str) -> int | None:
    """Ask the server for the model's trained context length, if it says."""
    try:
        if settings.runtime == "ollama":
            info = runtime_adapters.context_defaults("ollama", settings.model, endpoint)
            return info["maximum"] if str(info.get("source", "")).startswith("Ollama") else None
        base = endpoint.rstrip("/")
        url = base + ("/models" if base.endswith("/v1") else "/v1/models")
        with urlopen(url, timeout=5) as response:
            value = json.load(response)["data"][0]["meta"]["n_ctx_train"]
        return value if type(value) is int and value > 0 else None
    except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError):
        return None


class ServerManager:
    """Start, restart and stop a llama.cpp (or other) server from a command template."""

    def __init__(
        self,
        template: str,
        endpoint: str,
        model: str,
        start_timeout: float,
        log_dir: Path,
        health: Callable[[], bool],
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.template, self.model, self.start_timeout = template, model, start_timeout
        self.log_dir, self.health, self.sleep = log_dir, health, sleep
        parts = urlsplit(endpoint)
        self.host, self.port = parts.hostname or "127.0.0.1", parts.port or 80
        self.ctx_bound = "{ctx}" in template
        self.proc: subprocess.Popen[bytes] | None = None
        self.ctx: int | None = None
        self.starts = 0

    def ensure(self, ctx: int) -> None:
        alive = self.proc is not None and self.proc.poll() is None
        if alive and (self.ctx == ctx or not self.ctx_bound) and self.health():
            return
        self.start(ctx)

    def start(self, ctx: int) -> None:
        self.stop()
        values = {
            "ctx": ctx,
            "port": self.port,
            "host": self.host,
            "model": shlex.quote(self.model),
        }
        command = shlex.split(self.template.format_map(values))
        self.starts += 1
        log_path = self.log_dir / f"server-{self.starts:02d}-ctx{ctx}.log"
        LOG.info("starting server (ctx=%d): %s  [log: %s]", ctx, shlex.join(command), log_path)
        with open(log_path, "wb") as log:
            try:
                # The server must outlive this call, so a context manager does not apply.
                self.proc = subprocess.Popen(  # pylint: disable=consider-using-with
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            except OSError as exc:
                raise ServerStartError(f"could not run {command[0]}: {exc}") from exc
        self.ctx = ctx
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            code = self.proc.poll()
            if code is not None:
                raise ServerStartError(f"server exited with {code}: {self._tail(log_path)}")
            if self.health():
                LOG.info("server ready (ctx=%d)", ctx)
                return
            self.sleep(1.0)
        self.stop()
        raise ServerStartError(f"server not ready after {self.start_timeout:.0f}s")

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        LOG.info("stopping server (pid %d)", proc.pid)
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        for _ in range(10):
            if not self.health():
                break
            self.sleep(1.0)

    @staticmethod
    def _tail(path: Path, size: int = 1200) -> str:
        try:
            return path.read_bytes()[-size:].decode("utf-8", errors="replace").strip()
        except OSError:
            return "(no log)"


# ------------------------------------------------------------------------------ the tuner


class Tuner:
    def __init__(
        self,
        settings: TunerSettings,
        corpus: list[str],
        out_dir: Path,
        executor: Callable[..., dict[str, Any]] = harness_tester.execute_trial,
        health: Callable[[], bool] | None = None,
        server: ServerManager | None = None,
        sleep: Callable[[float], None] = time.sleep,
        models: Callable[[], list[str]] | None = None,
    ):
        self.s, self.corpus, self.out_dir = settings, corpus, out_dir
        self.executor, self.server, self.sleep = executor, server, sleep
        self.endpoint = runtime_adapters.validate_endpoint(
            settings.endpoint or runtime_adapters.DEFAULT_ENDPOINTS[settings.runtime] or ""
        )
        self.health = health or self._default_health
        self.models = models or (
            lambda: runtime_adapters.list_models(settings.runtime, self.endpoint)
        )
        self.command = [
            sys.executable,
            settings.adapter,
            "--runtime",
            settings.runtime,
            "--endpoint",
            self.endpoint,
            "--timeout",
            str(settings.timeout),
        ]
        if settings.measure_ttft:
            self.command.append("--stream")
        self.rng = random.Random(settings.seed)
        self.used_offsets: set[int] = set()
        self.cache: dict[tuple[str, str, int, int, int], ProbeResult] = {}
        self.results: list[ProbeResult] = []
        self.executed = 0
        self.started = time.monotonic()
        self.probe_log = out_dir / "probes.jsonl"
        self.report: dict[str, Any] = {
            "status": "running",
            "runtime": settings.runtime,
            "model": settings.model,
            "endpoint": self.endpoint,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "settings": asdict(settings),
        }
        self._load_previous()

    # -- plumbing

    def _default_health(self) -> bool:
        try:
            runtime_adapters.list_models(self.s.runtime, self.endpoint, timeout=3.0)
            return True
        except (RuntimeError, ValueError, OSError):
            return False

    def _load_previous(self) -> None:
        if not self.probe_log.is_file():
            return
        with open(self.probe_log, encoding="utf-8") as source:
            for line in source:
                try:
                    result = ProbeResult(**json.loads(line))
                except (ValueError, TypeError):
                    continue  # a half-written last line from a killed run
                self.results.append(result)
                self.used_offsets.add(result.offset)
                key = (result.phase, result.tag, result.ctx, result.words, result.max_tokens)
                self.cache[key] = result
        if self.results:
            LOG.info("resumed %d earlier probes from %s", len(self.results), self.probe_log)

    def _check_budget(self) -> None:
        if self.executed >= self.s.max_probes:
            raise BudgetExceeded(f"reached --max-probes ({self.s.max_probes})")
        minutes = (time.monotonic() - self.started) / 60
        if minutes > self.s.max_minutes:
            raise BudgetExceeded(f"ran for {minutes:.0f} min, over --max-minutes")
        memory = available_memory_mb()
        if memory is not None and memory < self.s.min_free_mem_mb:
            raise BudgetExceeded(f"only {memory} MB RAM available (< --min-free-mem-mb)")
        free_mb = shutil.disk_usage(self.out_dir).free // (1024 * 1024)
        if free_mb < self.s.min_free_disk_mb:
            raise BudgetExceeded(f"only {free_mb} MB disk free (< --min-free-disk-mb)")

    def _ensure_server(self, ctx: int) -> None:
        if self.server is not None:
            self.server.ensure(ctx)
            return
        deadline = time.monotonic() + self.s.recovery_wait
        while not self.health():
            if time.monotonic() >= deadline:
                raise InfrastructureError(
                    f"{self.s.runtime} at {self.endpoint} is not answering; start it (or give "
                    "--server-cmd so the tuner can) and re-run with --resume"
                )
            LOG.warning("server not answering at %s; waiting", self.endpoint)
            self.sleep(2.0)

    def _pick_offset(self, words: int) -> int:
        top = len(self.corpus) - words
        if top < 0:
            raise PreflightError(f"corpus has only {len(self.corpus)} words; {words} needed")
        for _ in range(100):
            offset = self.rng.randrange(top + 1)
            if offset not in self.used_offsets:
                break
        self.used_offsets.add(offset)
        return offset

    def _record(self, result: ProbeResult) -> ProbeResult:
        self.results.append(result)
        self.cache[(result.phase, result.tag, result.ctx, result.words, result.max_tokens)] = result
        with open(self.probe_log, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(asdict(result), allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        verdict = "PASS" if result.ok else f"FAIL ({result.reason})"
        if result.ok and result.reason != "ok":
            verdict = f"PASS ({result.reason})"
        tokens = f"est {result.est_tokens} tok"
        if result.prompt_tokens is not None:
            tokens += f", actual {result.prompt_tokens}"
        speed = ""
        if result.ttft is not None:
            speed += f" ttft={result.ttft:.1f}s"
        if result.decode_tps is not None:
            speed += f" decode={result.decode_tps:.1f}tok/s"
        LOG.info(
            "probe #%02d [%s/%s] ctx=%d words=%d (%s) -> %s in %.1fs%s",
            result.n,
            result.phase,
            result.tag,
            result.ctx,
            result.words,
            tokens,
            verdict,
            result.elapsed,
            speed,
        )
        if result.error:
            LOG.debug("probe #%02d error: %s", result.n, result.error)
        return result

    def probe(self, phase: str, tag: str, ctx: int, words: int, max_tokens: int) -> ProbeResult:
        key = (phase, tag, ctx, words, max_tokens)
        if key in self.cache:
            LOG.info("probe [%s/%s] ctx=%d words=%d: reusing earlier result", *key[:4])
            return self.cache[key]
        self._check_budget()
        n = len(self.results) + 1
        seed = self.s.seed + n
        base: dict[str, Any] = {
            "n": n,
            "phase": phase,
            "tag": tag,
            "ctx": ctx,
            "words": words,
            "est_tokens": self.s.est_tokens(words),
            "max_tokens": max_tokens,
            "seed": seed,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._ensure_server(ctx)
        except ServerStartError as exc:
            return self._record(
                ProbeResult(
                    **base,
                    ok=False,
                    reason="server_start_failed",
                    elapsed=0.0,
                    output_tokens=None,
                    score=None,
                    offset=-1,
                    error=str(exc)[:400],
                )
            )
        offset = self._pick_offset(words)
        params: dict[str, Any] = {"temperature": 0, "max_tokens": max_tokens}
        if self.s.json_mode:
            params["response_format"] = "json"
        if self.s.runtime == "ollama":
            params["context_size"] = ctx
        request = {
            "protocol_version": 1,
            "model": self.s.model,
            "harness": "auto-tuner-v1",
            "parameters": params,
            "case": {
                "id": f"probe-{n}",
                "input": build_prompt(self.corpus, offset, words),
                "evaluation": {
                    "metric": "json",
                    "json_type": "object",
                    "required_keys": ["summary"],
                },
            },
            "repetition": 0,
            "seed": seed,
        }
        self.executed += 1
        trial = self.executor(self.command, request, str(HERE), self.s.timeout + 30)
        ok, reason = classify(trial, self.s.min_score, max_tokens)
        response = trial.get("response") or {}
        preview = str(response.get("output") or "")[:REPLY_PREVIEW_CHARS]
        error = (trial.get("error") or "")[:400]
        if reason == "low_score":
            error = f"score {response.get('score')} < {self.s.min_score}; reply starts: {preview!r}"
        ttft = measurement(response.get("ttft_seconds"))
        result = self._record(
            ProbeResult(
                **base,
                ok=ok,
                reason=reason,
                elapsed=float(trial["elapsed_seconds"]),
                output_tokens=response.get("output_tokens"),
                score=response.get("score"),
                offset=offset,
                error=error,
                ttft=ttft,
                decode_tps=decode_rate(
                    response.get("output_tokens"), measurement(response.get("decode_seconds"))
                ),
                reply_preview=preview,
                prompt_tokens=_count_or_none(response.get("prompt_tokens")),
                finish_reason=str(response.get("finish_reason") or ""),
            )
        )
        if not ok:
            if reason == "timeout" and self.server is not None:
                self.server.stop()  # a timed-out request may still be running server-side
            self.sleep(self.s.cooldown)
        return result

    # -- phases

    def preflight(self) -> int:
        """Check the set-up with one small probe; returns the largest context to search to."""
        s = self.s
        if not Path(s.adapter).is_file():
            raise PreflightError(f"adapter not found: {s.adapter}")
        if len(self.corpus) < s.start_words:
            raise PreflightError("corpus is too small; fetch it with examples/moby_dick_sweep.py")
        if self.server is None and not self.health():
            raise PreflightError(f"nothing is answering at {self.endpoint}; start the server first")
        smoke_ctx = s.fixed_ctx if s.skip_context_search else s.min_ctx
        assert smoke_ctx is not None
        if s.runtime == "ollama" and self.server is None:
            models = self.models()
            if s.model not in models:
                raise PreflightError(f"model {s.model!r} not found; server has: {models}")
        LOG.info("[preflight] smoke test at ctx=%d", smoke_ctx)
        smoke = self.probe("pre", "smoke", smoke_ctx, s.probe_words, s.probe_tokens)
        if not smoke.ok:
            raise PreflightError(
                f"the smoke test failed ({smoke.reason}): {smoke.error or 'no detail'}. "
                "Check the model name, endpoint and --timeout before tuning."
            )
        if s.skip_context_search:
            return smoke_ctx
        ceiling = s.max_ctx
        if ceiling is None:
            found = discover_max_ctx(s, self.endpoint)
            if found is None:
                raise PreflightError("could not find the model's maximum context; pass --max-ctx")
            ceiling = min(found, s.hard_ctx_cap)
            LOG.info("[preflight] model advertises %d; searching up to %d", found, ceiling)
        if ceiling < s.min_ctx:
            raise PreflightError(f"maximum context {ceiling} is below --min-ctx {s.min_ctx}")
        return ceiling

    def phase_context(self, upper: int) -> int | None:
        s = self.s
        if s.skip_context_search:
            LOG.info("[P1] skipped; using fixed context %d", upper)
            return upper

        def ok(ctx: int) -> bool:
            return self.probe("P1", "search", ctx, s.probe_words, s.probe_tokens).ok

        LOG.info("[P1] context search (%s), range %d..%d", s.ctx_strategy, s.min_ctx, upper)
        if s.ctx_strategy == "down":
            passed, failed = search_down(
                ok, upper, s.min_ctx, s.ctx_granularity, s.tolerance, "[P1]"
            )
        else:
            passed, failed = search_up(ok, s.min_ctx, upper, s.ctx_granularity, s.tolerance, "[P1]")
        self.report["phase1"] = {"max_stable_ctx": passed, "first_failing_ctx": failed}
        return passed

    def phase_chunk(self, ctx: int) -> tuple[int | None, int | None]:
        s = self.s
        limit = min(s.chunk_limit(ctx), len(self.corpus))
        if limit < s.word_granularity:
            raise PreflightError(f"context {ctx} is too small for any chunk after the reply")

        def ok(words: int) -> bool:
            return self.probe("P2", "search", ctx, words, s.max_tokens).ok

        LOG.info(
            "[P2] chunk search at ctx=%d: start %d words, limit %d words",
            ctx,
            s.start_words,
            limit,
        )
        passed, failed = search_up(
            ok, s.start_words, limit, s.word_granularity, s.tolerance, "[P2]"
        )
        self.report["phase2"] = {
            "ctx": ctx,
            "max_chunk_words": passed,
            "first_failing_words": failed,
            "limit_words": limit,
            "limited_by": "context size" if failed is None else "failure",
            "ttft_by_words": [
                {"words": r.words, "ttft_seconds": r.ttft}
                for r in sorted(self.results, key=lambda r: r.words)
                if r.phase == "P2" and r.tag == "search" and r.ok and r.ttft is not None
            ],
        }
        return passed, failed

    def phase_optimise(self, ceiling: int, max_chunk: int) -> dict[str, Any]:
        s = self.s
        safe = snap(max_chunk * (1 - s.buffer), s.word_granularity)
        confirmed = False
        for attempt in range(1, 4):
            LOG.info(
                "[P3] confirming %d words (%d runs at ctx=%d)", safe, s.confirm_repeats, ceiling
            )
            runs = [
                self.probe("P3", f"confirm{attempt}-{i}", ceiling, safe, s.max_tokens)
                for i in range(s.confirm_repeats)
            ]
            if all(r.ok for r in runs):
                confirmed = True
                break
            smaller = snap(safe * (1 - s.buffer), s.word_granularity)
            LOG.warning("[P3] %d words unstable; backing off to %d", safe, smaller)
            if smaller >= safe:
                break
            safe = smaller
        needed = min(ceiling, snap(s.est_tokens(safe) + s.max_tokens, s.ctx_granularity, up=True))
        candidates = (
            [ceiling]
            if s.skip_context_search
            else sweep_contexts(needed, ceiling, s.sweep_points, s.ctx_granularity)
        )
        LOG.info("[P3] latency sweep over contexts %s with %d words", candidates, safe)
        rows: list[dict[str, Any]] = []
        for ctx in candidates:
            self.probe("P3", "warmup", ctx, s.probe_words, s.probe_tokens)
            runs = [
                self.probe("P3", f"sweep{ctx}-{i}", ctx, safe, s.max_tokens)
                for i in range(s.sweep_repeats)
            ]
            if not all(r.ok for r in runs):
                LOG.warning("[P3] ctx=%d failed during the sweep; excluded", ctx)
                rows.append({"ctx": ctx, "stable": False})
                continue
            latency = statistics.median(r.elapsed for r in runs)
            ttfts = [r.ttft for r in runs if r.ttft is not None]
            decode = [r.decode_tps for r in runs if r.decode_tps is not None]
            end_to_end = [
                r.output_tokens / r.elapsed for r in runs if r.output_tokens and r.elapsed
            ]
            rows.append(
                {
                    "ctx": ctx,
                    "stable": True,
                    "median_seconds": latency,
                    "median_ttft_seconds": statistics.median(ttfts) if ttfts else None,
                    "median_decode_tokens_per_second": (
                        statistics.median(decode) if decode else None
                    ),
                    "median_end_to_end_tokens_per_second": (
                        statistics.median(end_to_end) if end_to_end else None
                    ),
                    "latency_min_seconds": min(r.elapsed for r in runs),
                    "latency_max_seconds": max(r.elapsed for r in runs),
                }
            )
            LOG.info(
                "[P3] ctx=%d median latency %.1fs (range %.1f-%.1fs)%s",
                ctx,
                latency,
                rows[-1]["latency_min_seconds"],
                rows[-1]["latency_max_seconds"],
                f", median ttft {statistics.median(ttfts):.1f}s" if ttfts else "",
            )
        stable = sorted(
            (r for r in rows if r["stable"]), key=lambda r: (r["median_seconds"], r["ctx"])
        )
        fastest = stable[0] if stable else None
        # Overlapping min-max ranges mean the runs cannot say which context is quicker.
        distinguishable = None
        if len(stable) > 1 and fastest is not None:
            distinguishable = fastest["latency_max_seconds"] < stable[1]["latency_min_seconds"]
        recommended = None
        if fastest is not None and distinguishable is not False:
            reason = "only one context tested" if distinguishable is None else "fastest"
            recommended = {"context_size": fastest["ctx"], "reason": reason}
        elif stable:
            LOG.warning("[P3] latency ranges overlap; speed does not separate the contexts")
            recommended = {
                "context_size": min(r["ctx"] for r in stable),
                "reason": "latency ranges overlap; smallest tested context that holds the prompt",
            }
        return {
            "confirmed_stable": confirmed,
            "buffer": s.buffer,
            "safe_chunk_words": safe,
            "safe_chunk_tokens_estimate": s.est_tokens(safe),
            "min_context_for_safe_chunk": needed,
            "usable_upper_limit": {"context_size": ceiling, "chunk_words": safe},
            "latency_sweep": rows,
            "fastest": fastest,
            "fastest_distinguishable": distinguishable,
            "recommended_context": recommended,
        }

    def write_report(self) -> None:
        self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.report["probe_count"] = len(self.results)
        self.report["reply_cap_probes"] = sum(r.reason == "reply_cap" for r in self.results)
        ratios = [
            r.prompt_tokens / r.est_tokens for r in self.results if r.prompt_tokens and r.est_tokens
        ]
        self.report["token_estimate"] = {
            "samples": len(ratios),
            "median_actual_over_estimated": statistics.median(ratios) if ratios else None,
        }
        self.report["probes"] = [asdict(r) for r in self.results]
        path = self.out_dir / "report.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.report, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)

    def tune(self) -> dict[str, Any]:
        """Run all phases. Always writes report.json, even when stopped early."""
        try:
            self.report["machine"] = harness_tester.machine_info()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Hardware probing must never block tuning.
            LOG.warning("could not read machine details: %s", exc)
        try:
            upper = self.preflight()
            ceiling = self.phase_context(upper)
            if ceiling is None:
                self.report["status"] = "no_stable_context"
                LOG.error("no context from %d up passed; try a smaller --min-ctx", self.s.min_ctx)
                return self.report
            max_chunk, _ = self.phase_chunk(ceiling)
            if max_chunk is None:
                self.report["status"] = "no_stable_chunk"
                LOG.error("even %d words failed at ctx=%d", self.s.start_words, ceiling)
                return self.report
            self.report["recommendation"] = self.phase_optimise(ceiling, max_chunk)
            self.report["status"] = "completed"
            return self.report
        except TunerError as exc:
            self.report["status"] = f"stopped: {type(exc).__name__}"
            self.report["reason"] = str(exc)
            raise
        except KeyboardInterrupt:
            self.report["status"] = "interrupted"
            raise
        finally:
            if self.server is not None:
                self.server.stop()
            self.write_report()


# ----------------------------------------------------------------------------------- CLI


def load_env_file(path: Path) -> int:
    """Load KEY=VALUE lines into os.environ without overriding values already set."""
    if not path.is_file():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value
            loaded += 1
    return loaded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=".env")
    known, _ = pre.parse_known_args(argv)
    load_env_file(Path(known.env_file))

    defaults = TunerSettings()

    def env(name: str, fallback: Any) -> Any:
        return os.environ.get(f"TUNER_{name.upper()}", fallback)

    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--env-file", default=".env", help="KEY=VALUE file of TUNER_* defaults (default: .env)")
    add("--runtime", choices=RUNTIMES, default=env("runtime", None), required=False)
    add("--model", default=env("model", None), help="model name (ollama) or model path/label")
    add("--endpoint", default=env("endpoint", None), help="server URL (default per runtime)")
    add(
        "--server-cmd",
        default=env("server_cmd", None),
        help="command to start the server; placeholders {ctx} {port} {host} {model}",
    )
    add("--server-start-timeout", type=float, default=env("server_start_timeout", 180.0))
    add("--min-ctx", type=int, default=env("min_ctx", defaults.min_ctx))
    add(
        "--max-ctx",
        type=int,
        default=env("max_ctx", None),
        help="largest context to try (default: what the model advertises, capped)",
    )
    add(
        "--hard-ctx-cap",
        type=int,
        default=env("hard_ctx_cap", defaults.hard_ctx_cap),
        help="cap applied only to an advertised maximum, to protect small machines",
    )
    add(
        "--ctx-strategy",
        choices=("down", "up"),
        default=env("ctx_strategy", "down"),
        help="down: start at the maximum and halve; up: start small and double",
    )
    add(
        "--ctx",
        dest="fixed_ctx",
        type=int,
        default=env("ctx", None),
        help="context your server already uses (with --skip-context-search)",
    )
    add(
        "--skip-context-search",
        action="store_true",
        default=str(env("skip_context_search", "")).lower() in ("1", "true", "yes"),
    )
    add("--ctx-granularity", type=int, default=env("ctx_granularity", defaults.ctx_granularity))
    add("--probe-words", type=int, default=env("probe_words", defaults.probe_words))
    add("--probe-max-tokens", type=int, default=env("probe_max_tokens", defaults.probe_max_tokens))
    add("--start-words", type=int, default=env("start_words", defaults.start_words))
    add(
        "--max-tokens",
        type=int,
        default=env("max_tokens", defaults.max_tokens),
        help="reply length reserved inside the context",
    )
    add("--word-granularity", type=int, default=env("word_granularity", defaults.word_granularity))
    add("--tokens-per-word", type=float, default=env("tokens_per_word", defaults.tokens_per_word))
    add("--overhead-tokens", type=int, default=env("overhead_tokens", defaults.overhead_tokens))
    add(
        "--tolerance",
        type=float,
        default=env("tolerance", defaults.tolerance),
        help="stop narrowing when the gap is under this fraction of the passing value",
    )
    add("--buffer", type=float, default=env("buffer", defaults.buffer))
    add("--confirm-repeats", type=int, default=env("confirm_repeats", defaults.confirm_repeats))
    add("--sweep-points", type=int, default=env("sweep_points", defaults.sweep_points))
    add("--sweep-repeats", type=int, default=env("sweep_repeats", defaults.sweep_repeats))
    add(
        "--min-score",
        type=float,
        default=env("min_score", defaults.min_score),
        help="0 = a probe passes if it completes; 1 = reply must be valid JSON",
    )
    add("--timeout", type=float, default=env("timeout", defaults.timeout), help="seconds per probe")
    add("--max-probes", type=int, default=env("max_probes", defaults.max_probes))
    add("--max-minutes", type=float, default=env("max_minutes", defaults.max_minutes))
    add("--min-free-mem-mb", type=int, default=env("min_free_mem_mb", defaults.min_free_mem_mb))
    add("--min-free-disk-mb", type=int, default=env("min_free_disk_mb", defaults.min_free_disk_mb))
    add("--cooldown", type=float, default=env("cooldown", defaults.cooldown))
    add("--recovery-wait", type=float, default=env("recovery_wait", defaults.recovery_wait))
    add("--seed", type=int, default=env("seed", defaults.seed))
    add(
        "--json-mode",
        action="store_true",
        default=str(env("json_mode", "")).lower() in ("1", "true", "yes"),
        help="ask the server to constrain replies to well-formed JSON",
    )
    add(
        "--ttft",
        dest="measure_ttft",
        action=argparse.BooleanOptionalAction,
        default=str(env("ttft", "1")).lower() in ("1", "true", "yes"),
        help="stream replies to record time to first token and decode speed (default: on)",
    )
    add("--corpus", default=env("corpus", defaults.corpus))
    add(
        "--output-dir",
        default=env("output_dir", defaults.output_dir),
        help="parent folder for run folders",
    )
    add("--resume", help="continue an earlier run folder (reuses its probes.jsonl)")
    add(
        "--log-level",
        default=env("log_level", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> TunerSettings:
    names = {field.name for field in fields(TunerSettings)}
    values = {k: v for k, v in vars(args).items() if k in names and v is not None}
    settings = TunerSettings(**values)
    settings.validate()
    return settings


def setup_logging(level: str, out_dir: Path) -> None:
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    detail = logging.FileHandler(out_dir / "tuner.log", encoding="utf-8")
    detail.setLevel(logging.DEBUG)
    for handler in (console, detail):
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
        )
        LOG.addHandler(handler)


def run_folder(settings: TunerSettings, resume: str | None) -> Path:
    if resume:
        folder = Path(resume)
        if not folder.is_dir():
            raise ValueError(f"cannot resume: {folder} is not a folder")
        return folder
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(settings.model).name or settings.model)[:60]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(settings.output_dir) / f"{settings.runtime}-{slug}-{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _raise_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    server: ServerManager | None = None
    try:
        args = parse_args(argv)
        if not args.runtime or not args.model:
            print(
                "Error: --runtime and --model are required (or TUNER_RUNTIME / TUNER_MODEL)",
                file=sys.stderr,
            )
            return 1
        settings = settings_from_args(args)
        corpus_path = Path(settings.corpus)
        if not corpus_path.is_file():
            raise ValueError(
                f"corpus not found: {corpus_path} (run: python3 examples/moby_dick_sweep.py fetch)"
            )
        corpus = corpus_path.read_text(encoding="utf-8").split()
        out_dir = run_folder(settings, args.resume)
        setup_logging(args.log_level, out_dir)
        signal.signal(signal.SIGTERM, _raise_interrupt)
        LOG.info("run folder: %s", out_dir)
        tuner = Tuner(settings, corpus, out_dir)
        if settings.server_cmd:
            if tuner.health():
                raise ValueError(
                    f"something already answers at {tuner.endpoint}; stop it so the tuner can "
                    "start the server with the right context"
                )
            server = ServerManager(
                settings.server_cmd,
                tuner.endpoint,
                settings.model,
                settings.server_start_timeout,
                out_dir,
                tuner.health,
            )
            tuner.server = server
        report = tuner.tune()
    except (TunerError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; probes so far are in the run folder.", file=sys.stderr)
        return 130
    summary = {k: v for k, v in report.items() if k not in ("probes", "settings", "machine")}
    summary["run_folder"] = str(out_dir)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
