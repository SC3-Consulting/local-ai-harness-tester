import contextlib
import http.server
import io
import json
import os
import shlex
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import auto_tuner as at


def trial(ok=True, error=None, output="{}", elapsed=1.0, score=0.0):
    if not ok:
        return {"status": "failed", "elapsed_seconds": elapsed, "response": None, "error": error}
    response = {"output": output, "score": score, "output_tokens": 10}
    return {"status": "succeeded", "elapsed_seconds": elapsed, "response": response, "error": None}


class SearchTests(unittest.TestCase):
    def run_down(self, limit, high=32768, low=2048):
        calls = []

        def probe(value):
            calls.append(value)
            return value <= limit

        return at.search_down(probe, high, low, 256, 0.05), calls

    def test_search_down_halves_then_closes_in_on_the_limit(self):
        (passed, failed), calls = self.run_down(10000)
        self.assertEqual(calls[:3], [32768, 16384, 8192])
        self.assertLessEqual(passed, 10000)
        self.assertGreater(failed, 10000)
        self.assertLessEqual(failed - passed, 512)

    def test_search_down_edges(self):
        self.assertEqual(self.run_down(99999)[0], (32768, None))
        self.assertEqual(self.run_down(100)[0][0], None)
        self.assertEqual(self.run_down(100)[1][-1], 2048)

    def test_search_down_does_not_loop_when_granularity_exceeds_gap(self):
        self.assertEqual(at.search_down(lambda v: False, 300, 128, 256, 0.05), (None, 128))

    def test_search_up_doubles_to_the_limit_or_the_first_failure(self):
        calls = []

        def probe(value):
            calls.append(value)
            return value <= 1700

        passed, failed = at.search_up(probe, 500, 8000, 50, 0.05)
        self.assertEqual(calls[:3], [500, 1000, 2000])
        self.assertLessEqual(passed, 1700)
        self.assertGreater(failed, 1700)
        self.assertEqual(at.search_up(lambda v: True, 500, 1500, 50, 0.05), (1500, None))
        self.assertEqual(at.search_up(lambda v: False, 500, 1500, 50, 0.05), (None, 500))

    def test_sweep_contexts_are_sorted_unique_and_keep_both_ends(self):
        values = at.sweep_contexts(2304, 16384, 4, 256)
        self.assertEqual(values, sorted(set(values)))
        self.assertEqual((values[0], values[-1]), (2304, 16384))
        self.assertEqual(at.sweep_contexts(4096, 4096, 4, 256), [4096])


class ClassifyTests(unittest.TestCase):
    def test_reasons(self):
        cases = [
            (trial(), (True, "ok")),
            (trial(output="  "), (False, "empty_output")),
            (trial(ok=False, error="TimeoutExpired: x"), (False, "timeout")),
            (
                trial(ok=False, error="Harness exited 2: Request to u failed: timed out"),
                (False, "timeout"),
            ),
            (trial(ok=False, error="HTTP 500 from x: runner terminated"), (False, "server_error")),
            (trial(ok=False, error="ValueError: bad"), (False, "error")),
        ]
        for value, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(at.classify(value, 0.0), expected)
        self.assertEqual(at.classify(trial(score=0.0), 1.0), (False, "low_score"))

    def test_a_reply_stopped_by_the_reply_limit_is_not_a_capacity_failure(self):
        def reply(**extra):
            return trial(score=0.0) | {"response": {"output": "{", "score": 0.0, **extra}}

        capped = reply(finish_reason="length", output_tokens=200)
        self.assertEqual(at.classify(capped, 1.0, 200), (True, "reply_cap"))
        early = reply(finish_reason="length", output_tokens=90)
        self.assertEqual(at.classify(early, 1.0, 200), (False, "low_score"))
        self.assertEqual(at.classify(reply(output_tokens=200), 1.0, 200), (False, "low_score"))
        self.assertEqual(at.classify(capped, 1.0), (False, "low_score"))


class TunerTests(unittest.TestCase):
    CTX_LIMIT = 12000  # fake server cannot load a larger context
    WORD_LIMIT = 3000  # fake server runs out of room for a longer prompt

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.corpus = [f"w{i}" for i in range(20000)]
        self.prompts = []
        self.healthy = True
        self.settings = at.TunerSettings(
            model="m",
            max_ctx=32768,
            confirm_repeats=1,
            sweep_points=3,
            sweep_repeats=1,
            cooldown=0,
            recovery_wait=0,
            min_free_mem_mb=0,
            min_free_disk_mb=0,
            adapter=__file__,
        )

    def executor(self, command, request, cwd, timeout):
        ctx = request["parameters"]["context_size"]
        prompt = request["case"]["input"]
        self.prompts.append(prompt)
        words = len(prompt.split()) - len(at.PROMPT_HEAD.split())
        if ctx > self.CTX_LIMIT:
            return trial(ok=False, error="HTTP 500: out of memory")
        if words > self.WORD_LIMIT:
            return trial(ok=False, error="TimeoutExpired: too slow")
        return trial(elapsed=1.0 + ctx / 100000)

    def tuner(self, executor=None, settings=None):
        return at.Tuner(
            settings or self.settings,
            self.corpus,
            self.out,
            executor=executor or self.executor,
            health=lambda: self.healthy,
            models=lambda: ["m"],
            sleep=lambda seconds: None,
        )

    def test_full_run_finds_limits_buffers_them_and_never_repeats_text(self):
        report = self.tuner().tune()
        self.assertEqual(report["status"], "completed")
        ceiling = report["phase1"]["max_stable_ctx"]
        self.assertTrue(10000 < ceiling <= self.CTX_LIMIT)
        chunk = report["phase2"]["max_chunk_words"]
        self.assertTrue(2500 < chunk <= self.WORD_LIMIT)
        rec = report["recommendation"]
        self.assertLess(rec["safe_chunk_words"], chunk)
        self.assertEqual(rec["fastest"]["ctx"], min(r["ctx"] for r in rec["latency_sweep"]))
        self.assertEqual(len(self.prompts), len(set(self.prompts)))
        self.assertTrue((self.out / "report.json").is_file())

    def test_server_down_is_an_infrastructure_error_not_a_capacity_result(self):
        self.healthy = False
        tuner = self.tuner()
        with self.assertRaises(at.PreflightError):
            tuner.tune()
        self.assertEqual(json.loads((self.out / "report.json").read_text())["probe_count"], 0)

    def test_preflight_failure_reports_the_adapter_error(self):
        tuner = self.tuner(lambda *args: trial(ok=False, error="HTTP 404: model not found"))
        with self.assertRaisesRegex(at.PreflightError, "model not found"):
            tuner.tune()

    def test_probe_budget_stops_cleanly_and_keeps_the_report(self):
        self.settings.max_probes = 3
        tuner = self.tuner()
        with self.assertRaises(at.BudgetExceeded):
            tuner.tune()
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["probe_count"], 3)
        self.assertTrue(report["status"].startswith("stopped"))

    def test_resume_reuses_logged_probes(self):
        self.settings.max_probes = 4
        with self.assertRaises(at.BudgetExceeded):
            self.tuner().tune()
        self.settings.max_probes = 80
        seen = []
        self.tuner(lambda *args: seen.append(args) or self.executor(*args)).tune()
        self.assertGreater(len(seen), 0)
        lines = (self.out / "probes.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), len({json.loads(line)["n"] for line in lines}))
        self.assertEqual(json.loads(lines[0])["n"], 1)

    def test_up_strategy_gives_the_same_ceiling_within_tolerance(self):
        down = self.tuner().tune()["phase1"]["max_stable_ctx"]
        self.settings.ctx_strategy = "up"
        with tempfile.TemporaryDirectory() as other:
            self.out = Path(other)
            up = self.tuner().tune()["phase1"]["max_stable_ctx"]
        self.assertLessEqual(abs(up - down), 0.1 * down)

    def test_ttft_and_decode_speed_are_recorded_and_reported(self):
        def timed(command, request, cwd, timeout):
            reply = self.executor(command, request, cwd, timeout)
            if reply["status"] == "succeeded":
                reply["response"].update(ttft_seconds=0.5, decode_seconds=2.0, output_tokens=21)
            return reply

        tuner = self.tuner(timed)
        self.assertIn("--stream", tuner.command)
        report = tuner.tune()
        first = report["probes"][0]
        self.assertEqual((first["ttft"], first["decode_tps"]), (0.5, 10.0))
        self.assertTrue(report["phase2"]["ttft_by_words"])
        row = report["recommendation"]["latency_sweep"][0]
        self.assertEqual(row["median_ttft_seconds"], 0.5)
        self.assertEqual(row["median_decode_tokens_per_second"], 10.0)

    def test_ttft_can_be_switched_off_and_missing_timings_stay_empty(self):
        self.settings.measure_ttft = False
        tuner = self.tuner()
        self.assertNotIn("--stream", tuner.command)
        result = tuner.probe("pre", "smoke", 2048, 100, 16)
        self.assertEqual((result.ttft, result.decode_tps), (None, None))

    def test_json_check_gives_context_probes_room_for_a_full_reply(self):
        seen = []

        def spy(command, request, cwd, timeout):
            seen.append((request["parameters"]["max_tokens"], request["case"]["evaluation"]))
            return trial(score=1.0)

        self.tuner(spy).preflight()
        self.assertEqual(seen[0][0], self.settings.probe_max_tokens)
        self.settings.min_score = 1.0
        self.tuner(spy, settings=self.settings).preflight()
        self.assertEqual(seen[1][0], self.settings.max_tokens)
        self.assertEqual(seen[1][1]["required_keys"], ["summary"])

    def test_json_mode_is_requested_only_when_enabled(self):
        seen = []

        def spy(command, request, cwd, timeout):
            seen.append(request["parameters"].get("response_format"))
            return trial()

        self.tuner(spy).probe("pre", "smoke", 2048, 100, 16)
        self.settings.json_mode = True
        self.tuner(spy, settings=self.settings).probe("pre", "smoke2", 2048, 100, 16)
        self.assertEqual(seen, [None, "json"])

    def test_overlapping_latency_ranges_do_not_name_a_fastest_context(self):
        self.settings.sweep_repeats = 3
        times = iter([10.0, 14.0, 12.0] * 20)

        def noisy(command, request, cwd, timeout):
            reply = self.executor(command, request, cwd, timeout)
            if reply["status"] == "succeeded":
                reply["elapsed_seconds"] = next(times)
            return reply

        rec = self.tuner(noisy).tune()["recommendation"]
        self.assertFalse(rec["fastest_distinguishable"])
        smallest = min(r["ctx"] for r in rec["latency_sweep"] if r["stable"])
        self.assertEqual(rec["recommended_context"]["context_size"], smallest)
        self.assertIn("overlap", rec["recommended_context"]["reason"])

    def test_clearly_separated_latencies_name_the_fastest_context(self):
        rec = self.tuner().tune()["recommendation"]
        self.assertTrue(rec["fastest_distinguishable"])
        self.assertEqual(rec["recommended_context"]["reason"], "fastest")

    def test_low_score_failure_shows_the_start_of_the_reply(self):
        self.settings.min_score = 1.0
        tuner = self.tuner(lambda *args: trial(output="Sure! Here is", score=0.0))
        with self.assertRaisesRegex(at.PreflightError, "Sure! Here is"):
            tuner.tune()

    def test_replies_cut_off_by_the_reply_limit_do_not_fail_the_context(self):
        self.settings.min_score = 1.0
        calls = []

        def rambling(command, request, cwd, timeout):
            reply = self.executor(command, request, cwd, timeout)
            calls.append(1)
            if reply["status"] == "succeeded" and len(calls) % 3 == 0:
                limit = request["parameters"]["max_tokens"]
                reply["response"].update(score=0.0, finish_reason="length", output_tokens=limit)
            elif reply["status"] == "succeeded":
                reply["response"]["score"] = 1.0
            return reply

        report = self.tuner(rambling).tune()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(all(row["stable"] for row in report["recommendation"]["latency_sweep"]))
        self.assertGreater(report["reply_cap_probes"], 0)

    def test_actual_prompt_tokens_are_recorded_and_compared_with_the_estimate(self):
        def counted(command, request, cwd, timeout):
            reply = self.executor(command, request, cwd, timeout)
            if reply["status"] == "succeeded":
                reply["response"]["prompt_tokens"] = 428
            return reply

        tuner = self.tuner(counted)
        result = tuner.probe("pre", "smoke", 2048, 100, 16)
        self.assertEqual(result.prompt_tokens, 428)
        tuner.write_report()
        ratio = tuner.report["token_estimate"]["median_actual_over_estimated"]
        self.assertAlmostEqual(ratio, 428 / 214)

    def test_run_folder_name_uses_the_model_file_name_not_its_path(self):
        settings = at.TunerSettings(model="/data/models/q.gguf", output_dir=self.temp.name)
        folder = at.run_folder(settings, None)
        self.assertTrue(folder.name.startswith("ollama-q.gguf-"))
        self.assertNotIn("data", folder.name)

    def test_losing_the_server_mid_run_stops_cleanly_with_a_report(self):
        calls = []

        def dying(command, request, cwd, timeout):
            calls.append(1)
            if len(calls) == 3:
                self.healthy = False
                return trial(ok=False, error="Harness exited 2: Connection refused")
            return self.executor(command, request, cwd, timeout)

        with self.assertRaises(at.InfrastructureError):
            self.tuner(dying).tune()
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["probe_count"], 3)
        self.assertIn("InfrastructureError", report["status"])

    def test_settings_validation(self):
        bad = [
            {"runtime": "colibri"},
            {"model": " "},
            {"buffer": 1.0},
            {"runtime": "llama.cpp"},
            {"skip_context_search": True},
            {"min_ctx": 100},
        ]
        for change in bad:
            settings = at.TunerSettings(**{"model": "m", **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                settings.validate()
        at.TunerSettings(model="m").validate()


@unittest.skipIf(os.name == "nt", "the managed server uses POSIX process groups")
class ServerManagerTests(unittest.TestCase):
    """Starts and stops real child processes; no model or inference server is needed."""

    SERVER = (
        "import http.server, sys\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200)\n"
        "        self.end_headers()\n"
        "    def log_message(self, *args):\n"
        "        pass\n"
        "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n"
    )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.script = self.dir / "fake_server.py"
        self.script.write_text(self.SERVER)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.python = shlex.quote(sys.executable)

    def healthy(self):
        try:
            urlopen(f"http://127.0.0.1:{self.port}/", timeout=1).close()
            return True
        except OSError:
            return False

    def manager(self, template, start_timeout=20):
        manager = at.ServerManager(
            template,
            f"http://127.0.0.1:{self.port}",
            "model.gguf",
            start_timeout,
            self.dir,
            self.healthy,
            sleep=lambda seconds: time.sleep(0.1),
        )
        self.addCleanup(manager.stop)
        return manager

    def test_restarts_only_when_the_context_changes_and_stops_cleanly(self):
        manager = self.manager(f"{self.python} {self.script} {{port}} {{ctx}}")
        manager.ensure(2048)
        manager.ensure(2048)
        self.assertEqual((manager.starts, manager.ctx), (1, 2048))
        manager.ensure(4096)
        self.assertEqual((manager.starts, manager.ctx), (2, 4096))
        self.assertTrue(self.healthy())
        manager.stop()
        self.assertFalse(self.healthy())

    def test_a_command_without_a_ctx_placeholder_is_not_restarted_for_new_contexts(self):
        manager = self.manager(f"{self.python} {self.script} {{port}}")
        manager.ensure(2048)
        manager.ensure(4096)
        self.assertEqual(manager.starts, 1)

    def test_a_process_that_dies_is_restarted(self):
        manager = self.manager(f"{self.python} {self.script} {{port}}")
        manager.ensure(2048)
        manager.proc.kill()
        manager.proc.wait()
        manager.ensure(2048)
        self.assertEqual(manager.starts, 2)
        self.assertTrue(self.healthy())

    def test_start_failures_are_reported_with_their_cause(self):
        early = self.manager(f"{self.python} -c 'import sys; sys.exit(3)'")
        with self.assertRaisesRegex(at.ServerStartError, "exited with 3"):
            early.start(2048)
        slow = self.manager(f"{self.python} -c 'import time; time.sleep(30)'", start_timeout=1)
        with self.assertRaisesRegex(at.ServerStartError, "not ready"):
            slow.start(2048)
        self.assertIsNone(slow.proc)
        with self.assertRaisesRegex(at.ServerStartError, "could not run"):
            self.manager("/nonexistent/server-binary").start(2048)


class DiscoveryTests(unittest.TestCase):
    def serve(self, payload):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_llama_cpp_maximum_comes_from_the_models_endpoint(self):
        settings = at.TunerSettings(runtime="llama.cpp", model="m")
        endpoint = self.serve({"data": [{"id": "m", "meta": {"n_ctx_train": 32768}}]})
        self.assertEqual(at.discover_max_ctx(settings, endpoint), 32768)

    def test_missing_malformed_or_unreachable_metadata_gives_none(self):
        settings = at.TunerSettings(runtime="llama.cpp", model="m")
        for payload in ({"data": []}, {"data": [{"meta": {"n_ctx_train": "big"}}]}, {}):
            with self.subTest(payload=payload):
                self.assertIsNone(at.discover_max_ctx(settings, self.serve(payload)))
        self.assertIsNone(at.discover_max_ctx(settings, "http://127.0.0.1:1"))

    def test_ollama_maximum_is_used_only_when_ollama_advertised_it(self):
        settings = at.TunerSettings(runtime="ollama", model="m")
        advertised = {"maximum": 131072, "source": "Ollama model_info context_length"}
        fallback = {"maximum": 8192, "source": "fallback estimate"}
        with patch.object(at.runtime_adapters, "context_defaults", return_value=advertised):
            self.assertEqual(at.discover_max_ctx(settings, "http://127.0.0.1:1"), 131072)
        with patch.object(at.runtime_adapters, "context_defaults", return_value=fallback):
            self.assertIsNone(at.discover_max_ctx(settings, "http://127.0.0.1:1"))


class CliTests(unittest.TestCase):
    def test_main_runs_the_tuner_and_reports_the_exit_status(self):
        class StubTuner:
            status = "completed"

            def __init__(self, settings, corpus, out_dir):
                self.endpoint = "http://127.0.0.1:1"

            def health(self):
                return False

            def tune(self):
                return {"status": StubTuner.status, "probes": [], "settings": {}}

        def close_log_handlers():
            for handler in list(at.LOG.handlers):
                handler.close()
                at.LOG.removeHandler(handler)

        self.addCleanup(close_log_handlers)
        with tempfile.TemporaryDirectory() as temp:
            corpus = Path(temp) / "corpus.txt"
            corpus.write_text("word " * 50)
            args = ["--env-file", os.devnull, "--runtime", "ollama", "--model", "m"]
            args += ["--corpus", str(corpus)]
            for status, code in (("completed", 0), ("no_stable_context", 2)):
                StubTuner.status = status
                output = io.StringIO()
                with (
                    patch.object(at, "Tuner", StubTuner),
                    patch.object(at.signal, "signal"),
                    contextlib.redirect_stdout(output),
                ):
                    self.assertEqual(
                        at.main([*args, "--output-dir", str(Path(temp) / status)]), code
                    )
                self.assertEqual(json.loads(output.getvalue())["status"], status)
            close_log_handlers()

    def test_env_file_fills_defaults_without_overriding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".env"
            path.write_text(
                "# note\nTUNER_MODEL='llama3.2:1b'\nexport TUNER_TIMEOUT=45\nTUNER_SEED=9\n"
            )
            with patch.dict(os.environ, {"TUNER_SEED": "1"}, clear=False):
                os.environ.pop("TUNER_MODEL", None)
                os.environ.pop("TUNER_TIMEOUT", None)
                args = at.parse_args(["--env-file", str(path), "--runtime", "ollama"])
        self.assertEqual((args.model, args.timeout, args.seed), ("llama3.2:1b", 45.0, 1))

    def test_missing_model_is_a_clean_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(at.main(["--env-file", os.devnull, "--runtime", "ollama"]), 1)
        self.assertIn("--model", err.getvalue())


if __name__ == "__main__":
    unittest.main()
