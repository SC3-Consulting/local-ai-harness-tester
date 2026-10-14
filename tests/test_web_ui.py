import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import harness_tester as ht
from web_ui import BenchmarkServer


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server = BenchmarkServer(("127.0.0.1", 0), self.root / "web.sqlite3", self.root)
        machine = patch.object(ht, "machine_info", return_value={"machine": "web-test"})
        machine.start()
        self.addCleanup(machine.stop)
        worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def request(self, path, body=None, token=True, extra=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Harness-Token"] = self.server.token
        headers.update(extra or {})
        req = Request(
            self.url + path,
            headers=headers,
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urlopen(req, timeout=10) as response:
            return json.load(response)

    def test_page_and_authorization(self):
        with urlopen(self.url, timeout=10) as response:
            text = response.read().decode()
            self.assertIn("Run benchmark", text)
            self.assertIn(self.server.token, text)
            self.assertIn('id="model-select"', text)
            self.assertIn("Refresh model list", text)
            self.assertIn("/api/models?runtime=", text)
            self.assertIn('style="white-space:pre-line;overflow-wrap:anywhere"', text)
            self.assertIn('runtime+": "+state.reason).join("\\n")', text)
            self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        with self.assertRaises(HTTPError) as error:
            self.request("/api/runs", token=False)
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error:
            self.request("/api/runs", extra={"Host": "attacker.invalid"})
        self.assertEqual(error.exception.code, 403)
        self.assertEqual(self.request("/api/runs"), [])
        self.assertEqual(len(self.request("/api/suite")), 3)

    def test_cross_origin_post_blocked(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/api/jobs", {}, extra={"Origin": "http://attacker.invalid"})
        self.assertEqual(error.exception.code, 403)

    def test_invalid_request_and_unknown_routes(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/api/jobs", {"config": {}, "runtime": "unknown"})
        self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError) as error:
            self.request("/api/missing")
        self.assertEqual(error.exception.code, 404)

    def test_job_persistence_show_compare_recommend(self):
        config = {
            "model": "test-model",
            "harness": "test-harness",
            "cases": [{"id": "a", "input": "hi"}],
            "repetitions": 1,
        }

        def benchmark(connection, prepared, base_dir, progress=None):
            with patch.object(
                ht,
                "execute_trial",
                return_value={
                    "status": "succeeded",
                    "elapsed_seconds": 1,
                    "response": {"score": 1},
                    "error": None,
                },
            ):
                return original(connection, prepared, base_dir, progress=progress)

        original = ht.run_benchmark
        ids = []
        with patch.object(ht, "run_benchmark", side_effect=benchmark):
            for _ in range(2):
                job = self.request(
                    "/api/jobs", {"config": config, "runtime": "custom", "command": ["unused"]}
                )
                for _ in range(100):
                    state = self.request("/api/job?id=" + job["id"])
                    if state["status"] != "running":
                        break
                    time.sleep(0.01)
                self.assertEqual(state["status"], "completed")
                self.assertEqual(state["completed_trials"], 1)
                self.assertEqual(state["total_trials"], 1)
                ids.append(state["run_id"])
        self.assertEqual(len(self.request("/api/runs")), 2)
        result = self.request("/api/run?id=" + ids[0])
        self.assertEqual(result["config"]["model"], "test-model")
        self.assertEqual(len(result["trials"]), 1)
        self.assertTrue(self.request("/api/compare?id=" + ids[0] + "&id=" + ids[1])["comparable"])
        self.assertEqual(
            self.request("/api/recommend?id=" + ids[1])["recommended"]["mean_score"], 1
        )

    def test_busy_job_rejected_and_failure_exposed(self):
        self.server.jobs["busy"] = {"id": "busy", "status": "running"}
        body = {
            "config": {"model": "m", "harness": "h", "cases": [{"id": "a", "input": "x"}]},
            "runtime": "custom",
            "command": ["unused"],
        }
        with self.assertRaises(HTTPError) as error:
            self.request("/api/jobs", body)
        self.assertEqual(error.exception.code, 400)
        with patch.object(ht, "run_benchmark", side_effect=ValueError("explicit failure")):
            self.server.run_job("busy", body["config"], self.root)
        self.assertEqual(self.server.jobs["busy"]["status"], "failed")
        self.assertIn("explicit failure", self.server.jobs["busy"]["error"])

    def test_server_runtime_endpoint_and_absolute_adapter_command(self):
        config = {
            "model": "test-model",
            "harness": "test-harness",
            "cases": [{"id": "a", "input": "hi", "evaluation": {"metric": "json"}}],
            "cwd": ".",
        }
        for runtime, endpoint in (
            ("ollama", "http://127.0.0.1:11434"),
            ("llama.cpp", "http://127.0.0.1:8080"),
            ("colibri", "http://127.0.0.1:8000/v1"),
        ):
            with self.subTest(runtime=runtime), patch.object(threading.Thread, "start"):
                self.server.jobs.clear()
                self.server.start_job({"config": config, "runtime": runtime, "endpoint": endpoint})
                self.assertEqual(len(self.server.jobs), 1)
        with self.assertRaises(ValueError):
            self.server.start_job({"config": config, "runtime": "colibri", "endpoint": 123})

    def test_model_discovery_route_and_unsupported_runtimes(self):
        with patch(
            "runtime_adapters.list_models",
            return_value=["model-b", "model-a"],
        ) as discover:
            result = self.request(
                "/api/models?runtime=ollama&endpoint=http%3A%2F%2F127.0.0.1%3A11434"
            )
        self.assertEqual(result, {"runtime": "ollama", "models": ["model-b", "model-a"]})
        discover.assert_called_once_with("ollama", "http://127.0.0.1:11434")
        with self.assertRaises(HTTPError) as error:
            self.request("/api/models?runtime=airllm")
        self.assertEqual(error.exception.code, 400)

    def test_model_discovery_timeout_returns_json_error(self):
        message = "Request to http://127.0.0.1:8080/v1/models failed: timed out"
        with patch("runtime_adapters.list_models", side_effect=RuntimeError(message)):
            with self.assertRaises(HTTPError) as error:
                self.request("/api/models?runtime=llama.cpp")
        self.assertEqual(error.exception.code, 502)
        self.assertEqual(json.load(error.exception), {"error": message})
        self.assertEqual(self.request("/api/runs"), [])

    def test_latency_recommendation_query_and_invalid_threshold(self):
        with patch("harness_tester.recommend", return_value={"ranking": []}) as recommend:
            self.assertEqual(
                self.request("/api/recommend?id=reference&objective=latency&min_quality=0.8"),
                {"ranking": []},
            )
        self.assertEqual(recommend.call_args.args[1:], ("reference", "latency", 0.8))
        with self.assertRaises(HTTPError) as error:
            self.request("/api/recommend?id=reference&min_quality=invalid")
        self.assertEqual(error.exception.code, 400)

    def test_context_defaults_route(self):
        expected = {"supported": True, "maximum": 32768, "minimum": 2048, "step": 2048}
        with patch("runtime_adapters.context_defaults", return_value=expected) as defaults:
            result = self.request(
                "/api/context-defaults?runtime=ollama&model=test-model&endpoint=http://localhost:11434"
            )
        self.assertEqual(result, expected)
        defaults.assert_called_once_with("ollama", "test-model", "http://localhost:11434")

    def test_runtime_detection_route(self):
        def availability(runtime, endpoint):
            return {"available": runtime in ("ollama", "custom"), "reason": endpoint or "local"}

        with patch("runtime_adapters.runtime_availability", side_effect=availability) as detect:
            result = self.request("/api/runtimes?ollama=http://localhost:1234")
        self.assertTrue(result["ollama"]["available"])
        self.assertFalse(result["airllm"]["available"])
        self.assertTrue(result["custom"]["available"])
        self.assertEqual(result["ollama"]["reason"], "http://localhost:1234")
        self.assertEqual(detect.call_count, 5)

    def test_progress_visible_while_job_is_running(self):
        first_trial = threading.Event()
        release = threading.Event()
        config = {
            "model": "progress-model",
            "harness": "progress-harness",
            "cases": [{"id": "a", "input": "hi"}],
            "repetitions": 2,
        }
        original = ht.run_benchmark

        def benchmark(connection, prepared, base_dir, progress=None):
            def report(run_id, completed, total):
                progress(run_id, completed, total)
                if completed == 1:
                    first_trial.set()
                    if not release.wait(5):
                        raise TimeoutError("Test did not release the benchmark")

            with patch.object(
                ht,
                "execute_trial",
                return_value={
                    "status": "failed",
                    "elapsed_seconds": 1,
                    "response": None,
                    "error": "test failure",
                },
            ):
                return original(connection, prepared, base_dir, progress=report)

        with patch.object(ht, "run_benchmark", side_effect=benchmark):
            job = self.request(
                "/api/jobs", {"config": config, "runtime": "custom", "command": ["unused"]}
            )
            try:
                self.assertTrue(first_trial.wait(5))
                state = self.request("/api/job?id=" + job["id"])
                self.assertEqual(state["status"], "running")
                self.assertEqual(state["completed_trials"], 1)
                self.assertEqual(state["total_trials"], 2)
            finally:
                release.set()
            for _ in range(100):
                state = self.request("/api/job?id=" + job["id"])
                if state["status"] != "running":
                    break
                time.sleep(0.01)
            self.assertEqual(state["status"], "completed_with_failures")
            self.assertEqual(state["completed_trials"], 2)


if __name__ == "__main__":
    unittest.main()
