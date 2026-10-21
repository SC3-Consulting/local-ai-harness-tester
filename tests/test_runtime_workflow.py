import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import harness_tester as ht
from evaluations import basic_suite


class FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/chat":
            prompt = request["messages"][0]["content"]
            output = (
                "[1,2,3]" if "array" in prompt else '{"answer":4}' if "object" in prompt else "4"
            )
            response = {"message": {"content": output}, "eval_count": 4}
        elif self.path == "/api/embed":
            response = {"embeddings": [[1, 0] for _ in request["input"]]}
        else:
            self.send_error(404)
            return
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class RuntimeWorkflowTests(unittest.TestCase):
    def test_command_http_scoring_persistence_comparison_recommendation(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as directory:
            with ht.connect(Path(directory) / "results.sqlite3") as connection:
                config = {
                    "model": "mock-model",
                    "harness": "mock-ollama",
                    "command": [
                        sys.executable,
                        str(Path(ht.__file__).parent / "runtime_adapters.py"),
                        "--runtime",
                        "ollama",
                        "--endpoint",
                        f"http://127.0.0.1:{server.server_port}",
                    ],
                    "cases": basic_suite(),
                    "parameter_grid": {"temperature": [0, 0.5]},
                    "repetitions": 2,
                }
                with contextlib.redirect_stderr(io.StringIO()):
                    first = ht.run_benchmark(connection, config, directory)
                    second = ht.run_benchmark(connection, config, directory)
                result = ht.summarise(connection, first)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(len(result["configurations"]), 2)
                self.assertEqual(result["configurations"][0]["mean_score"], 1)
                self.assertEqual(result["configurations"][0]["generation_metrics"]["valid_json"], 1)
                self.assertTrue(ht.compare(connection, [first, second])["comparable"])
                self.assertEqual(ht.recommend(connection, second)["recommended"]["mean_score"], 1)
                self.assertEqual(
                    connection.execute("SELECT count(*) FROM trials").fetchone()[0], 24
                )
