import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import harness_tester as ht


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = ht.connect(self.root / "results.sqlite3")
        self.addCleanup(self.db.close)
        self.config = {
            "model": "model-v1",
            "harness": "harness-v1",
            "command": [sys.executable, "-c", "pass"],
            "cases": [{"id": "a", "input": "hello"}, {"id": "b", "input": "world"}],
            "parameter_grid": {"quality": [0.5, 0.9], "delay": [1, 2]},
            "repetitions": 2,
        }
        machine = patch.object(ht, "machine_info", return_value={"machine": "test"})
        machine.start()
        self.addCleanup(machine.stop)

    def run_config(self, config=None, execute=None):
        def synthetic(command, request, cwd, timeout):
            return {
                "status": "succeeded",
                "elapsed_seconds": request["parameters"]["delay"],
                "response": {"score": request["parameters"]["quality"], "output_tokens": 10},
                "error": None,
            }

        with patch.object(ht, "execute_trial", side_effect=execute or synthetic):
            with contextlib.redirect_stderr(io.StringIO()):
                return ht.run_benchmark(self.db, config or self.config, self.root)

    def test_grid_repetitions_persistence_and_ranking(self):
        run_id = self.run_config()
        with sqlite3.connect(self.root / "results.sqlite3") as reopened:
            self.assertEqual(reopened.execute("SELECT count(*) FROM trials").fetchone()[0], 16)
        summary = ht.summarise(self.db, run_id)
        self.assertEqual(len(summary["configurations"]), 4)
        self.assertTrue(all(c["eligible"] for c in summary["configurations"]))
        best = ht.recommend(self.db, run_id)["recommended"]
        self.assertEqual(best["parameters"], {"quality": 0.9, "delay": 1})
        self.assertEqual(best["mean_score"], 0.9)
        self.assertEqual(best["mean_latency_seconds"], 1)
        self.assertEqual(best["tokens_per_second"], 10)
        self.assertEqual(best["successful_trials"], 4)

    def test_parameter_defaults_and_single_configuration(self):
        config = {**self.config, "parameters": {"quality": 0.7, "delay": 3}}
        config.pop("parameter_grid")
        run_id = self.run_config(config)
        self.assertEqual(
            ht.summarise(self.db, run_id)["configurations"][0]["parameters"], config["parameters"]
        )
        config["parameter_grid"] = {"quality": [0.8]}
        self.assertEqual(ht.parameter_sets(config), [{"quality": 0.8, "delay": 3}])

    def test_latency_recommendation_quality_threshold(self):
        def synthetic(command, request, cwd, timeout):
            quality = request["parameters"]["quality"]
            return {
                "status": "succeeded",
                "elapsed_seconds": 10 if quality == 0.9 else 1,
                "response": {"score": quality},
                "error": None,
            }

        run_id = self.run_config(execute=synthetic)
        self.assertEqual(ht.recommend(self.db, run_id)["recommended"]["mean_score"], 0.9)
        fastest = ht.recommend(self.db, run_id, "latency", 0.5)
        self.assertEqual(fastest["recommended"]["mean_score"], 0.5)
        self.assertEqual(fastest["recommended"]["mean_latency_seconds"], 1)
        filtered = ht.recommend(self.db, run_id, "latency", 0.9)
        self.assertEqual(filtered["recommended"]["mean_score"], 0.9)
        self.assertEqual(filtered["min_quality"], 0.9)
        with self.assertRaisesRegex(ValueError, "minimum quality"):
            ht.recommend(self.db, run_id, "latency", 1.0)
        for threshold in (-1, 2, float("nan"), float("inf"), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                ht.recommend(self.db, run_id, "latency", threshold)
        with self.assertRaises(ValueError):
            ht.recommend(self.db, run_id, "unknown")

    def test_failures_are_persisted_and_excluded(self):
        def execute(command, request, cwd, timeout):
            if request["parameters"]["quality"] == 0.9:
                return {
                    "status": "failed",
                    "elapsed_seconds": 1,
                    "response": None,
                    "error": "test failure",
                }
            return {
                "status": "succeeded",
                "elapsed_seconds": 2,
                "response": {"score": 0.5},
                "error": None,
            }

        run_id = self.run_config(execute=execute)
        summary = ht.summarise(self.db, run_id)
        self.assertEqual(summary["status"], "completed_with_failures")
        self.assertEqual(sum(c["failed_trials"] for c in summary["configurations"]), 8)
        best = ht.recommend(self.db, run_id)["recommended"]
        self.assertEqual(best["mean_score"], 0.5)
        self.assertIsNone(best["tokens_per_second"])

    def test_all_failures_do_not_produce_recommendation(self):
        run_id = self.run_config(
            execute=lambda *args: {
                "status": "failed",
                "elapsed_seconds": 1,
                "response": None,
                "error": "failed",
            }
        )
        with self.assertRaisesRegex(ValueError, "No complete"):
            ht.recommend(self.db, run_id)

    def test_progress_counts_persisted_trials_including_failures(self):
        updates = []

        def progress(run_id, completed, total):
            with sqlite3.connect(self.root / "results.sqlite3") as reader:
                persisted = reader.execute(
                    "SELECT count(*) FROM trials WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            self.assertEqual(persisted, completed)
            updates.append((completed, total))

        failure = {"status": "failed", "elapsed_seconds": 1, "response": None, "error": "failed"}
        with patch.object(ht, "execute_trial", return_value=failure):
            with contextlib.redirect_stderr(io.StringIO()):
                ht.run_benchmark(self.db, self.config, self.root, progress=progress)
        self.assertEqual(updates, [(completed, 16) for completed in range(17)])

    def test_interrupted_run_retains_trials_but_is_not_eligible(self):
        count = 0

        def interrupt(command, request, cwd, timeout):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt
            return {
                "status": "succeeded",
                "elapsed_seconds": 1,
                "response": {"score": 1},
                "error": None,
            }

        with self.assertRaises(KeyboardInterrupt):
            self.run_config(execute=interrupt)
        run_id = self.db.execute("SELECT id FROM runs").fetchone()[0]
        summary = ht.summarise(self.db, run_id)
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(sum(c["successful_trials"] for c in summary["configurations"]), 1)
        self.assertFalse(any(c["eligible"] for c in summary["configurations"]))

    def test_comparison_and_scope_isolation(self):
        first = self.run_config()
        second = self.run_config()
        self.assertTrue(ht.compare(self.db, [first, second])["comparable"])
        for field, value in (
            ("model", "different-model"),
            ("harness", "different-harness"),
            ("command", [sys.executable, "-c", "print('changed')"]),
            ("cases", [{"id": "different", "input": "new"}]),
            ("timeout_seconds", 1),
        ):
            with self.subTest(field=field):
                other = self.run_config({**self.config, field: value})
                self.assertFalse(ht.compare(self.db, [first, other])["comparable"])
        with patch.object(ht, "machine_info", return_value={"machine": "different"}):
            other = self.run_config()
            self.assertFalse(ht.compare(self.db, [first, other])["comparable"])
            with self.assertRaisesRegex(ValueError, "different machine"):
                ht.recommend(self.db, first)
        self.assertEqual(len(ht.recommend(self.db, first)["ranking"]), 4)

    def test_latest_measurement_not_best_historical_sample(self):
        first = self.run_config()

        def lower_quality(command, request, cwd, timeout):
            return {
                "status": "succeeded",
                "elapsed_seconds": 1,
                "response": {"score": 0.1},
                "error": None,
            }

        latest = self.run_config(execute=lower_quality)
        recommendation = ht.recommend(self.db, first)
        self.assertEqual(recommendation["recommended"]["run_id"], latest)
        self.assertEqual(recommendation["recommended"]["mean_score"], 0.1)

    def test_invalid_configs_fail_before_storage(self):
        for changes in (
            {"model": ""},
            {"command": "echo"},
            {"cases": []},
            {"cases": [{"id": "a"}, {"id": "a"}]},
            {"repetitions": 0},
            {"repetitions": True},
            {"parameter_grid": {"x": []}},
            {"parameter_grid": {"x": [1, 1]}},
            {"parameters": []},
            {"timeout_seconds": float("inf")},
            {"seed": None},
            {"cwd": 12},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_config({**self.config, **changes})
        self.assertEqual(self.db.execute("SELECT count(*) FROM runs").fetchone()[0], 0)

    def test_invalid_reference_ids(self):
        with self.assertRaisesRegex(ValueError, "Unknown run"):
            ht.summarise(self.db, "missing")
        run_id = self.run_config()
        with self.assertRaisesRegex(ValueError, "two distinct"):
            ht.compare(self.db, [run_id, run_id])

    def test_command_protocol_and_errors(self):
        command = [
            sys.executable,
            "-c",
            "import json,sys; r=json.load(sys.stdin); "
            "print(json.dumps({'score':r['parameters']['quality'],'output_tokens':2}))",
        ]
        result = ht.execute_trial(command, {"parameters": {"quality": 0.8}}, self.root, 5)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["response"]["score"], 0.8)
        self.assertGreater(result["elapsed_seconds"], 0)
        for command, message in (
            ([sys.executable, "-c", "print('invalid')"], "JSONDecodeError"),
            ([sys.executable, "-c", "print('{\"score\":2}')"], "between 0 and 1"),
            ([sys.executable, "-c", "print('{\"score\":NaN}')"], "Non-finite"),
            (
                [sys.executable, "-c", "import sys; sys.stderr.write('oops'); sys.exit(2)"],
                "exited 2: oops",
            ),
            ([str(self.root / "missing")], "FileNotFoundError"),
            ([sys.executable, "-c", "import time; time.sleep(1)"], "TimeoutExpired"),
        ):
            with self.subTest(command=command):
                trial = ht.execute_trial(command, {}, self.root, 0.1)
                self.assertEqual(trial["status"], "failed")
                self.assertIn(message, trial["error"])

    def test_response_validation(self):
        for value in (
            [],
            {},
            {"score": True},
            {"score": -1},
            {"score": float("nan")},
            {"score": 0.8, "output_tokens": -1},
            {"score": 0.8, "output_tokens": 1.2},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ht.validate_response(value)

    def test_cli_end_to_end(self):
        config = {
            **self.config,
            "command": [sys.executable, "-c", "print('{\"score\":0.75}')"],
            "parameter_grid": {},
            "repetitions": 1,
        }
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config))
        args = ["--db", str(self.root / "cli.sqlite3")]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ht.main([*args, "run", str(config_path)]), 0)
        run_id = json.loads(output.getvalue())["run_id"]
        for action in ("list", "show", "recommend"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    ht.main([*args, action, *([] if action == "list" else [run_id])]), 0
                )
            self.assertIsNotNone(json.loads(output.getvalue()))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ht.main([*args, "show", "missing"]), 1)


if __name__ == "__main__":
    unittest.main()
