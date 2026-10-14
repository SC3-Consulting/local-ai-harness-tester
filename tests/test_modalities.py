import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import harness_tester as ht


class ModalityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = ht.connect(self.root / "results.sqlite3")
        self.addCleanup(self.db.close)
        self.config = {
            "model": "test-model",
            "harness": "test-harness",
            "command": [sys.executable, "-c", "print('{\"embedding\":[1,0]}')"],
            "cases": [
                {
                    "id": "reference",
                    "task": "embedding",
                    "input": "text",
                    "evaluation": {"metric": "cosine_similarity", "reference": [1, 0]},
                }
            ],
            "repetitions": 2,
        }

    def run_config(self, config=None):
        with contextlib.redirect_stderr(io.StringIO()):
            return ht.run_benchmark(self.db, config or self.config, self.root)

    def test_reference_similarity_and_throughput_are_persisted(self):
        run_id = self.run_config()
        summary = ht.summarise(self.db, run_id)
        self.assertEqual(summary["tasks"], ["embedding"])
        result = summary["configurations"][0]
        self.assertEqual(result["mean_score"], 1)
        self.assertEqual(result["embedding_metrics"], {"cosine_similarity": 1})
        self.assertAlmostEqual(result["embeddings_per_second"], 1 / result["mean_latency_seconds"])
        self.assertIsNone(result["tokens_per_second"])
        response = json.loads(self.db.execute("SELECT response FROM trials").fetchone()[0])
        self.assertEqual(response["embedding"], [1, 0])
        self.assertEqual(response["embedding_dimensions"], 2)
        self.assertTrue(ht.recommend(self.db, run_id)["recommended"]["eligible"])

    def test_cosine_normalization_and_extreme_finite_vectors(self):
        case = self.config["cases"][0]
        for vector, similarity, score in (
            ([10, 0], 1, 1),
            ([0, 1], 0, 0.5),
            ([-1, 0], -1, 0),
            ([1e308, 0], 1, 1),
            ([1e-308, 0], 1, 1),
        ):
            with self.subTest(vector=vector):
                response = ht.evaluate_embedding({"embedding": vector, "score": 0.123}, case)
                self.assertEqual(response["score"], score)
                self.assertEqual(response["embedding_metrics"]["cosine_similarity"], similarity)
        self.assertAlmostEqual(ht.cosine_similarity([1, 1], [1, 0]), 1 / math.sqrt(2))

    def retrieval_case(self):
        return {
            "id": "retrieval",
            "task": "embedding",
            "input": "query",
            "documents": [{"id": key, "input": key} for key in ("a", "b", "c")],
            "evaluation": {"metric": "retrieval", "relevant_ids": ["b", "c"], "k": 2},
        }

    def test_retrieval_metrics_and_deterministic_ties(self):
        case = self.retrieval_case()
        response = ht.evaluate_embedding(
            {"embedding": [1, 0], "document_embeddings": {"c": [-1, 0], "b": [1, 0], "a": [1, 0]}},
            case,
        )
        self.assertEqual(response["retrieval_ranking"], ["a", "b", "c"])
        self.assertEqual(response["score"], 0.5)
        self.assertEqual(
            response["embedding_metrics"], {"recall_at_k": 0.5, "reciprocal_rank": 0.5}
        )
        self.assertEqual(response["embedding_count"], 4)
        config = {
            **self.config,
            "cases": [case],
            "command": [sys.executable, "-c", f"print({json.dumps(response)!r})"],
        }
        run_id = self.run_config(config)
        result = ht.summarise(self.db, run_id)["configurations"][0]
        self.assertEqual(result["mean_score"], 0.5)
        self.assertAlmostEqual(result["embeddings_per_second"], 4 / result["mean_latency_seconds"])

    def test_invalid_embeddings_fail_trials_without_aborting_run(self):
        for value in (None, [], [0, 0], [True, 0], [1, 0, 0], [1e309, 0]):
            config = {
                **self.config,
                "command": [sys.executable, "-c", f"print({json.dumps({'embedding': value})!r})"],
            }
            with self.subTest(value=value):
                run_id = self.run_config(config)
                result = ht.summarise(self.db, run_id)["configurations"][0]
                self.assertEqual(result["failed_trials"], 2)
                self.assertFalse(result["eligible"])
                row = self.db.execute(
                    "SELECT response, error FROM trials WHERE run_id = ?", (run_id,)
                ).fetchone()
                self.assertIsNone(row["response"])
                self.assertTrue(row["error"])

    def test_invalid_retrieval_responses(self):
        case = self.retrieval_case()
        for documents in (
            None,
            [],
            {"a": [1, 0]},
            {"a": [1, 0], "b": [1], "c": [0, 1]},
            {"a": [1, 0], "b": [0, 0], "c": [0, 1]},
            {"a": [1, 0], "b": [0, 1], "c": [1, 1], "extra": [1, 0]},
        ):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                ht.evaluate_embedding({"embedding": [1, 0], "document_embeddings": documents}, case)

    def test_invalid_cases_fail_before_creating_run(self):
        base = self.config["cases"][0]
        cases = [
            {**base, "task": "unknown"},
            {**base, "evaluation": None},
            {**base, "evaluation": {"metric": "unknown"}},
            {**base, "evaluation": {"metric": "cosine_similarity", "reference": [0, 0]}},
            {**base, "documents": []},
            {**base, "input": {"parts": []}},
            {**base, "input": {"parts": [{"type": "image"}]}},
            {**base, "input": {"parts": [{"type": "text", "text": 1}]}},
            {**base, "input": {"parts": [{"type": "unsupported"}]}},
            {**base, "input": {"parts": [{"type": "audio", "path": "x", "mime_type": 1}]}},
            {**base, "task": "generation"},
        ]
        retrieval = self.retrieval_case()
        cases.extend(
            [
                {**retrieval, "documents": []},
                {**retrieval, "documents": [{"id": "a", "input": "a"}] * 2},
                {**retrieval, "evaluation": {"metric": "retrieval", "relevant_ids": ["missing"]}},
                {**retrieval, "evaluation": {"metric": "retrieval", "relevant_ids": ["a"], "k": 0}},
                {**retrieval, "evaluation": {"metric": "retrieval", "relevant_ids": ["a"], "k": 4}},
                {**retrieval, "evaluation": {"metric": "retrieval", "relevant_ids": ["a", "a"]}},
            ]
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.run_config({**self.config, "cases": [case]})
        self.assertEqual(self.db.execute("SELECT count(*) FROM runs").fetchone()[0], 0)

    def media_config(self):
        parts = [{"type": "text", "text": "Describe these inputs"}]
        for kind in ("image", "audio", "video"):
            (self.root / kind).write_bytes(f"synthetic {kind}".encode())
            parts.append({"type": kind, "path": kind})
        return {
            **self.config,
            "command": [
                sys.executable,
                "-c",
                "import json,sys; r=json.load(sys.stdin); "
                "assert r['protocol_version']==2; print(json.dumps({'score':1,'request':r}))",
            ],
            "cases": [{"id": "media", "input": {"parts": parts}}],
        }

    def test_media_paths_protocol_and_fingerprints(self):
        config = self.media_config()
        run_id = self.run_config(config)
        response = json.loads(self.db.execute("SELECT response FROM trials").fetchone()[0])
        parts = response["request"]["case"]["input"]["parts"]
        for part in parts[1:]:
            self.assertEqual(part["path"], str(self.root / part["type"]))
            self.assertEqual(part["sha256"], ht.file_digest(part["path"]))
        second = self.run_config(config)
        self.assertTrue(ht.compare(self.db, [run_id, second])["comparable"])
        (self.root / "image").write_bytes(b"changed")
        changed = self.run_config(config)
        self.assertFalse(ht.compare(self.db, [run_id, changed])["comparable"])
        self.assertEqual(config["cases"][0]["input"]["parts"][1]["path"], "image")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                ht.main(["--db", str(self.root / "results.sqlite3"), "show", run_id]), 0
            )
        stored_parts = json.loads(output.getvalue())["config"]["cases"][0]["input"]["parts"]
        self.assertEqual(stored_parts, parts)

    def test_missing_media_is_configuration_error(self):
        config = self.media_config()
        (self.root / "image").unlink()
        with self.assertRaisesRegex(ValueError, "Media file does not exist"):
            self.run_config(config)
        self.assertEqual(self.db.execute("SELECT count(*) FROM runs").fetchone()[0], 0)

    def test_media_changed_during_trial_is_failure(self):
        config = self.media_config()
        config["command"] = [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('image').write_bytes(b'changed'); "
            "print('{\"score\":1}')",
        ]
        run_id = self.run_config(config)
        summary = ht.summarise(self.db, run_id)
        self.assertEqual(summary["status"], "completed_with_failures")
        errors = [
            row[0]
            for row in self.db.execute("SELECT error FROM trials WHERE run_id = ?", (run_id,))
        ]
        self.assertTrue(all("Media file changed" in error for error in errors))

    def test_multimodal_embedding_documents_resolve_and_validate(self):
        case = self.retrieval_case()
        self.media_config()
        case["input"] = {"parts": [{"type": "image", "path": "image"}]}
        case["documents"][0]["input"] = {"parts": [{"type": "audio", "path": "audio"}]}
        ht.validate_config({**self.config, "cases": [case]})
        prepared = ht.prepare_cases([case], self.root)[0]
        ht.verify_media(prepared)
        self.assertEqual(
            prepared["documents"][0]["input"]["parts"][0]["path"], str(self.root / "audio")
        )
        (self.root / "audio").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Media file changed"):
            ht.verify_media(prepared)

    def test_embedding_sweep_recommends_measured_winner(self):
        examples = Path(ht.__file__).parent / "examples"
        config = ht.load_json(examples / "embedding_sweep.json")
        config["command"][0] = sys.executable
        with contextlib.redirect_stderr(io.StringIO()):
            run_id = ht.run_benchmark(self.db, config, examples)
        best = ht.recommend(self.db, run_id)["recommended"]
        self.assertEqual(best["parameters"], {"variant": "aligned"})
        self.assertEqual(best["mean_score"], 1)
        self.assertEqual(best["successful_trials"], 6)


if __name__ == "__main__":
    unittest.main()
