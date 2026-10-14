import base64
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import runtime_adapters as adapters


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *_args):
        return self.payload


def read_request(call):
    request = call.call_args.args[0]
    return request.full_url, json.loads(request.data)


class EndpointTests(unittest.TestCase):
    def test_runtime_availability_without_inference(self):
        with patch("runtime_adapters.list_models", return_value=[]):
            self.assertTrue(adapters.runtime_availability("ollama")["available"])
        with patch("runtime_adapters.list_models", side_effect=RuntimeError("offline")):
            result = adapters.runtime_availability("llama.cpp")
        self.assertFalse(result["available"])
        self.assertIn("offline", result["reason"])
        with patch("runtime_adapters.importlib.util.find_spec", return_value=None):
            result = adapters.runtime_availability("airllm")
        self.assertFalse(result["available"])
        self.assertIn("airllm, torch", result["reason"])
        with patch("runtime_adapters.importlib.util.find_spec", return_value=object()):
            self.assertTrue(adapters.runtime_availability("airllm")["available"])
        self.assertTrue(adapters.runtime_availability("custom")["available"])

    def test_context_defaults_use_advertised_ollama_limit(self):
        with patch(
            "runtime_adapters._post_json",
            return_value={"model_info": {"llama.context_length": 32768, "general.name": "test"}},
        ) as post:
            defaults = adapters.context_defaults("ollama", "test-model", "http://localhost:11434")
        self.assertTrue(defaults["supported"])
        self.assertEqual(defaults["maximum"], 32768)
        self.assertEqual(defaults["minimum"], 2048)
        self.assertEqual(defaults["step"], 2048)
        self.assertIn("Ollama", defaults["source"])
        post.assert_called_once_with(
            "http://localhost:11434/api/show", {"model": "test-model"}, 5.0
        )

    def test_context_defaults_fallback_small_models_and_server_fixed_context(self):
        with patch("runtime_adapters._post_json", side_effect=RuntimeError("offline")):
            defaults = adapters.context_defaults("ollama", "missing")
        self.assertEqual(defaults["maximum"], 8192)
        self.assertIn("offline", defaults["note"])
        self.assertEqual(defaults["source"], "fallback estimate")
        with patch(
            "runtime_adapters._post_json",
            return_value={"model_info": {"small.context_length": 1024}},
        ):
            defaults = adapters.context_defaults("ollama", "small")
        self.assertEqual(
            (defaults["minimum"], defaults["maximum"], defaults["step"]), (1024, 1024, 1024)
        )
        for runtime in ("llama.cpp", "colibri"):
            with self.subTest(runtime=runtime):
                self.assertFalse(adapters.context_defaults(runtime, "m")["supported"])
        with self.assertRaises(ValueError):
            adapters.context_defaults("unknown", "m")

    def test_airllm_context_defaults_read_local_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(
                json.dumps({"max_position_embeddings": 16384}), encoding="utf-8"
            )
            defaults = adapters.context_defaults("airllm", directory)
        self.assertEqual(defaults["maximum"], 16384)
        self.assertIn("truncation", defaults["note"])

    def test_model_discovery_for_ollama_and_openai_servers(self):
        responses = {
            "ollama": {"models": [{"name": "zeta"}, {"name": "alpha"}, {"name": "alpha"}]},
            "llama.cpp": {"data": [{"id": "zeta"}, {"id": "alpha"}]},
            "colibri": {"data": [{"id": "colibri-small"}]},
        }
        expected_paths = {
            "ollama": "http://127.0.0.1:11434/api/tags",
            "llama.cpp": "http://127.0.0.1:8080/v1/models",
            "colibri": "http://localhost:8000/v1/models",
        }
        endpoints = {"ollama": None, "llama.cpp": None, "colibri": "http://localhost:8000/v1"}
        for runtime, payload in responses.items():
            with (
                self.subTest(runtime=runtime),
                patch("runtime_adapters.urlopen", return_value=FakeResponse(payload)) as open_url,
            ):
                models = adapters.list_models(runtime, endpoints[runtime])
            request = open_url.call_args.args[0]
            self.assertEqual(request.full_url, expected_paths[runtime])
            self.assertEqual(
                models,
                sorted(
                    {
                        entry.get("name", entry.get("id"))
                        for entry in payload.get("models", payload.get("data"))
                    },
                    key=str.casefold,
                ),
            )

    def test_model_discovery_rejects_unsupported_runtime_and_invalid_payloads(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            adapters.list_models("airllm")
        for payload in ([], {}, {"models": [None]}, {"data": [{"name": "no-id"}]}):
            with (
                self.subTest(payload=payload),
                patch("runtime_adapters.urlopen", return_value=FakeResponse(payload)),
                self.assertRaises(RuntimeError),
            ):
                adapters.list_models("ollama")

    def test_endpoint_validation_normalizes_url_without_io(self):
        self.assertEqual(
            adapters.validate_endpoint("HTTPS://localhost:8000/v1/"),
            "https://localhost:8000/v1",
        )
        for endpoint in (
            "",
            "localhost:8000",
            "file://model",
            "http://u:p@host",
            "http://host/path?q=secret",
            "http://host/#fragment",
            "http://host:70000",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                adapters.validate_endpoint(endpoint)

    def test_default_and_versioned_endpoints_join_once(self):
        for endpoint in ("http://localhost:8000/v1", "http://localhost:8000"):
            with patch(
                "runtime_adapters.urlopen",
                return_value=FakeResponse(
                    {
                        "choices": [{"message": {"content": "reply"}}],
                    }
                ),
            ) as open_url:
                adapters.adapt(
                    {
                        "model": "m",
                        "case": {
                            "input": "prompt",
                            "evaluation": {"metric": "exact_match", "expected": "reply"},
                        },
                    },
                    "llama.cpp",
                    endpoint,
                )
            url, _ = read_request(open_url)
            self.assertEqual(url, "http://localhost:8000/v1/chat/completions")


class NetworkRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = "http://127.0.0.1:8000/v1"

    def test_generation_requires_evaluation_before_inference(self):
        for runtime in adapters.DEFAULT_ENDPOINTS:
            with self.subTest(runtime=runtime), patch("runtime_adapters.urlopen") as open_url:
                with self.assertRaisesRegex(ValueError, "explicit evaluation"):
                    adapters.adapt({"model": "m", "case": {"input": "hi"}}, runtime)
                open_url.assert_not_called()

    def test_request_seed_mapping_and_explicit_parameter_precedence(self):
        request = {
            "model": "m",
            "seed": 17,
            "parameters": {},
            "case": {
                "input": "hi",
                "evaluation": {"metric": "exact_match", "expected": "ok"},
            },
        }
        for runtime in ("ollama", "llama.cpp", "colibri"):
            payload = (
                {"message": {"content": "ok"}}
                if runtime == "ollama"
                else {
                    "choices": [{"message": {"content": "ok"}}],
                }
            )
            with patch("runtime_adapters.urlopen", return_value=FakeResponse(payload)) as open_url:
                adapters.adapt(request, runtime)
            _, body = read_request(open_url)
            options = body["options"] if runtime == "ollama" else body
            if runtime == "colibri":
                self.assertNotIn("seed", options)
            else:
                self.assertEqual(options["seed"], 17)
                with patch(
                    "runtime_adapters.urlopen", return_value=FakeResponse(payload)
                ) as open_url:
                    adapters.adapt({**request, "parameters": {"seed": 4}}, runtime)
                _, body = read_request(open_url)
                self.assertEqual((body["options"] if runtime == "ollama" else body)["seed"], 4)
        for seed in (True, -1, None, 1.5):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                adapters.adapt({**request, "seed": seed}, "ollama")

    def test_ollama_chat_maps_parameters_and_returns_raw_output(self):
        request = {
            "model": "local-model",
            "parameters": {
                "context_size": 4096,
                "temperature": 0.25,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 64,
                "repeat_penalty": 1.1,
                "seed": 7,
                "stop": ["STOP"],
            },
            "case": {
                "input": {
                    "parts": [
                        {"type": "text", "text": "Say"},
                        {"type": "text", "text": "hello"},
                    ]
                },
                "evaluation": {"metric": "exact_match", "expected": "hello"},
            },
        }
        with patch(
            "runtime_adapters.urlopen",
            return_value=FakeResponse(
                {
                    "message": {"content": "hello"},
                    "eval_count": 3,
                }
            ),
        ) as open_url:
            result = adapters.adapt(request, "ollama")
        url, body = read_request(open_url)
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(body["messages"][0]["content"], "Say\nhello")
        self.assertEqual(
            body["options"],
            {
                "num_ctx": 4096,
                "temperature": 0.25,
                "top_p": 0.8,
                "top_k": 20,
                "num_predict": 64,
                "repeat_penalty": 1.1,
                "seed": 7,
                "stop": ["STOP"],
            },
        )
        self.assertEqual(result, {"output": "hello", "output_tokens": 3})

    def test_ollama_encodes_image_but_rejects_other_media(self):
        image = Path(__file__).parent / "_runtime_adapter_image_fixture"
        image.write_bytes(b"fake image")
        self.addCleanup(image.unlink, missing_ok=True)
        request = {
            "model": "vision",
            "parameters": {},
            "case": {
                "evaluation": {"metric": "exact_match", "expected": "a picture"},
                "input": {
                    "parts": [
                        {"type": "text", "text": "describe"},
                        {"type": "image", "path": str(image)},
                    ]
                },
            },
        }
        with patch(
            "runtime_adapters.urlopen",
            return_value=FakeResponse(
                {
                    "message": {"content": "a picture"},
                }
            ),
        ) as open_url:
            result = adapters.adapt(request, "ollama")
        _, body = read_request(open_url)
        self.assertEqual(body["messages"][0]["images"], [base64.b64encode(b"fake image").decode()])
        self.assertEqual(
            result,
            {
                "output": "a picture",
            },
        )
        request["case"]["input"]["parts"].append({"type": "audio", "path": str(image)})
        with self.assertRaisesRegex(ValueError, "does not support audio"):
            adapters.adapt(request, "ollama")

    def test_ollama_embedding_and_retrieval(self):
        request = {
            "model": "embed-model",
            "parameters": {},
            "case": {
                "task": "embedding",
                "input": "query",
                "documents": [
                    {"id": "a", "input": "first"},
                    {
                        "id": "b",
                        "input": {
                            "parts": [
                                {"type": "text", "text": "second"},
                            ]
                        },
                    },
                ],
                "evaluation": {"metric": "retrieval", "relevant_ids": ["a"]},
            },
        }
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        with patch(
            "runtime_adapters.urlopen",
            return_value=FakeResponse(
                {
                    "embeddings": vectors,
                }
            ),
        ) as open_url:
            result = adapters.adapt(request, "ollama", "http://localhost:11434")
        url, body = read_request(open_url)
        self.assertEqual(url, "http://localhost:11434/api/embed")
        self.assertEqual(body["input"], ["query", "first", "second"])
        self.assertEqual(
            result,
            {
                "embedding": vectors[0],
                "document_embeddings": {"a": vectors[1], "b": vectors[2]},
            },
        )

    def test_openai_compatible_chat_and_embedding(self):
        generation = {
            "model": "gguf",
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 12,
                "seed": 4,
                "frequency_penalty": 0.2,
            },
            "case": {
                "input": "write",
                "evaluation": {
                    "metric": "json",
                    "json_type": "object",
                },
            },
        }
        with patch(
            "runtime_adapters.urlopen",
            return_value=FakeResponse(
                {
                    "choices": [{"message": {"content": '{"ok":true}'}}],
                    "usage": {"completion_tokens": 5},
                }
            ),
        ) as open_url:
            result = adapters.adapt(generation, "llama.cpp")
        url, body = read_request(open_url)
        self.assertEqual(url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(body["temperature"], 0.1)
        self.assertEqual(body["seed"], 4)
        self.assertEqual(result, {"output": '{"ok":true}', "output_tokens": 5})

        embedding = {
            "model": "embed-model",
            "parameters": {},
            "case": {"task": "embedding", "input": "q", "documents": [{"id": "doc", "input": "d"}]},
        }
        with patch(
            "runtime_adapters.urlopen",
            return_value=FakeResponse(
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ],
                }
            ),
        ) as open_url:
            result = adapters.adapt(embedding, "llama.cpp", self.endpoint)
        url, body = read_request(open_url)
        self.assertEqual(url, "http://127.0.0.1:8000/v1/embeddings")
        self.assertEqual(body["input"], ["q", "d"])
        self.assertEqual(
            result,
            {
                "embedding": [1.0, 0.0],
                "document_embeddings": {"doc": [0.0, 1.0]},
            },
        )

    def test_colibri_uses_openai_chat_api_and_rejects_embedding(self):
        request = {
            "model": "colibri-model",
            "parameters": {"temperature": 0.2},
            "case": {
                "input": "hi",
                "evaluation": {
                    "metric": "exact_match",
                    "expected": "there",
                },
            },
        }
        with patch(
            "runtime_adapters.urlopen",
            return_value=FakeResponse(
                {
                    "choices": [{"message": {"content": "there"}}],
                }
            ),
        ) as open_url:
            result = adapters.adapt(request, "colibri")
        url, body = read_request(open_url)
        self.assertEqual(url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(body["model"], "colibri-model")
        self.assertEqual(result, {"output": "there"})
        request["parameters"] = {"seed": 4}
        with self.assertRaisesRegex(ValueError, "does not support parameter"):
            adapters.adapt(request, "colibri")
        request["parameters"] = {}
        request["case"] = {"task": "embedding", "input": "hi"}
        with self.assertRaisesRegex(ValueError, "does not provide embeddings"):
            adapters.adapt(request, "colibri")

    def test_unsupported_options_fail_before_network_calls(self):
        request = {
            "model": "m",
            "parameters": {"context": 4096},
            "case": {
                "input": "hi",
                "evaluation": {"metric": "exact_match", "expected": "hi"},
            },
        }
        for runtime in ("llama.cpp", "colibri"):
            with (
                self.subTest(runtime=runtime),
                patch("runtime_adapters.urlopen") as open_url,
                self.assertRaisesRegex(ValueError, "does not support"),
            ):
                adapters.adapt(request, runtime)
            open_url.assert_not_called()
        request["parameters"] = {"num_ctx": 4096}
        request["parameters"] = {"n_ctx": 4096}
        with patch("runtime_adapters.urlopen") as open_url:
            with self.assertRaisesRegex(ValueError, "does not support"):
                adapters.adapt(request, "ollama")
        open_url.assert_not_called()

    def test_timeout_http_and_malformed_responses_are_explicit(self):
        request = {
            "model": "m",
            "parameters": {},
            "case": {
                "input": "hi",
                "evaluation": {"metric": "exact_match", "expected": "hi"},
            },
        }
        with patch("runtime_adapters.urlopen", side_effect=URLError("offline")) as open_url:
            with self.assertRaisesRegex(RuntimeError, "Request .* failed"):
                adapters.adapt(request, "ollama", timeout=1.25)
        self.assertEqual(open_url.call_args.kwargs["timeout"], 1.25)
        with patch("runtime_adapters.urlopen", return_value=FakeResponse({"unexpected": True})):
            with self.assertRaisesRegex(RuntimeError, "message.content"):
                adapters.adapt(request, "ollama")
        with patch("runtime_adapters.urlopen", return_value=FakeResponse([])):
            with self.assertRaisesRegex(RuntimeError, "non-object JSON"):
                adapters.adapt(request, "ollama")


class AirLLMTests(unittest.TestCase):
    def test_airllm_uses_local_model_and_returns_generated_suffix(self):
        class FakeIds(list):
            @property
            def shape(self):
                return (1, len(self[0]))

        class FakeTokenizer:
            def __init__(self):
                self.call = None
                self.decoded = None

            def __call__(self, text, **kwargs):
                self.call = (text, kwargs)
                return {"input_ids": FakeIds([[11, 12]])}

            def decode(self, tokens, skip_special_tokens):
                self.decoded = (tokens, skip_special_tokens)
                return "answer"

        class FakeModel:
            def __init__(self):
                self.tokenizer = FakeTokenizer()
                self.call = None

            def generate(self, input_ids, **kwargs):
                self.call = (input_ids, kwargs)
                return types.SimpleNamespace(sequences=[[11, 12, 21, 22]])

        model = FakeModel()
        calls = []
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: False, manual_seed_all=lambda seed: None
            ),
            manual_seed=lambda seed: calls.append(("seed", seed)),
        )
        fake_airllm = types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(
                from_pretrained=lambda path, **kwargs: (
                    calls.append(
                        (
                            "load",
                            path,
                            kwargs,
                            os.environ.get("HF_HUB_OFFLINE"),
                            os.environ.get("TRANSFORMERS_OFFLINE"),
                        )
                    )
                    or model
                )
            )
        )
        model_path = Path(__file__).parent / "_runtime_adapter_model_fixture"
        model_path.mkdir()
        self.addCleanup(model_path.rmdir)
        with patch.dict(sys.modules, {"torch": fake_torch, "airllm": fake_airllm}):
            request = {
                "model": "a-display-name",
                "parameters": {
                    "model_path": str(model_path),
                    "context_size": 32,
                    "max_tokens": 10,
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
                "seed": 5,
                "case": {
                    "input": "question",
                    "evaluation": {
                        "metric": "exact_match",
                        "expected": "answer",
                    },
                },
            }
            result = adapters.adapt(request, "airllm")
        self.assertEqual(result, {"output": "answer", "output_tokens": 2})
        self.assertEqual(
            calls[0],
            (
                "load",
                str(model_path),
                {},
                "1",
                "1",
            ),
        )
        self.assertEqual(calls[1], ("seed", 5))
        self.assertEqual(
            model.tokenizer.call,
            (
                "question",
                {
                    "return_tensors": "pt",
                    "truncation": True,
                    "max_length": 32,
                },
            ),
        )
        self.assertEqual(
            model.call[1],
            {
                "max_new_tokens": 10,
                "temperature": 0.3,
                "top_p": 0.9,
                "do_sample": True,
            },
        )
        self.assertEqual(model.tokenizer.decoded, ([21, 22], True))

    def test_airllm_rejects_embedding_unknown_options_and_remote_models(self):
        request = {
            "model": "org/model",
            "parameters": {"future_parameter": 3},
            "case": {
                "input": "hi",
                "evaluation": {"metric": "exact_match", "expected": "hi"},
            },
        }
        with patch.dict(sys.modules, {}):
            with self.assertRaisesRegex(ValueError, "does not support parameter"):
                adapters.adapt(request, "airllm")
        request["parameters"] = {}
        with self.assertRaisesRegex(ValueError, "existing local model directory"):
            adapters.adapt(request, "airllm")
        request["model"] = "org/model"
        request["case"] = {"task": "embedding", "input": "hi"}
        with self.assertRaisesRegex(ValueError, "generation only"):
            adapters.adapt(request, "airllm")


class CommandTests(unittest.TestCase):
    def test_cli_reads_one_request_and_writes_only_json_to_stdout(self):
        source = io.StringIO(
            json.dumps(
                {
                    "model": "m",
                    "parameters": {},
                    "case": {
                        "input": "hello",
                        "evaluation": {"metric": "exact_match", "expected": "world"},
                    },
                }
            )
        )
        output = io.StringIO()
        with (
            patch.object(sys, "stdin", source),
            patch.object(sys, "stdout", output),
            patch(
                "runtime_adapters.urlopen",
                return_value=FakeResponse({"message": {"content": "world"}}),
            ),
        ):
            status = adapters.main(["--runtime", "ollama"])
        self.assertEqual(status, 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "output": "world",
            },
        )

    def test_cli_validation_failure_is_reported_on_stderr(self):
        source = io.StringIO("{}")
        output = io.StringIO()
        errors = io.StringIO()
        with (
            patch.object(sys, "stdin", source),
            patch.object(sys, "stdout", output),
            patch("runtime_adapters.sys.stderr", errors),
        ):
            status = adapters.main(["--runtime", "colibri", "--endpoint", "bad-url"])
        self.assertEqual(status, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("endpoint must be", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
