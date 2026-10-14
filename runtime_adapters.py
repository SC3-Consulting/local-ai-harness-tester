"""Self-contained command adapters for local inference runtimes.

Each adapter reads one harness request from stdin and writes one JSON response
to stdout. Network clients use only the Python standard library; AirLLM is
imported lazily and is only used with an existing local model directory.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

DEFAULT_ENDPOINTS = {
    "ollama": "http://127.0.0.1:11434",
    "llama.cpp": "http://127.0.0.1:8080",
    "airllm": None,
    "colibri": "http://127.0.0.1:8000/v1",
}
DEFAULT_TIMEOUT = 300.0


def validate_endpoint(endpoint: str) -> str:
    """Validate and normalise an HTTP(S) endpoint URL without making a request."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty HTTP(S) URL")
    endpoint = endpoint.strip()
    try:
        parsed = urlsplit(endpoint)
        valid_port = parsed.port is None or 1 <= parsed.port <= 65535
    except ValueError as exc:
        raise ValueError(f"Invalid endpoint URL: {exc}") from exc
    invalid_authority = (
        parsed.scheme.lower() not in ("http", "https") or not parsed.hostname or not valid_port
    )
    if invalid_authority or any(
        (parsed.username is not None, parsed.password is not None, parsed.query, parsed.fragment)
    ):
        raise ValueError(
            "endpoint must be an HTTP(S) URL with a host, optional port/path, "
            "and no credentials, query, or fragment"
        )
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _endpoint_for(runtime: str, endpoint: str | None) -> str:
    """Determine the effective endpoint for a given runtime, validating it if necessary.

    Args:
        runtime (str): The name of the runtime (e.g., "ollama", "llama.cpp", "airllm", "colibri").
        endpoint (str | None): The user-provided endpoint URL, or None to use the default.

    Returns:
        str: The validated and normalised endpoint URL.

    Raises:
        ValueError: If the endpoint is invalid or not applicable for the runtime.
    """
    if endpoint is None:
        endpoint = DEFAULT_ENDPOINTS[runtime]
    if endpoint is None:
        raise ValueError("airllm is an in-process runtime and does not use --endpoint")
    return validate_endpoint(endpoint)


def _api_url(endpoint: str, path: str) -> str:
    """Construct the full API URL by combining the endpoint and the path, handling versioning.

    Args:
        endpoint (str): The base endpoint URL.
        path (str): The API path to append to the endpoint.

    Returns:
        str: The full API URL.
    """
    parsed = urlsplit(endpoint)
    prefix = parsed.path.rstrip("/")
    if path.startswith("/v1/") and prefix.endswith("/v1"):
        path = path[3:]
    return urlunsplit((parsed.scheme, parsed.netloc, prefix + path, "", ""))


def _post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send a POST request with a JSON body to the specified URL and return the JSON response.

    Args:
        url (str): The URL to send the POST request to.
        body (dict[str, Any]): The JSON-serializable body of the request.
        timeout (float): The timeout for the request in seconds.

    Returns:
        dict[str, Any]: The JSON-decoded response from the server.

    Raises:
        RuntimeError: If the request fails or the response is not valid JSON.
    """
    request = Request(
        url,
        data=json.dumps(body, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail or exc.reason}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Runtime returned invalid JSON from {url}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Runtime returned a non-object JSON response from {url}")
    return decoded


def list_models(runtime: str, endpoint: str | None = None, timeout: float = 5.0) -> list[str]:
    """List models exposed by a running Ollama or OpenAI-compatible server."""
    if runtime not in ("ollama", "llama.cpp", "colibri"):
        raise ValueError(f"Model listing is not supported for {runtime}")
    endpoint = _endpoint_for(runtime, endpoint)
    path = "/api/tags" if runtime == "ollama" else "/v1/models"
    url = _api_url(endpoint, path)
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail or exc.reason}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Runtime returned invalid JSON from {url}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Runtime returned a non-object JSON response from {url}")
    entries = decoded.get("models") if runtime == "ollama" else decoded.get("data")
    if not isinstance(entries, list):
        raise RuntimeError(f"Runtime returned no model list from {url}")
    models = []
    for entry in entries:
        key = "name" if runtime == "ollama" else "id"
        if not isinstance(entry, dict) or not isinstance(entry.get(key), str):
            raise RuntimeError(f"Runtime returned an invalid model entry from {url}")
        models.append(entry[key])
    return sorted(set(models), key=str.casefold)


def runtime_availability(runtime: str, endpoint: str | None = None) -> dict[str, Any]:
    """Check API reachability or optional dependencies without running inference."""
    if runtime == "custom":
        return {"available": True, "reason": "Custom command support is always available."}
    if runtime == "airllm":
        missing = [name for name in ("airllm", "torch") if importlib.util.find_spec(name) is None]
        return {
            "available": not missing,
            "reason": (
                "Missing local dependencies: " + ", ".join(missing)
                if missing
                else "Local dependencies detected; model readiness is not verified."
            ),
        }
    try:
        models = list_models(runtime, endpoint, timeout=2.0)
        return {"available": True, "reason": f"Server detected; {len(models)} model(s) listed."}
    except (RuntimeError, ValueError, OSError) as exc:
        return {"available": False, "reason": str(exc)}


def context_defaults(runtime: str, model: str, endpoint: str | None = None) -> dict[str, Any]:
    """Suggest sweep bounds from advertised limits, not a memory-fit prediction."""
    if runtime not in DEFAULT_ENDPOINTS and runtime != "custom":
        raise ValueError(f"Unknown runtime: {runtime}")
    if runtime in ("llama.cpp", "colibri"):
        return {
            "supported": False,
            "maximum": None,
            "minimum": None,
            "step": None,
            "source": "server configuration",
            "note": "This adapter cannot vary context per request. Configure context on the "
            "server and benchmark separate server configurations.",
        }
    maximum = 8192 if runtime in ("ollama", "custom") else 4096
    source = "fallback estimate"
    note = "No advertised limit available; verify the model/runtime limit before running."
    limits = []
    try:
        if runtime == "ollama" and model.strip():
            metadata = _post_json(
                _api_url(_endpoint_for(runtime, endpoint), "/api/show"), {"model": model}, 5.0
            )
            info = metadata.get("model_info", {})
            if not isinstance(info, dict):
                raise ValueError("Ollama model_info must be an object")
            limits = [value for key, value in info.items() if key.endswith(".context_length")]
            source = "Ollama model_info context_length"
        elif runtime == "airllm" and model.strip():
            with (Path(model).expanduser() / "config.json").open(encoding="utf-8") as config_file:
                info = json.load(config_file)
            if not isinstance(info, dict):
                raise ValueError("Local model config.json must be an object")
            limits = [
                info[key]
                for key in ("max_position_embeddings", "n_positions", "n_ctx")
                if key in info
            ]
            source = "local model config.json"
        valid = [value for value in limits if type(value) is int and value > 0]
        if valid:
            maximum = min(valid)
            note = "Advertised model limit; not a guarantee of runtime support or available memory."
        else:
            source = "fallback estimate"
    except (OSError, ValueError, RuntimeError) as exc:
        source = "fallback estimate"
        note = f"Context metadata unavailable: {exc}. Verify the limit before running."
    if runtime == "airllm":
        note += " AirLLM context_size controls input truncation, not KV-cache allocation."
    return {
        "supported": True,
        "maximum": maximum,
        "minimum": min(2048, maximum),
        "step": min(2048, maximum),
        "source": source,
        "note": note,
    }


def _request_model(request: dict[str, Any]) -> str:
    """Extract and validate the model from the request.

    Args:
        request (dict[str, Any]): The request object containing the model information.

    Returns:
        str: The validated model string.

    Raises:
        ValueError: If the model is missing or not a non-empty string.
    """
    model = request.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("request.model must be a non-empty string")
    return model


def _case(request: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate the case from the request.

    Args:
        request (dict[str, Any]): The request object containing the case information.

    Returns:
        dict[str, Any]: The validated case object.

    Raises:
        ValueError: If the case is missing, not an object, or contains invalid fields.
    """
    case = request.get("case")
    if not isinstance(case, dict):
        raise ValueError("request.case must be an object")
    task = case.get("task", "generation")
    if task not in ("generation", "embedding"):
        raise ValueError(f"Unsupported task: {task}")
    if "input" not in case:
        raise ValueError("request.case.input is required")
    if task == "generation" and not isinstance(case.get("evaluation"), dict):
        raise ValueError("Generation cases require an explicit evaluation object")
    return case


def _text_input(value: Any) -> str:
    """Extract and validate text input from the given value.

    Args:
        value (Any): The input value, which can be a string or an object containing parts.

    Returns:
        str: The concatenated text input.

    Raises:
        ValueError: If the input is not valid text or contains unsupported parts.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, dict) or not isinstance(value.get("parts"), list):
        raise ValueError("input must be text or an object containing parts")
    text: list[str] = []
    for part in value["parts"]:
        if not isinstance(part, dict):
            raise ValueError("Each input part must be an object")
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            text.append(part["text"])
        elif kind in ("image", "audio", "video"):
            raise ValueError(f"This runtime does not support {kind} input")
        else:
            raise ValueError(f"Invalid text input part: {kind}")
    return "\n".join(text)


def _ollama_chat_content(value: Any) -> tuple[str, list[str]]:
    """Extract and validate chat content for the Ollama adapter.

    Args:
        value (Any): The input value, which can be a string or an object containing parts.

    Returns:
        tuple[str, list[str]]: The concatenated text content and a list of base64-encoded images.

    Raises:
        ValueError: If the input is not valid text or contains unsupported parts.
    """
    if isinstance(value, str):
        return value, []
    if not isinstance(value, dict) or not isinstance(value.get("parts"), list):
        raise ValueError("input must be text or an object containing parts")
    text: list[str] = []
    images: list[str] = []
    for part in value["parts"]:
        if not isinstance(part, dict):
            raise ValueError("Each input part must be an object")
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            text.append(part["text"])
        elif kind == "image":
            image_path = part.get("path")
            if not isinstance(image_path, str) or not Path(image_path).is_file():
                raise ValueError(f"Image file does not exist: {image_path}")
            images.append(base64.b64encode(Path(image_path).read_bytes()).decode("ascii"))
        elif kind in ("audio", "video"):
            raise ValueError(f"Ollama adapter does not support {kind} input")
        else:
            raise ValueError(f"Invalid chat input part: {kind}")
    return "\n".join(text), images


def _params(request: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate the parameters from the request.

    Args:
        request (dict[str, Any]): The request object containing the parameters.

    Returns:
        dict[str, Any]: The validated parameters object.

    Raises:
        ValueError: If the parameters are missing or not an object.
    """
    params = request.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError("request.parameters must be an object")
    return params


def _reject_unknown(params: dict[str, Any], allowed: set[str], runtime: str) -> None:
    """Reject any unknown parameters that are not in the allowed set.

    Args:
        params (dict[str, Any]): The parameters to check.
        allowed (set[str]): The set of allowed parameter names.
        runtime (str): The name of the runtime for error messages.

    Raises:
        ValueError: If any unknown parameters are found.
    """
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{runtime} does not support parameter(s): {', '.join(unknown)}")


def _number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    """Validate and convert a numeric parameter.

    Args:
        name (str): The name of the parameter.
        value (Any): The value of the parameter to validate.
        minimum (float | None, optional): The minimum allowed value. Defaults to None.
        maximum (float | None, optional): The maximum allowed value. Defaults to None.
        exclusive_minimum (bool, optional): Whether the minimum is exclusive. Defaults to False.

    Returns:
        float: The validated numeric value.

    Raises:
        ValueError: If the value is not a finite number or violates the specified constraints.
    """
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"parameter {name} must be a finite number")
    numeric = float(value)
    if minimum is not None and (numeric <= minimum if exclusive_minimum else numeric < minimum):
        comparator = "greater than" if exclusive_minimum else "at least"
        raise ValueError(f"parameter {name} must be {comparator} {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"parameter {name} must be at most {maximum}")
    return numeric


def _integer(name: str, value: Any, *, minimum: int = 1) -> int:
    """Validate and convert an integer parameter.

    Args:
        name (str): The name of the parameter.
        value (Any): The value of the parameter to validate.
        minimum (int, optional): The minimum allowed value. Defaults to 1.

    Returns:
        int: The validated integer value.

    Raises:
        ValueError: If the value is not an integer or is less than the minimum.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"parameter {name} must be an integer >= {minimum}")
    return value


def _stops(name: str, value: Any) -> str | list[str]:
    """Validate and convert a stop parameter.

    Args:
        name (str): The name of the parameter.
        value (Any): The value of the parameter to validate.

    Returns:
        str | list[str]: A non-empty string or list of non-empty strings.

    Raises:
        ValueError: If the value is not a non-empty string or a list of non-empty strings.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return value
    raise ValueError(f"parameter {name} must be a non-empty string or array of strings")


def _ollama_options(params: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate Ollama-specific options from the parameters.

    Args:
        params (dict[str, Any]): The input parameters to validate and transform.

    Returns:
        dict[str, Any]: The validated and transformed Ollama options.

    Raises:
        ValueError: If any parameter is invalid or unknown.
    """
    allowed = {
        "context",
        "context_size",
        "num_ctx",
        "temperature",
        "temp",
        "top_p",
        "top_k",
        "seed",
        "max_tokens",
        "num_predict",
        "repeat_penalty",
        "repetition_penalty",
        "stop",
    }
    _reject_unknown(params, allowed, "ollama")
    options: dict[str, Any] = {}
    for names, key, transform in (
        (("context", "context_size", "num_ctx"), "num_ctx", _integer),
        (("temperature", "temp"), "temperature", lambda k, v: _number(k, v, minimum=0)),
        (("top_p",), "top_p", lambda k, v: _number(k, v, minimum=0, maximum=1)),
        (("top_k",), "top_k", _integer),
        (("seed",), "seed", lambda k, v: _integer(k, v, minimum=0)),
        (("max_tokens", "num_predict"), "num_predict", _integer),
        (
            ("repeat_penalty", "repetition_penalty"),
            "repeat_penalty",
            lambda k, v: _number(k, v, minimum=0, exclusive_minimum=True),
        ),
        (("stop",), "stop", _stops),
    ):
        present = [name for name in names if name in params]
        if len(present) > 1:
            raise ValueError(f"Specify only one of {', '.join(names)}")
        if present:
            name = present[0]
            options[key] = transform(name, params[name])
    return options


def _openai_params(params: dict[str, Any], runtime: str, allowed: set[str]) -> dict[str, Any]:
    """Extract and validate OpenAI-specific parameters from the input.

    Args:
        params (dict[str, Any]): The input parameters to validate and transform.
        runtime (str): The name of the runtime for error messages.
        allowed (set[str]): The set of allowed parameter names.

    Returns:
        dict[str, Any]: The validated and transformed OpenAI parameters.

    Raises:
        ValueError: If any parameter is invalid or unknown.
    """
    _reject_unknown(params, allowed, runtime)
    result: dict[str, Any] = {}
    if "temperature" in params:
        result["temperature"] = _number("temperature", params["temperature"], minimum=0)
    if "top_p" in params:
        result["top_p"] = _number("top_p", params["top_p"], minimum=0, maximum=1)
    if "max_tokens" in params:
        result["max_tokens"] = _integer("max_tokens", params["max_tokens"])
    if "stop" in params:
        result["stop"] = _stops("stop", params["stop"])
    if "seed" in params:
        result["seed"] = _integer("seed", params["seed"], minimum=0)
    if "frequency_penalty" in params:
        result["frequency_penalty"] = _number(
            "frequency_penalty", params["frequency_penalty"], minimum=-2, maximum=2
        )
    if "presence_penalty" in params:
        result["presence_penalty"] = _number(
            "presence_penalty", params["presence_penalty"], minimum=-2, maximum=2
        )
    return result


def _finish_generation(
    request: dict[str, Any], output: str, output_tokens: int | None = None
) -> dict[str, Any]:
    """Finalise the generation response.

    Args:
        request (dict[str, Any]): The original generation request.
        output (str): The generated output text.
        output_tokens (int | None, optional): The number of output tokens. Defaults to None.

    Returns:
        dict[str, Any]: The finalized generation response.
    """
    _case(request)
    response: dict[str, Any] = {"output": output}
    if output_tokens is not None:
        response["output_tokens"] = output_tokens
    return response


def _usage_tokens(response: dict[str, Any]) -> int | None:
    usage = response.get("usage")
    if isinstance(usage, dict):
        count = usage.get("completion_tokens")
        if type(count) is int and count >= 0:
            return count
    return None


def _openai_output(response: dict[str, Any]) -> str:
    """Extract the generated output text from an OpenAI-style response.

    Args:
        response (dict[str, Any]): The response from the OpenAI API.

    Returns:
        str: The generated output text.

    Raises:
        RuntimeError: If the response does not contain the expected output.
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Runtime response is missing choices[0]")
    choice = choices[0]
    message = choice.get("message")
    output = message.get("content") if isinstance(message, dict) else None
    if output is None:
        output = choice.get("text")
    if not isinstance(output, str):
        raise RuntimeError("Runtime response does not contain generated text")
    return output


def _embedding_inputs(case: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    """Extract the query and documents for an embedding request.

    Args:
        case (dict[str, Any]): The generation case containing the input and documents.

    Returns:
        tuple[str, list[tuple[str, str]]]: Query text and document ID/text tuples.

    Raises:
        ValueError: If any document is missing an ID or input.
    """
    query = _text_input(case["input"])
    documents: list[tuple[str, str]] = []
    for document in case.get("documents", []):
        if not isinstance(document, dict) or not isinstance(document.get("id"), str):
            raise ValueError("Each retrieval document must contain an id")
        documents.append((document["id"], _text_input(document.get("input"))))
    return query, documents


def _embedding_response(vectors: list[Any], documents: list[tuple[str, str]]) -> dict[str, Any]:
    """Construct the embedding response from the vectors and documents.

    Args:
        vectors (list[Any]): The list of embedding vectors returned by the runtime.
        documents (list[tuple[str, str]]): The list of document ID and text tuples.

    Returns:
        dict[str, Any]: Query and document embeddings in harness response format.

    Raises:
        RuntimeError: If vectors are missing or the returned count is incorrect.
    """
    if not vectors:
        raise RuntimeError("Runtime returned no embeddings")
    result: dict[str, Any] = {"embedding": vectors[0]}
    if documents:
        if len(vectors) != len(documents) + 1:
            raise RuntimeError("Runtime returned an unexpected number of embeddings")
        result["document_embeddings"] = {
            document_id: vector for (document_id, _), vector in zip(documents, vectors[1:])
        }
    return result


def run_ollama(request: dict[str, Any], endpoint: str, timeout: float) -> dict[str, Any]:
    """Run a generation or embedding request using the Ollama runtime.

    Args:
        request (dict[str, Any]): The generation request.
        endpoint (str): The Ollama API endpoint.
        timeout (float): The request timeout in seconds.

    Returns:
        dict[str, Any]: The generation or embedding response.

    Raises:
        RuntimeError: If the Ollama API response is invalid or missing required fields.
    """
    case = _case(request)
    model = _request_model(request)
    params = _params(request)
    if case.get("task", "generation") == "embedding":
        if params:
            _reject_unknown(params, set(), "ollama embedding")
        query, documents = _embedding_inputs(case)
        inputs = [query, *(text for _, text in documents)]
        response = _post_json(
            _api_url(endpoint, "/api/embed"),
            {"model": model, "input": inputs},
            timeout,
        )
        vectors = response.get("embeddings")
        if not isinstance(vectors, list):
            raise RuntimeError("Ollama /api/embed response is missing embeddings")
        return _embedding_response(vectors, documents)

    content, images = _ollama_chat_content(case["input"])
    options = _ollama_options(params)
    message: dict[str, Any] = {"role": "user", "content": content}
    if images:
        message["images"] = images
    response = _post_json(
        _api_url(endpoint, "/api/chat"),
        {"model": model, "messages": [message], "stream": False, "options": options},
        timeout,
    )
    message_response = response.get("message")
    output = message_response.get("content") if isinstance(message_response, dict) else None
    if not isinstance(output, str):
        raise RuntimeError("Ollama /api/chat response is missing message.content")
    token_count = response.get("eval_count")
    return _finish_generation(
        request, output, token_count if type(token_count) is int and token_count >= 0 else None
    )


def run_openai_compatible(
    request: dict[str, Any], endpoint: str, timeout: float, runtime: str
) -> dict[str, Any]:
    """Run a generation or embedding request using an OpenAI-compatible runtime.

    Args:
        request (dict[str, Any]): The generation request.
        endpoint (str): The API endpoint for the OpenAI-compatible runtime.
        timeout (float): The request timeout in seconds.
        runtime (str): The name of the runtime (e.g., "colibri", "llama.cpp").

    Returns:
        dict[str, Any]: The generation or embedding response.

    Raises:
        RuntimeError: If the API response is invalid or missing required fields.
    """
    case = _case(request)
    model = _request_model(request)
    params = _params(request)
    if case.get("task", "generation") == "embedding":
        if runtime == "colibri":
            raise ValueError("Colibri's documented API does not provide embeddings")
        if params:
            _reject_unknown(params, set(), "llama.cpp embedding")
        query, documents = _embedding_inputs(case)
        response = _post_json(
            _api_url(endpoint, "/v1/embeddings"),
            {"model": model, "input": [query, *(text for _, text in documents)]},
            timeout,
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding response is missing data")
        try:
            ordered = sorted(data, key=lambda row: row["index"])
            vectors = [row["embedding"] for row in ordered]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Embedding response data is malformed") from exc
        return _embedding_response(vectors, documents)

    _text_input(case["input"])
    if isinstance(case["input"], str):
        content = case["input"]
    else:
        content = "\n".join(
            part["text"] for part in case["input"]["parts"] if part["type"] == "text"
        )
    if runtime == "colibri":
        payload_params = _openai_params(
            params, runtime, {"temperature", "top_p", "max_tokens", "stop"}
        )
        if isinstance(payload_params.get("stop"), list) and len(payload_params["stop"]) > 4:
            raise ValueError("Colibri accepts at most four stop sequences")
    else:
        payload_params = _openai_params(
            params,
            runtime,
            {
                "temperature",
                "top_p",
                "max_tokens",
                "stop",
                "seed",
                "frequency_penalty",
                "presence_penalty",
            },
        )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        **payload_params,
    }
    response = _post_json(_api_url(endpoint, "/v1/chat/completions"), body, timeout)
    return _finish_generation(request, _openai_output(response), _usage_tokens(response))


def _airllm_parameters(params: dict[str, Any]) -> tuple[dict[str, Any], int | None, int | None]:
    """Parse and validate the parameters for the AirLLM runtime.

    Args:
        params (dict[str, Any]): The input parameters for the AirLLM runtime.

    Returns:
        tuple[dict[str, Any], int | None, int | None]: Generation parameters,
            context size, and optional seed.

    Raises:
        ValueError: If any parameter is invalid or if mutually exclusive parameters are specified.
    """
    allowed = {
        "context",
        "context_size",
        "num_ctx",
        "max_tokens",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "do_sample",
        "seed",
        "model_path",
    }
    _reject_unknown(params, allowed, "airllm")
    for aliases in (("context", "context_size", "num_ctx"), ("max_tokens", "max_new_tokens")):
        if sum(name in params for name in aliases) > 1:
            raise ValueError(f"Specify only one of {', '.join(aliases)}")
    generation: dict[str, Any] = {}
    context = None
    context_keys = [key for key in ("context", "context_size", "num_ctx") if key in params]
    if context_keys:
        key = context_keys[0]
        context = _integer(key, params[key])
    if "max_tokens" in params or "max_new_tokens" in params:
        key = "max_tokens" if "max_tokens" in params else "max_new_tokens"
        generation["max_new_tokens"] = _integer(key, params[key])
    for name in ("temperature",):
        if name in params:
            generation[name] = _number(name, params[name], minimum=0)
    if "top_p" in params:
        generation["top_p"] = _number("top_p", params["top_p"], minimum=0, maximum=1)
    if "top_k" in params:
        generation["top_k"] = _integer("top_k", params["top_k"])
    if "repetition_penalty" in params:
        generation["repetition_penalty"] = _number(
            "repetition_penalty",
            params["repetition_penalty"],
            minimum=0,
            exclusive_minimum=True,
        )
    if "do_sample" in params:
        if type(params["do_sample"]) is not bool:
            raise ValueError("parameter do_sample must be a boolean")
        generation["do_sample"] = params["do_sample"]
        if not params["do_sample"] and any(
            name in params for name in ("temperature", "top_p", "top_k")
        ):
            raise ValueError("temperature, top_p, and top_k require do_sample=true in AirLLM")
    elif "temperature" in params and params["temperature"] == 0:
        if "top_p" in params or "top_k" in params:
            raise ValueError("top_p and top_k cannot be used with temperature=0")
        generation["do_sample"] = False
    elif any(name in params for name in ("temperature", "top_p", "top_k")):
        generation["do_sample"] = True
    if "seed" in params:
        _integer("seed", params["seed"], minimum=0)
    return generation, context, params.get("seed")


def _airllm_result_tokens(result: Any) -> list[int]:
    """Extract the generated token sequence from an AirLLM result.

    Args:
        result (Any): The result object returned by the AirLLM generate() method.

    Returns:
        list[int]: The list of generated token IDs.

    Raises:
        RuntimeError: If the result does not contain valid token sequences.
    """
    sequences = getattr(result, "sequences", result)
    if hasattr(sequences, "tolist"):
        sequences = sequences.tolist()
    if not isinstance(sequences, (list, tuple)) or not sequences:
        raise RuntimeError("AirLLM generate() returned no token sequences")
    sequence = sequences[0] if isinstance(sequences[0], (list, tuple)) else sequences
    if hasattr(sequence, "tolist"):
        sequence = sequence.tolist()
    if not isinstance(sequence, (list, tuple)):
        raise RuntimeError("AirLLM generate() returned malformed token sequences")
    return list(sequence)


@contextlib.contextmanager
def _offline_hub() -> Iterator[None]:
    """Temporarily put Hugging Face and Transformers in offline mode.

    Yields:
        Iterator[None]: Control while offline mode is active.

    Example:
        with _offline_hub():
            # Perform operations that require offline mode
            ...
    """
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_airllm(request: dict[str, Any]) -> dict[str, Any]:
    """Run a text generation request using the AirLLM runtime.

    Args:
        request (dict[str, Any]): The generation request containing the input text and parameters.

    Returns:
        dict[str, Any]: The generation response containing the generated text and metadata.

    Raises:
        ValueError: If the request is invalid or if required parameters are missing.
        RuntimeError: If the AirLLM runtime fails to generate a valid response.
    """
    case = _case(request)
    if case.get("task", "generation") == "embedding":
        raise ValueError("AirLLM adapter supports generation only, not embeddings")
    text = _text_input(case["input"])
    params = _params(request)
    generation, context, seed = _airllm_parameters(params)
    model_path_value = params.get("model_path", _request_model(request))
    if not isinstance(model_path_value, str) or not model_path_value.strip():
        raise ValueError("AirLLM parameter model_path must be a local directory path")
    model_path = Path(model_path_value).expanduser()
    if not model_path.is_dir():
        raise ValueError(
            "AirLLM requires request.model to be an existing local model directory; "
            "remote model IDs and automatic downloads are disabled"
        )
    try:
        with _offline_hub(), contextlib.redirect_stdout(sys.stderr):
            torch: Any = importlib.import_module("torch")
            airllm = importlib.import_module("airllm")
            auto_model: Any = getattr(airllm, "AutoModel")

            model = auto_model.from_pretrained(str(model_path))
            tokenizer = model.tokenizer
            tokenized = tokenizer(
                text,
                return_tensors="pt",
                truncation=context is not None,
                **({"max_length": context} if context is not None else {}),
            )
            input_ids = tokenized["input_ids"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            result = model.generate(input_ids, **generation)
            tokens = _airllm_result_tokens(result)
            input_length = (
                input_ids.shape[-1]
                if hasattr(input_ids, "shape")
                else len(
                    input_ids[0]
                    if input_ids and isinstance(input_ids[0], (list, tuple))
                    else input_ids
                )
            )
            generated = tokens[input_length:]
            output = tokenizer.decode(generated, skip_special_tokens=True)
    except ImportError as exc:
        raise RuntimeError(
            "AirLLM is optional; install airllm and its runtime dependencies to use it"
        ) from exc
    return _finish_generation(request, output, len(generated))


def adapt(
    request: dict[str, Any],
    runtime: str,
    endpoint: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Run one runtime adapter for a parsed harness request.

    Args:
        request (dict[str, Any]): The parsed harness request.
        runtime (str): The name of the runtime to use.
        endpoint (str | None): The endpoint URL for the runtime, if applicable.
        timeout (float): The request timeout in seconds.

    Returns:
        dict[str, Any]: The response from the runtime adapter.

    Raises:
        ValueError: If any of the input arguments are invalid.
        RuntimeError: If the runtime adapter fails to process the request.
    """
    if runtime not in DEFAULT_ENDPOINTS:
        raise ValueError(f"Unsupported runtime: {runtime}")
    if not isinstance(request, dict):
        raise ValueError("The request must be a JSON object")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    case = _case(request)
    if (
        case.get("task", "generation") == "generation"
        and runtime != "colibri"
        and "seed" in request
        and "seed" not in _params(request)
    ):
        seed = _integer("seed", request["seed"], minimum=0)
        request = {**request, "parameters": {**_params(request), "seed": seed}}
    if runtime == "airllm":
        if endpoint is not None:
            raise ValueError("airllm is in-process and does not accept --endpoint")
        return run_airllm(request)
    base = _endpoint_for(runtime, endpoint)
    if runtime == "ollama":
        return run_ollama(request, base, timeout)
    return run_openai_compatible(request, base, timeout, runtime)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the runtime adapter command-line interface.

    Args:
        argv (list[str] | None): Arguments, defaulting to sys.argv[1:].

    Returns:
        int: The exit code of the program. 0 indicates success, 2 indicates an error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=sorted(DEFAULT_ENDPOINTS))
    parser.add_argument("--endpoint", help="HTTP(S) service URL (not used by AirLLM)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP request timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    args = parser.parse_args(argv)
    try:
        if args.runtime == "airllm" and args.endpoint is not None:
            raise ValueError("airllm is in-process and does not accept --endpoint")
        if args.runtime != "airllm":
            _endpoint_for(args.runtime, args.endpoint)
        request = json.load(sys.stdin)
        response = adapt(request, args.runtime, args.endpoint, args.timeout)
        sys.stdout.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
        return 0
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"runtime adapter error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
