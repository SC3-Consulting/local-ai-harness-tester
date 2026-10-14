"""Run command-based benchmarks with persistent, comparable measurements."""

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evaluations import evaluate_generation, validate_evaluation
from hardware import gpu_inventory, memory_inventory, stable_gpus


def canonical(value):
    """Return a canonical JSON string representation of the value."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    """Return a SHA-256 hex digest of the canonical JSON representation of the value."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_json(path):
    """Load a JSON file from the given path, rejecting non-finite values."""
    with open(path, encoding="utf-8") as source:
        return json.load(source, parse_constant=reject_constant)


def reject_constant(value):
    """Raise an error for non-finite JSON values."""
    raise ValueError(f"Non-finite JSON value: {value}")


def validate_vector(vector, name):
    """Validate and normalise a numeric vector, ensuring it is non-empty, finite, and non-zero.

    Args:
        vector (list of int or float): The numeric vector to validate and normalise.
        name (str): The name of the vector, used in error messages.

    Returns:
        list of float: The normalised numeric vector.
    """
    if (
        not isinstance(vector, list)
        or not vector
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
    ):
        raise ValueError(f"{name} must be a non-empty array of finite numbers")
    scale = max(abs(v) for v in vector)
    if scale == 0:
        raise ValueError(f"{name} must not be a zero vector")
    scaled = [v / scale for v in vector]
    norm = math.sqrt(math.fsum(v * v for v in scaled))
    return [v / norm for v in scaled]


def cosine_similarity(left, right):
    """Compute the cosine similarity between two numeric vectors.

    Args:
        left (list of int or float): The first numeric vector.
        right (list of int or float): The second numeric vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    left = validate_vector(left, "embedding")
    right = validate_vector(right, "reference embedding")
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match")
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right))))


def validate_input(value):
    """Validate the structure of an input object, ensuring it contains valid parts.

    Args:
        value (dict): The input object to validate.

    Raises:
        ValueError: If the input object or any of its parts are invalid.
    """
    # Legacy adapters may use arbitrary JSON input; structured parts are opt-in.
    if not isinstance(value, dict) or "parts" not in value:
        return
    parts = value["parts"]
    if not isinstance(parts, list) or not parts:
        raise ValueError("input.parts must be a non-empty array")
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Each input part must be an object")
        kind = part.get("type")
        if kind == "text":
            if not isinstance(part.get("text"), str):
                raise ValueError("Text parts require a text string")
        elif kind in ("image", "audio", "video"):
            if not isinstance(part.get("path"), str) or not part["path"].strip():
                raise ValueError(f"{kind} parts require a local file path")
            if "mime_type" in part and (
                not isinstance(part["mime_type"], str) or not part["mime_type"].strip()
            ):
                raise ValueError("mime_type must be a non-empty string")
        else:
            raise ValueError(f"Unsupported input part type: {kind}")


def validate_case(case):
    """Validate the structure and content of a test case object.

    Args:
        case (dict): The test case object to validate.

    Raises:
        ValueError: If the test case is invalid or contains inconsistent data.
    """
    task = case.get("task", "generation")
    if task not in ("generation", "embedding"):
        raise ValueError(f"Unsupported case task: {task}")
    validate_input(case["input"])
    if task != "embedding":
        if "documents" in case:
            raise ValueError("documents require task 'embedding'")
        validate_evaluation(case)
        return
    evaluation = case.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Embedding cases require an evaluation object")
    metric = evaluation.get("metric")
    if metric == "cosine_similarity":
        validate_vector(evaluation.get("reference"), "evaluation.reference")
        if "documents" in case:
            raise ValueError("documents are only supported for retrieval evaluation")
    elif metric == "retrieval":
        documents = case.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError("Retrieval requires a non-empty documents array")
        ids = []
        for document in documents:
            if (
                not isinstance(document, dict)
                or not isinstance(document.get("id"), str)
                or not document["id"]
                or "input" not in document
            ):
                raise ValueError("Each retrieval document needs an id and input")
            ids.append(document["id"])
            validate_input(document["input"])
        if len(set(ids)) != len(ids):
            raise ValueError("Retrieval document IDs must be unique")
        relevant = evaluation.get("relevant_ids")
        if (
            not isinstance(relevant, list)
            or not relevant
            or any(not isinstance(v, str) or v not in ids for v in relevant)
            or len(set(relevant)) != len(relevant)
        ):
            raise ValueError("relevant_ids must be unique IDs present in documents")
        k = evaluation.get("k", 1)
        if type(k) is not int or not 1 <= k <= len(documents):
            raise ValueError("Retrieval k must be between 1 and the document count")
    else:
        raise ValueError(f"Unsupported embedding metric: {metric}")


def file_digest(path):
    """Compute the SHA-256 hex digest of a file at the given path.

    Args:
        path (str or Path): The path to the file to hash.

    Returns:
        str: The SHA-256 hex digest of the file.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def prepare_input(value, base_dir):
    """Prepare an input object by resolving media file paths and computing their SHA-256 digests.

    Args:
        value (dict): The input object to prepare.
        base_dir (str or Path): The base directory for resolving relative media file paths.

    Returns:
        dict: The prepared input object with resolved paths and SHA-256 digests for media files.

    Raises:
        ValueError: If any media file does not exist.
    """
    if not isinstance(value, dict) or "parts" not in value:
        return value
    parts = []
    for part in value["parts"]:
        if part["type"] == "text":
            parts.append(dict(part))
            continue
        path = (Path(base_dir) / part["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"Media file does not exist: {path}")
        parts.append({**part, "path": str(path), "sha256": file_digest(path)})
    return {**value, "parts": parts}


def prepare_cases(cases, base_dir):
    """Prepare cases by resolving media paths and computing SHA-256 digests.

    Args:
        cases (list of dict): The list of test cases to prepare.
        base_dir (str or Path): The base directory for resolving relative media file paths.

    Returns:
        list of dict: Prepared cases with resolved paths and media SHA-256 digests.

    Raises:
        ValueError: If any media file does not exist.
    """
    prepared = []
    for case in cases:
        updated = {**case, "input": prepare_input(case["input"], base_dir)}
        if "documents" in case:
            updated["documents"] = [
                {**doc, "input": prepare_input(doc["input"], base_dir)} for doc in case["documents"]
            ]
        prepared.append(updated)
    return prepared


def verify_media(case):
    """Verify media files have not changed by comparing their SHA-256 digests.

    Args:
        case (dict): The test case containing media files to verify.

    Raises:
        ValueError: If any media file has changed since it was prepared.
    """
    inputs = [case["input"], *(doc["input"] for doc in case.get("documents", []))]
    for value in inputs:
        if isinstance(value, dict) and "parts" in value:
            for part in value["parts"]:
                if part["type"] != "text" and file_digest(part["path"]) != part["sha256"]:
                    raise ValueError(f"Media file changed during benchmark: {part['path']}")


def evaluate_embedding(response, case):
    """Evaluate the embedding response against the expected evaluation criteria in the test case.

    Args:
        response (dict): The response containing the embedding and optionally document embeddings.
        case (dict): The test case containing the evaluation criteria and reference embeddings.

    Returns:
        dict: The response augmented with evaluation scores and metrics.

    Raises:
        ValueError: If the response does not meet the expected structure or evaluation criteria.
    """
    embedding = response.get("embedding")
    validate_vector(embedding, "embedding")
    evaluation = case["evaluation"]
    if evaluation["metric"] == "cosine_similarity":
        similarity = cosine_similarity(embedding, evaluation["reference"])
        return {
            **response,
            "score": (similarity + 1) / 2,
            "embedding_metrics": {"cosine_similarity": similarity},
            "embedding_count": 1,
            "embedding_dimensions": len(embedding),
        }
    documents = response.get("document_embeddings")
    ids = [doc["id"] for doc in case["documents"]]
    if not isinstance(documents, dict) or set(documents) != set(ids):
        raise ValueError("document_embeddings must contain exactly the configured document IDs")
    similarities = {key: cosine_similarity(embedding, documents[key]) for key in ids}
    ranking = sorted(ids, key=lambda key: (-similarities[key], key))
    relevant = set(evaluation["relevant_ids"])
    k = evaluation.get("k", 1)
    recall = len(relevant.intersection(ranking[:k])) / len(relevant)
    reciprocal_rank = next(1 / (i + 1) for i, key in enumerate(ranking) if key in relevant)
    return {
        **response,
        "score": recall,
        "embedding_metrics": {"recall_at_k": recall, "reciprocal_rank": reciprocal_rank},
        "retrieval_ranking": ranking,
        "retrieval_similarities": similarities,
        "embedding_count": 1 + len(ids),
        "embedding_dimensions": len(embedding),
    }


def machine_info():
    """Retrieve information about the current machine, including CPU, memory, and GPU details.

    Returns:
        dict: Machine information including hostname, OS, CPU, memory, and GPUs.
    """
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    memory = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        memory = next(
            (
                int(line.split()[1])
                for line in meminfo.read_text(encoding="utf-8").splitlines()
                if line.startswith("MemTotal:")
            ),
            None,
        )
    gpus = []
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        vendor, product = device / "vendor", device / "device"
        if vendor.exists() and product.exists():
            gpus.append(
                {
                    "vendor": vendor.read_text(encoding="utf-8").strip(),
                    "device": product.read_text(encoding="utf-8").strip(),
                }
            )
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "memory_kib": memory,
        "gpus": gpus,
        "gpu_hardware": stable_gpus(gpu_inventory()),
    }


def environment_info():
    """Retrieve comprehensive information about the current environment, including machine details, memory inventory, and GPU telemetry.

    Returns:
        dict: Environment information including machine info, memory inventory, and GPU telemetry.
    """
    return {**machine_info(), **memory_inventory(), "gpu_telemetry": gpu_inventory()}


def validate_config(config):
    """Validate the structure and content of the test harness configuration.

    Args:
        config (dict): The configuration object to validate.

    Returns:
        dict: The validated and canonicalized configuration object.

    Raises:
        ValueError: If the configuration is invalid or contains inconsistent data.
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    for name in ("model", "harness"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"{name} must be a non-empty, versioned identifier")
    command = config.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) for arg in command)
        or not command[0]
    ):
        raise ValueError("command must be a non-empty array of string arguments")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty array")
    ids = []
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or "input" not in case
        ):
            raise ValueError("Each case needs a non-empty string id and an input")
        ids.append(case["id"])
        validate_case(case)
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique")
    for name in ("parameters", "parameter_grid"):
        if not isinstance(config.get(name, {}), dict):
            raise ValueError(f"{name} must be an object")
    for name, values in config.get("parameter_grid", {}).items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid parameter {name} needs a non-empty array")
        if len({canonical(v) for v in values}) != len(values):
            raise ValueError(f"Grid parameter {name} contains duplicate values")
    repetitions = config.get("repetitions", 3)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    timeout = config.get("timeout_seconds", 300)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    seed = config.get("seed", 0)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(config.get("cwd", "."), str):
        raise ValueError("cwd must be a path string")
    canonical(config)
    return config


def parameter_sets(config):
    """Generate all combinations of parameters based on the parameter grid in the configuration.

    Args:
        config (dict): Configuration containing the parameter grid and defaults.

    Returns:
        list of dict: A list of parameter sets, each represented as a dictionary.
    """
    grid = config.get("parameter_grid", {})
    keys = sorted(grid)
    return [
        {**config.get("parameters", {}), **dict(zip(keys, values))}
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def connect(path):
    """Connect to the SQLite database and initialize the schema if necessary.

    Args:
        path (str): The path to the SQLite database file.

    Returns:
        sqlite3.Connection: The connection object to the SQLite database.
    """
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            config TEXT NOT NULL,
            machine TEXT NOT NULL,
            scope TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trials (
            run_id TEXT NOT NULL REFERENCES runs(id),
            parameters TEXT NOT NULL,
            case_id TEXT NOT NULL,
            repetition INTEGER NOT NULL,
            status TEXT NOT NULL,
            elapsed_seconds REAL NOT NULL,
            response TEXT,
            error TEXT,
            PRIMARY KEY (run_id, parameters, case_id, repetition)
        );
    """)
    return connection


def validate_response(response):
    """Validate the structure and content of a harness response.

    Args:
        response (dict): The response object to validate.

    Returns:
        dict: The validated and canonicalized response object.

    Raises:
        ValueError: If the response is invalid or contains inconsistent data.
    """
    if not isinstance(response, dict):
        raise ValueError("Harness response must be a JSON object")
    score = response.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("Harness score must be a finite number between 0 and 1")
    if "output_tokens" in response:
        tokens = response["output_tokens"]
        if type(tokens) is not int or tokens < 0:
            raise ValueError("output_tokens must be a non-negative integer")
    canonical(response)
    return response


def execute_trial(command, request, cwd, timeout):
    """Execute a single trial of the test harness with the given command and request.

    Args:
        command (list of str): The command to run the test harness.
        request (dict): The request object to send to the harness.
        cwd (str or Path): The working directory for the harness process.
        timeout (float): The maximum time to allow the harness to run, in seconds.

    Returns:
        dict: Status, elapsed time, response, and error message (if any).
    """
    started = time.perf_counter()
    response, error = None, None
    try:
        case = request.get("case")
        if case is not None:
            verify_media(case)
        process = subprocess.run(
            command,
            input=canonical(request),
            text=True,
            capture_output=True,
            cwd=cwd,
            timeout=timeout,
            check=False,
        )
        if process.returncode != 0:
            raise ValueError(f"Harness exited {process.returncode}: {process.stderr[-4000:]}")
        case = request.get("case")
        if case is not None:
            verify_media(case)
        response = json.loads(process.stdout, parse_constant=reject_constant)
        if not isinstance(response, dict):
            raise ValueError("Harness response must be a JSON object")
        if case is not None and case.get("task") == "embedding":
            response = evaluate_embedding(response, case)
        elif case is not None:
            response = evaluate_generation(response, case)
        response = validate_response(response)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        response = None
    elapsed = time.perf_counter() - started
    return {
        "status": "failed" if error else "succeeded",
        "elapsed_seconds": elapsed,
        "response": response,
        "error": error,
    }


def run_benchmark(connection, config, base_dir, progress=None):
    """Run the benchmark for the given harness configuration and test cases.

    Args:
        connection (sqlite3.Connection): The connection object to the SQLite database.
        config (dict): The configuration object for the harness and test cases.
        base_dir (str or Path): Base directory for resolving configuration paths.
        progress (callable, optional): Receives run ID, completed count, and total
            after initialization and each persisted trial.

    Raises:
        ValueError: If the configuration is invalid or the working directory does not exist.

    Returns:
        str: The unique identifier for the benchmark run.
    """
    validate_config(config)
    cwd = (Path(base_dir) / config.get("cwd", ".")).resolve()
    if not cwd.is_dir():
        raise ValueError(f"Harness working directory does not exist: {cwd}")
    config = {**config, "cwd": str(cwd), "cases": prepare_cases(config["cases"], base_dir)}
    extended = any(
        "task" in case or isinstance(case["input"], dict) and "parts" in case["input"]
        for case in config["cases"]
    )
    protocol = 2 if extended else 1
    machine = machine_info()
    scope = digest(
        {
            "model": config["model"],
            "harness": config["harness"],
            "command": config["command"],
            "cwd": str(cwd),
            "cases": config["cases"],
            "machine": machine,
            "timeout_seconds": config.get("timeout_seconds", 300),
            "protocol": protocol,
        }
    )
    run_id = str(uuid.uuid4())
    connection.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
        (
            run_id,
            datetime.now(timezone.utc).isoformat(),
            "running",
            canonical(config),
            canonical(machine),
            scope,
        ),
    )
    connection.commit()
    print(f"Run {run_id}", file=sys.stderr, flush=True)
    jobs = list(
        itertools.product(
            parameter_sets(config),
            config["cases"],
            range(config.get("repetitions", 3)),
        )
    )
    random.Random(config.get("seed", 0)).shuffle(jobs)
    failed = False
    try:
        if progress is not None:
            progress(run_id, 0, len(jobs))
        for completed, (parameters, case, repetition) in enumerate(jobs, start=1):
            request = {
                "protocol_version": protocol,
                "model": config["model"],
                "harness": config["harness"],
                "parameters": parameters,
                "case": case,
                "repetition": repetition,
                "seed": config.get("seed", 0) + repetition,
            }
            trial = execute_trial(
                config["command"], request, cwd, config.get("timeout_seconds", 300)
            )
            connection.execute(
                "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    canonical(parameters),
                    case["id"],
                    repetition,
                    trial["status"],
                    trial["elapsed_seconds"],
                    canonical(trial["response"]) if trial["response"] else None,
                    trial["error"],
                ),
            )
            connection.commit()
            if progress is not None:
                progress(run_id, completed, len(jobs))
            if trial["error"]:
                failed = True
                print(
                    f"{case['id']} repetition {repetition}: {trial['error']}",
                    file=sys.stderr,
                    flush=True,
                )
    except (KeyboardInterrupt, Exception):
        connection.execute("UPDATE runs SET status = 'interrupted' WHERE id = ?", (run_id,))
        connection.commit()
        raise
    connection.execute(
        "UPDATE runs SET status = ? WHERE id = ?",
        ("completed_with_failures" if failed else "completed", run_id),
    )
    connection.commit()
    return run_id


def get_run(connection, run_id):
    """Retrieve the run information for the given run ID from the database.

    Args:
        connection (sqlite3.Connection): The connection object to the SQLite database.
        run_id (str): The unique identifier for the benchmark run.

    Returns:
        sqlite3.Row: The row containing the run information.

    Raises:
        ValueError: If the run ID is unknown.
    """
    row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown run: {run_id}")
    return row


def summarise(connection, run_id):
    """Summarise the results of a benchmark run, including statistics for each parameter set.

    Args:
        connection (sqlite3.Connection): The connection object to the SQLite database.
        run_id (str): The unique identifier for the benchmark run.

    Returns:
        dict: Run details and parameter summaries with counts, scores, latencies,
            and embedding metrics.

    Raises:
        ValueError: If the run ID is unknown.
    """
    run = get_run(connection, run_id)
    config = json.loads(run["config"])
    expected = len(config["cases"]) * config.get("repetitions", 3)
    groups: dict[str, list[sqlite3.Row]] = {canonical(p): [] for p in parameter_sets(config)}
    for row in connection.execute("SELECT * FROM trials WHERE run_id = ?", (run_id,)):
        groups[row["parameters"]].append(row)
    summaries = []
    for parameters, rows in groups.items():
        successes = [r for r in rows if r["status"] == "succeeded"]
        scores = [json.loads(r["response"])["score"] for r in successes]
        elapsed = [r["elapsed_seconds"] for r in successes]
        tokens = [json.loads(r["response"]).get("output_tokens") for r in successes]
        summaries.append(
            {
                "parameters": json.loads(parameters),
                "expected_trials": expected,
                "successful_trials": len(successes),
                "failed_trials": len(rows) - len(successes),
                "missing_trials": expected - len(rows),
                "eligible": run["status"] in ("completed", "completed_with_failures")
                and len(successes) == expected,
                "mean_score": statistics.mean(scores) if scores else None,
                "score_stdev": statistics.stdev(scores) if len(scores) > 1 else None,
                "mean_latency_seconds": statistics.mean(elapsed) if elapsed else None,
                "latency_stdev_seconds": statistics.stdev(elapsed) if len(elapsed) > 1 else None,
                "tokens_per_second": (
                    sum(tokens) / sum(elapsed)
                    if tokens and all(t is not None for t in tokens) and sum(elapsed) > 0
                    else None
                ),
                "embeddings_per_second": (
                    sum(json.loads(r["response"]).get("embedding_count", 0) for r in successes)
                    / sum(elapsed)
                    if successes
                    and sum(elapsed) > 0
                    and all(
                        json.loads(r["response"]).get("embedding_count") is not None
                        for r in successes
                    )
                    else None
                ),
                "embedding_metrics": {
                    metric: statistics.mean(values)
                    for metric in ("cosine_similarity", "recall_at_k", "reciprocal_rank")
                    if (
                        values := [
                            json.loads(r["response"])["embedding_metrics"][metric]
                            for r in successes
                            if metric in json.loads(r["response"]).get("embedding_metrics", {})
                        ]
                    )
                },
                "generation_metrics": {
                    metric: statistics.mean(values)
                    for metric in ("exact_match", "valid_json", "json_requirements_met")
                    if (
                        values := [
                            json.loads(r["response"])["generation_metrics"][metric]
                            for r in successes
                            if metric in json.loads(r["response"]).get("generation_metrics", {})
                        ]
                    )
                },
            }
        )
    return {
        "run_id": run_id,
        "created_at": run["created_at"],
        "status": run["status"],
        "model": config["model"],
        "harness": config["harness"],
        "scope": run["scope"],
        "tasks": sorted({case.get("task", "generation") for case in config["cases"]}),
        "machine": json.loads(run["machine"]),
        "configurations": summaries,
    }


def recommend(connection, run_id, objective="quality", min_quality=0.0):
    """
    Recommend the best configuration from previous evaluations within the same scope.

    Args:
        connection (sqlite3.Connection): The connection object to the SQLite database.
        run_id (str): The ID of the reference run for which to recommend a configuration.

    Returns:
        dict: The recommended configuration, ranking of all candidates, and related information.

    Raises:
        ValueError: If the reference run belongs to a different machine environment
            or no eligible configurations are found.
    """
    if objective not in ("quality", "latency"):
        raise ValueError("Recommendation objective must be quality or latency")
    if (
        not isinstance(min_quality, (int, float))
        or isinstance(min_quality, bool)
        or not math.isfinite(min_quality)
        or not 0 <= min_quality <= 1
    ):
        raise ValueError("Minimum quality must be a finite score between 0 and 1")
    reference = get_run(connection, run_id)
    if digest(machine_info()) != digest(json.loads(reference["machine"])):
        raise ValueError("Reference run belongs to a different machine environment")
    candidates = []
    for row in connection.execute(
        "SELECT id FROM runs WHERE scope = ? ORDER BY created_at", (reference["scope"],)
    ):
        summary = summarise(connection, row["id"])
        for result in summary["configurations"]:
            if result["eligible"]:
                candidates.append({"run_id": row["id"], **result})
    if not candidates:
        raise ValueError("No complete, successful configurations in this scope")
    # Use the most recent complete evaluation of each parameter set, not its best run.
    latest = {canonical(c["parameters"]): c for c in candidates}
    qualifying = [c for c in latest.values() if c["mean_score"] >= min_quality]
    if not qualifying:
        raise ValueError("No successful configurations meet the minimum quality score")
    ranked = sorted(
        qualifying,
        key=lambda c: (
            (c["mean_latency_seconds"], -c["mean_score"])
            if objective == "latency"
            else (-c["mean_score"], c["mean_latency_seconds"])
        )
        + (canonical(c["parameters"]),),
    )
    return {
        "scope": reference["scope"],
        "objective": (
            "latency_asc_then_mean_score_desc"
            if objective == "latency"
            else "mean_score_desc_then_latency_asc"
        ),
        "min_quality": min_quality,
        "recommended": ranked[0],
        "ranking": ranked,
        "note": "Best measured configuration, not a guarantee of optimal settings.",
    }


def compare(connection, run_ids):
    """Compare multiple benchmark runs to determine if they are comparable.

    Args:
        connection (sqlite3.Connection): The connection object to the SQLite database.
        run_ids (list of str): The list of run IDs to compare.

    Returns:
        dict: Comparability status, a note, and summaries of the runs.

    Raises:
        ValueError: If fewer than two distinct run IDs are provided.
    """
    if len(set(run_ids)) < 2:
        raise ValueError("Comparison requires at least two distinct run IDs")
    summaries = [summarise(connection, run_id) for run_id in run_ids]
    return {
        "comparable": len({r["scope"] for r in summaries}) == 1,
        "note": "Different scopes are shown side by side, not ranked together.",
        "runs": summaries,
    }


def main(argv=None):
    """Entry point for the command-line interface of the harness tester.

    Args:
        argv (list of str, optional): Command-line arguments.
            Defaults to None, which uses sys.argv.

    Returns:
        int: The exit code of the program (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="results.sqlite3", help="SQLite result database")
    commands = parser.add_subparsers(dest="action", required=True)
    run_parser = commands.add_parser("run", help="Run a configured parameter sweep")
    run_parser.add_argument("config")
    commands.add_parser("list", help="List stored runs")
    commands.add_parser("machine", help="Show hardware and available memory")
    serve_parser = commands.add_parser("serve", help="Start the local Web UI")
    serve_parser.add_argument("--port", type=int, default=8765)
    show_parser = commands.add_parser("show", help="Show summaries and individual trials")
    show_parser.add_argument("run_id")
    compare_parser = commands.add_parser("compare", help="Compare stored runs")
    compare_parser.add_argument("run_ids", nargs="+")
    recommend_parser = commands.add_parser("recommend", help="Rank current-machine settings")
    recommend_parser.add_argument("run_id", help="Reference run defining model/harness/suite")
    recommend_parser.add_argument("--objective", choices=("quality", "latency"), default="quality")
    recommend_parser.add_argument("--min-quality", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        if args.action == "serve":
            from web_ui import serve

            serve(args.db, args.port)
            return 0
        if args.action == "machine":
            print(json.dumps(environment_info(), indent=2, allow_nan=False))
            return 0
        with connect(args.db) as connection:
            if args.action == "run":
                config_path = Path(args.config).resolve()
                run_id = run_benchmark(connection, load_json(config_path), config_path.parent)
                output = summarise(connection, run_id)
            elif args.action == "list":
                output = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT id, created_at, status, scope FROM runs ORDER BY created_at DESC"
                    )
                ]
            elif args.action == "show":
                output = summarise(connection, args.run_id)
                output["config"] = json.loads(get_run(connection, args.run_id)["config"])
                output["trials"] = [
                    {
                        **dict(row),
                        "parameters": json.loads(row["parameters"]),
                        "response": json.loads(row["response"]) if row["response"] else None,
                    }
                    for row in connection.execute(
                        "SELECT * FROM trials WHERE run_id = ? "
                        "ORDER BY parameters, case_id, repetition",
                        (args.run_id,),
                    )
                ]
            elif args.action == "compare":
                output = compare(connection, args.run_ids)
            else:
                output = recommend(connection, args.run_id, args.objective, args.min_quality)
            print(json.dumps(output, indent=2, allow_nan=False))
            return 1 if args.action == "run" and output["status"] != "completed" else 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; completed trials remain stored.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
