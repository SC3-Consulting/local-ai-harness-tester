# Local AI Harness Tester

A Python 3.10+ local benchmark CLI and Web UI for WSL/Linux. The benchmark core has no runtime dependencies; individual model runtimes may need optional packages:

- Persist individual trials, responses, errors, configuration, and machine metadata in SQLite.
- Compare saved runs side by side, with an explicit comparability flag.
- Sweep the Cartesian product of parameter values and repeat every test case.
- Recommend the highest average quality configuration, breaking exact ties by lowest average latency.
- Run multimodal cases with text, local images, audio, and video.
- Evaluate embeddings against reference vectors or by retrieval ranking.
- Use a local HTTP UI to configure sweeps, run tests, inspect results, compare runs, and request recommendations.
- Discover CPU, RAM, GPU, driver, and VRAM telemetry when available.
- Evaluate exact answers and strict JSON output without trusting a model to grade itself.
- Search for the largest context and prompt size that an Ollama or llama.cpp model completes within a timeout, and record time to first token and decode speed; see [AUTO_TUNER.md](AUTO_TUNER.md).

## Web UI and local runtimes

While a benchmark runs, a percentage progress bar displays finished trials out
of the total (parameter combinations × cases × repetitions). Both successful
and failed trials count once saved; the bar updates with the job status polling.
It measures completed work, not remaining time, and cannot report partial
progress within a single model invocation. A benchmark can reach 100% with
failed trials; its final status still reports those failures explicitly.

```sh
python3 harness_tester.py serve --port 8765
```

Open http://127.0.0.1:8765. Choose Ollama, llama.cpp, AirLLM, Colibri, or a custom command; specify the model and versioned harness identifier; edit fixed parameters, sweep arrays, repetitions, and cases. The **Load basic / JSON tests** button supplies arithmetic, JSON object, and JSON array checks. The embedding button supplies a retrieval case. Media inputs use the documented `parts` protocol.

For a running Ollama, llama.cpp, or Colibri server, the UI queries its model-list
endpoint and presents available IDs in a dropdown. Change the runtime or server
endpoint, or click **Refresh model list**, to reload it. Ollama uses `/api/tags`;
llama.cpp and Colibri use the OpenAI-compatible `/v1/models` endpoint. If the server
is unreachable, reports no models, or the runtime is AirLLM/custom, manual model
entry remains available. AirLLM is local-path based and has no server-side listing.

### Context sweep bounds and step

The **Context sweep range** controls generate a `context_size` array from a
minimum, maximum, and fixed step. For example, 2,048 through 8,192 with a 2,048
step produces `[2048, 4096, 6144, 8192]`. If the step does not reach the maximum
exactly, the maximum is appended as a final value. Other sweep parameters are
preserved; context aliases `context` and `num_ctx` in the grid are replaced.
Controls apply on change, or via **Apply context range to sweep**. You can still
edit the JSON directly.

Defaults refresh when the runtime, endpoint, or selected model changes, or when
you click **Use model/runtime defaults**. Ollama's `/api/show` advertised context
length supplies the upper bound. AirLLM reads local model `config.json` context
metadata; its context parameter is input truncation, not KV-cache sizing.
If metadata is unavailable, the UI explicitly labels its fallback estimate
(8,192 for Ollama/custom, 4,096 for AirLLM). The usual minimum and step are 2,048,
reduced for smaller advertised limits. An advertised context limit is **not a
memory-fit prediction**: large contexts can fail or be impractically slow on
the current machine. Set a smaller maximum before running if appropriate.

llama.cpp and Colibri adapters do not support per-request context changes, so
the range controls are disabled for them; vary their server configuration
between runs instead. The fixed step selects the context values tested, not
the execution order: trials remain shuffled as documented below. Generation
is limited to 1,000 context values to catch accidentally excessive ranges.
The context range is generation-only and is disabled for embedding suites.
For AirLLM, relative model paths resolve from the UI working directory, and a
fixed `model_path` parameter takes precedence over the model identifier.

### Runtime availability

The Runtime dropdown greys out and disables undetected harnesses. HTTP runtimes
are checked concurrently using their model-list endpoints with a two-second
timeout; an empty but valid model list still means the server is detected.
AirLLM checks for its `airllm` and `torch` dependencies in the UI's Python
environment, without importing them or loading models. Custom commands remain
available. Detection does not establish model compatibility or inference readiness.

Use **Refresh runtime detection** after starting a server or installing dependencies.
For non-default addresses, expand **Runtime discovery endpoints**, edit the
appropriate address, and refresh detection; you do not need to select a disabled
runtime first. The status text explains detection failures. If the selected
runtime becomes unavailable, the UI selects the first available runtime (or custom).

The HTTP UI binds only to loopback and protects its APIs with a per-process token and Host/Origin checks. It is a **trusted local-user tool**, not a multi-user hosted service: custom commands execute with your account's permissions and local file paths are accessible. Do not expose it through a public reverse proxy. It runs one benchmark at a time to avoid competing measurements; runs and individual trials remain in SQLite after a restart, while UI job handles do not. Keep the server open until active benchmarks finish.

Servers for Ollama and llama.cpp must already be running and have the target models available. The app does not install runtimes or automatically provision servers. See [RUNTIMES.md](RUNTIMES.md) for the supported runtime parameters, setup, and modality limitations. Runtime adapters reject unsupported capabilities rather than dropping inputs or ignoring parameters.

The working directory field resolves relative to where you launched the UI; browser media paths resolve relative to that selected directory. CLI media paths still resolve relative to the configuration file. For custom commands without built-in evaluation, the harness must return `score` as before.

### Hardware and suitability

```sh
python3 harness_tester.py machine
```

The UI displays CPU/architecture/kernel, total and available RAM, and GPU telemetry. NVIDIA VRAM and driver data use `nvidia-smi`, including the WSL executable when available. Linux DRM provides fallback device/driver information and AMD VRAM files where exposed. Missing telemetry is reported with warnings/null values, not interpreted as zero VRAM. WSL RAM reflects memory available to the WSL guest, not necessarily the host's physical RAM.

Available RAM and free VRAM are live readings and do not affect recommendation fingerprints. Stable GPU identity/capacity/driver fields do. Successful trials establish that a tested configuration ran under its measured conditions; the app **does not predict hardware fit from a model name** or claim that synthetic tests establish suitability. Inspect failures, latency, and quality when assessing suitability on this machine.

Under WSL, GPUs may not appear in Linux DRM at all. The inventory also queries Windows
`Win32_VideoController` through PowerShell to discover host AMD/Intel/NVIDIA adapters,
matching native GPU names one-for-one to avoid duplicate NVIDIA entries. Host-only
adapters are marked `visibility: "windows-host"` and
`wsl_inference_available: null`: detection does not establish ROCm, Vulkan, or other
runtime support inside WSL. Their `adapter_ram_reported_bytes` is the Windows
`AdapterRAM` value (for example, 2147483648 for a reported 2 GiB integrated GPU).
This 32-bit property can be truncated or otherwise inaccurate, and does not measure
shared system-memory capacity; `vram_total_bytes` and `vram_free_bytes` remain null
unless native telemetry supplies them. Host-discovery failures produce explicit
warnings. Restart an already-running UI server to load updated detection code.

## Generation quality checks

Add `evaluation` to generation cases:

```json
{
  "id": "json-output",
  "input": "Return only a JSON object with numeric answer 4.",
  "expected": {"answer": 4},
  "evaluation": {"metric": "json", "json_type": "object", "required_keys": ["answer"]}
}
```

For `metric: "json"`, the tester parses the **entire output string**, rejects markdown fences and non-finite numbers, optionally checks JSON type/required keys, and optionally compares parsed output to `expected`. It records `valid_json` separately from `json_requirements_met`; the latter is the ranking score. Malformed model output is a successful execution with quality zero, not a harness crash. This is a focused structural check, not a complete JSON Schema validator.

For `metric: "exact_match"`, `expected` must be a string, and leading/trailing whitespace is ignored. Scoring overrides any adapter-provided score; the adapter must return a string `output`. Generation without evaluation retains harness-supplied scoring. Mean generation metrics appear in summaries and comparisons. The basic suite measures a few smoke-test capabilities, not broad model reasoning or safety quality.

## Quick start

No installation is needed:

```sh
python3 harness_tester.py run examples/sweep.json
python3 harness_tester.py list
python3 harness_tester.py show RUN_ID
python3 harness_tester.py recommend RUN_ID
python3 harness_tester.py compare FIRST_RUN_ID SECOND_RUN_ID
```

The example is **synthetic**, not a model benchmark. Replace its command and scoring with your real harness. Optional installation: `pip install -e .`, then use `local-ai-harness-tester` instead of `python3 harness_tester.py`.

Every command prints JSON to stdout; progress and failures go to stderr. The default database is `results.sqlite3` in the caller's working directory. To choose a persistent location, put `--db /path/to/results.sqlite3` **before** the subcommand. Its parent directory must already exist. Back up the database using SQLite's backup facility or copy it when no writer is active. Results may include private prompts and model outputs; the database is ignored by Git.

## Configuration

```json
{
  "model": "my-model-exact-version-and-quantization",
  "harness": "my-harness-v1-scoring-v2",
  "command": ["python3", "adapter.py"],
  "cwd": ".",
  "cases": [
    {"id": "arithmetic", "input": "What is 2 + 2?", "expected": "4"}
  ],
  "parameters": {"context_size": 4096},
  "parameter_grid": {
    "temperature": [0, 0.3, 0.7],
    "threads": [4, 8]
  },
  "repetitions": 5,
  "timeout_seconds": 300,
  "seed": 42
}
```

`cwd` is relative to the configuration file, defaults to its directory, and is stored as an absolute path. Use absolute executable paths when needed. `command` is an argument array, not a shell string; shell expansion is not performed. Only run trusted commands.

Grid values override fixed parameters with the same name. Without a grid, fixed parameters are evaluated once per case/repetition. Defaults: three repetitions, a 300-second per-trial timeout, and seed zero. The example above makes 30 invocations (six settings x one case x five repetitions).

Trials run **sequentially** in shuffled order to reduce ordering bias and avoid benchmarks competing with each other. The shuffle is reproducible from `seed`. Each invocation is a fresh process; each request's seed is the configured seed plus its zero-based repetition. The adapter must apply the parameters and seed to the model/runtime itself.

## Command-harness protocol

Legacy text/JSON cases retain protocol version 1. Cases with an explicit `task` or structured `input.parts` use version 2 for the whole run. Version 2 adds the media and embedding conventions below; all other request fields and command execution behavior are unchanged.

The tester writes one JSON object to the command's stdin:

```json
{
  "protocol_version": 1,
  "model": "my-model-exact-version-and-quantization",
  "harness": "my-harness-v1-scoring-v2",
  "parameters": {"temperature": 0.3, "threads": 4, "context_size": 4096},
  "case": {"id": "arithmetic", "input": "What is 2 + 2?", "expected": "4"},
  "repetition": 0,
  "seed": 42
}
```

The command must exit zero and print **one JSON object only** to stdout:

```json
{"score": 1.0, "output_tokens": 12, "output": "4"}
```

- For generation cases, `score` is required: a finite number from 0 to 1, with higher meaning better quality. For embedding cases, the tester computes it.
- `output_tokens` is optional: a non-negative integer, using consistent token accounting.
- Other JSON fields are preserved, e.g. output text or scoring detail.
- Send diagnostics to stderr. Nonzero exit, timeout, invalid JSON, or invalid metrics are stored as failed trials and reported explicitly.

The adapter is responsible for running the selected model, applying model/harness parameters, and computing a meaningful generation quality score. For example, use exact match against `expected` for deterministic tasks, or a documented local rubric. This tester does not infer quality from latency or make external model calls.

Latency is measured wall-clock time for the **whole trial**, including process startup, model loading, generation, scoring, response validation, and media integrity checks. Throughput is total reported output tokens divided by total successful wall-clock seconds, and is absent when any successful response omits tokens. It is not streaming/decode-only throughput. To measure a warm server, have the command call your already-running local server; keep warm-up behavior consistent and version it in the harness identifier.

## Multimodal tests

Use `input.parts` to combine media and text in one case:

```json
{
  "id": "media-question",
  "task": "generation",
  "input": {
    "parts": [
      {"type": "text", "text": "Describe what happens and what is said."},
      {"type": "image", "path": "assets/photo.png", "mime_type": "image/png"},
      {"type": "audio", "path": "assets/speech.wav", "mime_type": "audio/wav"},
      {"type": "video", "path": "assets/clip.mp4", "mime_type": "video/mp4"}
    ]
  },
  "expected": "Your task-specific reference answer"
}
```

Media paths resolve relative to the **configuration file**, independent of harness `cwd`. The tester checks that files exist, passes absolute paths and SHA-256 content fingerprints to the adapter, and stores those fingerprints in the run configuration/scope. Changed file content makes runs non-comparable. Files are checked before and after each invocation; missing or changed assets during a run fail the affected trials. Keep assets immutable while benchmarking. Media bytes are not copied into SQLite: retain the source files if you need to reproduce a run.

Supported part types are `text`, `image`, `audio`, and `video`. A media part requires `path`; `mime_type` is optional metadata. The tester does not decode media, enforce codec compatibility, upload assets, or download URLs. The adapter must load and preprocess media in the format its local model expects, including video frame sampling and audio resampling. Generation output quality is still scored by the adapter; use the same persistence, sweeps, comparisons, and recommendation commands.

Try `python3 harness_tester.py run examples/multimodal_sweep.json` for a runnable **synthetic file-access example** using the included SVG. It verifies the protocol, not model vision quality. Actual models may require different image formats.

## Embedding tests

Use `task: "embedding"` with one of these built-in evaluations. Input can be ordinary text/JSON or multimodal `parts`, including retrieval document inputs.

### Reference-vector similarity

```json
{
  "id": "reference-embedding",
  "task": "embedding",
  "input": "A cat sits on a mat",
  "evaluation": {
    "metric": "cosine_similarity",
    "reference": [1.0, 0.0, 0.0]
  }
}
```

The adapter returns `{"embedding": [0.9, 0.1, 0.0]}`. Vectors must be non-empty, finite, nonzero numeric arrays of equal dimension; boolean elements are invalid. Cosine similarity is recorded in `embedding_metrics` on its natural [-1, 1] scale, and transformed to `score = (cosine + 1) / 2` for quality ranking. Supplied vectors are normalized for comparison; original returned vectors are preserved. Any adapter-supplied `score` is replaced by the built-in score.

References must use the **same embedding coordinate space and dimensions** as the tested model. Matching a reference vector measures agreement, not general semantic quality; retrieval tests are typically more meaningful when comparing models or embedding dimensions.

### Retrieval ranking

```json
{
  "id": "cat-retrieval",
  "task": "embedding",
  "input": "Find the cat",
  "documents": [
    {"id": "cat", "input": "A cat sits on a mat"},
    {"id": "boat", "input": "A boat sails on the sea"}
  ],
  "evaluation": {"metric": "retrieval", "relevant_ids": ["cat"], "k": 1}
}
```

The adapter embeds the query and all documents in the same space, returning:

```json
{
  "embedding": [1.0, 0.0],
  "document_embeddings": {"cat": [0.9, 0.1], "boat": [0.0, 1.0]}
}
```

Document IDs must match exactly; each document vector must match the query dimension and satisfy the vector validation rules. The tester ranks by descending cosine similarity, breaking ties lexically by document ID. It stores the ranking and similarities and records:

- `recall_at_k`: relevant documents in the top k divided by all relevant documents; this is the quality `score`.
- `reciprocal_rank`: reciprocal of the first relevant document's one-based rank across the full ranking.

`relevant_ids` must be non-empty, unique, and present in the documents. `k` defaults to 1 and must be between 1 and the document count. Document and evaluation content are included in the comparison scope, so changed relevance labels, reference vectors, tasks, or k do not share recommendations.

`show`, `compare`, and `recommend` report average embedding metrics for the successful trials that use each metric. `embeddings_per_second` is the total query/reference and document vector count divided by total successful command time, reported only when every successful trial is an embedding trial. A retrieval trial counts one query plus all documents. This is **end-to-end vector throughput**, not raw model encoding speed; it includes adapter startup and any document embedding/cache work. Keep caching policy consistent and versioned in the harness identifier.

Run `python3 harness_tester.py run examples/embedding_sweep.json` to try a **synthetic** embedding sweep with both evaluation types. It should recommend `{"variant": "aligned"}`. Substitute a real embedding adapter and actual model parameters for useful machine-specific recommendations.

Generation and embedding cases can share a suite. Overall quality remains the equal-weight mean across all cases/repetitions; do not mix different scoring rubrics unless that weighting reflects your objective. Use separate suites for separate task-specific parameter recommendations.

## Persistence, comparisons, and recommendations

`run` prints its ID on stderr immediately after saving it, commits each trial, and returns a nonzero exit code if any trial fails. Ctrl-C marks the run interrupted while preserving completed trials. Abrupt termination may leave a `running` record; incomplete or interrupted runs are excluded from recommendations. There is no automatic resumption.

`show` includes the stored run configuration (including resolved media paths and fingerprints), per-configuration means, sample standard deviations, counts, throughput, and every individual response/error. Every case/repetition has equal weight. A configuration is eligible only if every expected trial succeeded and its run finished. Failed configurations are not scored as zero or silently treated as successful.

`compare` can show any saved runs, even different models/suites/machines. Its `comparable` flag is true only when their scope fingerprints match. Differences in model, harness, command, working directory, case content, timeout, or machine prevent a fair like-for-like comparison.

`recommend RUN_ID` uses that run as the scope reference, verifies that its machine metadata matches the current environment, and considers all stored runs in that same scope. For each parameter set it uses the **latest complete successful evaluation**, rather than cherry-picking its best historical run. It ranks by descending mean score, then ascending mean latency; exact remaining ties use a stable parameter ordering. Different repetition counts are permitted and shown; use the same counts and seeds for controlled experiments.

Machine metadata includes hostname, OS/kernel, architecture, CPU, logical CPU count, Linux memory capacity, and Linux GPU PCI vendor/device IDs when available. This is a conservative identity, not a full environment inventory: it does not automatically track GPU drivers, runtime libraries, model file changes, environment variables, thermal conditions, or background load. Include runtime, model/quantization, driver, scorer, and adapter revisions in your model/harness identifiers when they affect results. Updating an adapter in place without changing its identifier cannot be detected.

Recommendations are the **best measured settings in the tested grid**, not a claim of a global optimum or statistical significance. Quality scoring must reflect your workload. Use enough repetitions, inspect variability, benchmark on an otherwise idle machine, and validate the selected settings on a separate held-out suite. No real-model recommendation is available until you run your harness on this machine.

## Tests

Install the test dependencies in a virtual environment and run pytest:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -v
```

The benchmark core itself still has no runtime dependencies. Pytest discovers and runs the existing unittest-based tests without converting them. You can also run them with the standard library:

```sh
python3 -m unittest discover -s tests -v
```

## Recommendation objectives and charts

In **Stored runs**, choose quality-first or lowest latency meeting a minimum
quality score. Set **Minimum quality** between 0 and 1, then click a run's
**Recommend** button. Only complete successful configurations in the same scope
are eligible; if none meet the threshold, the UI reports an error rather than
relaxing it. Latency ties are broken by higher quality. Quality-first remains
the default, and the threshold applies to either objective.

The CLI supports the same options:

```sh
python3 harness_tester.py recommend RUN_ID --objective latency --min-quality 0.8
```

Click **Show** on a run or **Compare selected runs** to view context-window versus
mean latency and mean quality charts. These recognize numeric `context_size`,
`context`, or `num_ctx` parameters (one alias per configuration). Only complete
successful configurations are plotted. Each point represents a configuration,
not an individual repetition; hover or focus it for parameters, run ID, and
measurement. The charts do not join points because other parameters can differ.
Different scopes in a comparison carry a warning; the charts do not establish
causality or statistical significance. Results without a context parameter show
an explanatory message instead of a chart.

## Quality checks

`requirements.txt` includes pytest, isort, Black, Pylint, and mypy. Configuration lives in `pyproject.toml`. Run the same checks as CI:

```sh
.venv/bin/python -m isort --check-only harness_tester.py evaluations.py hardware.py web_ui.py runtime_adapters.py auto_tuner.py tests examples
.venv/bin/python -m black --check harness_tester.py evaluations.py hardware.py web_ui.py runtime_adapters.py auto_tuner.py tests examples
.venv/bin/python -m pylint harness_tester.py evaluations.py hardware.py web_ui.py runtime_adapters.py auto_tuner.py
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
```

To apply formatting, omit `--check-only` for isort and `--check` for Black. CI runs on Python 3.10 and 3.12. Tests use synthetic commands and mocked runtime APIs; they do not download model weights or establish real-model compatibility/performance.
