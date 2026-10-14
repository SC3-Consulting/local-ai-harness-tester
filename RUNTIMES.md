# Runtime adapters

`runtime_adapters.py` is a standalone stdin/stdout harness command. It reads
one JSON request, makes one inference call, and writes exactly one JSON object
to stdout. Diagnostics go to stderr; adapter failures exit with status 2.
No runtime server is launched by the adapter.

```sh
python runtime_adapters.py --runtime ollama < request.json
python runtime_adapters.py --runtime llama.cpp --endpoint http://127.0.0.1:8080 \
  --timeout 120 < request.json
python runtime_adapters.py --runtime colibri \
  --endpoint http://127.0.0.1:8000/v1 < request.json
python runtime_adapters.py --runtime airllm < request.json
```

The harness command can be configured as
`[sys.executable, "/absolute/path/to/runtime_adapters.py", "--runtime",
"ollama", "--endpoint", "http://127.0.0.1:11434"]`. An endpoint is optional
for the HTTP runtimes; defaults are Ollama `http://127.0.0.1:11434`,
llama.cpp `http://127.0.0.1:8080`, and Colibri
`http://127.0.0.1:8000/v1`. Pass the service base URL or an OpenAI-style
`/v1` base URL; paths are joined without duplicating `/v1`. `--timeout`
defaults to 300 seconds. The exported `validate_endpoint(endpoint: str) -> str`
helper performs synchronous, no-network URL preflight and returns the
normalized HTTP(S) URL; it rejects credentials, query strings, and fragments.

## Request/response behavior

Requests use the harness JSON protocol: `model`, `harness`, `parameters`,
`case`, and optional `protocol_version`. The model is required by all adapters.
Generation results contain `output` and include `output_tokens` when the runtime
reports it. Generation cases must contain an explicit `evaluation` object;
missing evaluations fail before inference. Generation responses never fabricate
a score: the parent harness evaluates `exact_match` or `json`.
Embedding results contain `embedding`; retrieval results also contain
`document_embeddings` keyed by configured document IDs. The parent harness
computes embedding scores.

Plain string input and structured text parts are supported by all runtimes.
Ollama additionally accepts image parts on generation requests and sends their
local file bytes as base64 in the native `/api/chat` images field. Audio/video
input is explicitly rejected. llama.cpp and Colibri adapters currently accept
text only; all image/audio/video inputs are explicitly rejected. No adapter
silently drops unsupported media. Embedding inputs must be text.

## Runtime support

| Runtime | Generation API | Embedding API | Parameter mapping |
| --- | --- | --- | --- |
| Ollama | `POST /api/chat`, non-streaming | `POST /api/embed` | `context`/`num_ctx` → `num_ctx`; `temperature`/`temp` → `temperature`; `top_p`, `top_k`, `seed`, `max_tokens`/`num_predict`, `repeat_penalty`/`repetition_penalty`, and `stop` map to Ollama `options`. |
| llama.cpp | OpenAI-compatible `POST /v1/chat/completions` | OpenAI-compatible `POST /v1/embeddings` | `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `frequency_penalty`, and `presence_penalty` are sent as request fields. |
| Colibri | OpenAI-compatible `POST /v1/chat/completions` | Not provided by its documented API | `temperature`, `top_p`, `max_tokens`, and up to four `stop` sequences are sent as request fields. |
| AirLLM | In-process `AutoModel.from_pretrained(...).generate(...)` | Not provided by the adapter | `context`/`num_ctx` truncates tokenizer input; `max_tokens`/`max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `do_sample`, and `seed` map to supported Transformers generation/tokenizer behavior. |

`context_size` is also accepted as an alias for Ollama `num_ctx` and AirLLM
tokenizer context. Supplying multiple context aliases is rejected. When a
generation request provides a top-level `seed` and parameters do not explicitly
set `seed`, Ollama, llama.cpp, and AirLLM use that validated non-negative integer.
Explicit parameter seeds take precedence. Colibri does not apply top-level
seeds because its API cannot honor them.

All unknown or unsupported parameters fail explicitly instead of being ignored.
Generation-only options are rejected for embedding calls. Parameter values are
validated before inference. Colibri's `seed` is deliberately rejected because
its server documents it as accepted but ineffective.

### Ollama

Install and start Ollama separately, and make the selected model available
locally. The adapter uses native chat and embedding APIs. Image generation
requests use the image paths supplied by the harness. Ollama `/api/embed`
receives the query and any retrieval documents together, then returns the
query and per-document vectors in the harness format.

### llama.cpp

Start a llama.cpp server separately with its OpenAI-compatible endpoints and
embedding support enabled for the selected model. The adapter calls chat
completion and embeddings endpoints; server/model configuration, context size,
and media processing are outside this adapter. `context`, `top_k`, and
repeat-penalty controls are therefore rejected rather than approximated.

### AirLLM

AirLLM is optional and imported only when selected. Install AirLLM and its
framework/runtime dependencies in the Python environment used to invoke the
adapter. Set `request.model` to an existing local model directory, or set the
fixed harness parameter `model_path` to that directory while keeping `model`
as the benchmark's model label. Remote model IDs, hub downloads, embeddings,
and `--endpoint` are not supported. The adapter requires a local directory
and temporarily enables Hugging Face/Transformers offline mode during model
loading and inference so no hub download is attempted. AirLLM's normal local
layer preparation may still write model shards as described in its upstream
documentation.

### Colibri

Colibri is an HTTP client integration, not an executable/model launcher. Start
Colibri's `coli serve` separately and select its checkpoint/model when starting
that server (for example, using the upstream `--model`/`COLI_MODEL` setup).
Configure the adapter's `--endpoint` to the server's OpenAI base URL and use
the request's `model` field for the server model ID. The checkpoint and
executable are server-level configuration, not per-trial harness parameters;
the adapter intentionally does not start or reconfigure the server. The
documented Colibri API provides chat completions but no embeddings endpoint.

## Upstream API references

- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) —
  `/api/chat` and `/api/embed`.
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) —
  OpenAI-compatible chat and embedding endpoints.
- [AirLLM README](https://github.com/lyogavin/airllm#readme) — documented
  `AutoModel.from_pretrained`, tokenizer, `generate`, and local model usage.
- [Colibri API guide](https://github.com/JustVugg/colibri/blob/main/docs/api.md)
  and [Colibri README](https://github.com/JustVugg/colibri#readme) —
  `coli serve`, its local OpenAI API, request parameters, and model selection.
