"""Build Moby Dick summary sweeps for the harness (excerpt sizes x context windows).

Usage (WSL bash, from the repo root):
  python examples/moby_dick_sweep.py fetch
  python examples/moby_dick_sweep.py ollama --model llama3.2:1b --ctx 16384 --sizes 1000,2000,4000,8000
  python examples/moby_dick_sweep.py ollama --model llama3.2:1b --sizes 4000 --ctx-sweep 8192,6144,4096,3072
  python examples/moby_dick_sweep.py llama.cpp --model local --ctx 8192 --sizes 1000,2000,4000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE / "moby_dick.txt"
URL = "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
ADAPTER = HERE.parent / "runtime_adapters.py"
TOKENS_PER_WORD = 1.35  # rough English estimate; real counts depend on the tokenizer
PROMPT = (
    "Summarise the following excerpt from Moby-Dick in 2-3 sentences. "
    'Reply with ONLY a raw JSON object of the form {"summary": "<text>"} '
    "and no markdown fences or other text.\n\nEXCERPT:\n"
)


def fetch() -> None:
    raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8-sig")
    start = raw.find("Call me Ishmael")  # skips the table of contents
    end = raw.find("*** END OF")
    if start < 0 or end < 0:
        sys.exit("Unexpected Gutenberg layout; trim the text manually into moby_dick.txt")
    BOOK.write_text(raw[start:end], encoding="utf-8")
    print(f"wrote {BOOK} ({BOOK.stat().st_size} bytes)")


def excerpt(words: list[str], tokens: int) -> str:
    return " ".join(words[: int(tokens / TOKENS_PER_WORD)])


def build(args: argparse.Namespace) -> None:
    if not BOOK.is_file():
        sys.exit("Run the fetch step first")
    words = BOOK.read_text(encoding="utf-8").split()
    sizes = [int(item) for item in args.sizes.split(",")]
    cases = [
        {
            "id": f"moby-{size}tok",
            "input": PROMPT + excerpt(words, size),
            "evaluation": {"metric": "json", "json_type": "object", "required_keys": ["summary"]},
        }
        for size in sizes
    ]
    grid: dict[str, list] = {"temperature": [0], "max_tokens": [200]}
    if args.runtime == "ollama":
        contexts = [int(item) for item in (args.ctx_sweep or str(args.ctx)).split(",")]
        grid["context_size"] = contexts
    command = [
        sys.executable,
        str(ADAPTER),
        "--runtime",
        args.runtime,
        "--timeout",
        str(args.timeout),
    ]
    if args.endpoint:
        command += ["--endpoint", args.endpoint]
    config = {
        "model": args.model,
        "harness": "moby-dick-summary-v1",
        "command": command,
        "cases": cases,
        "parameter_grid": grid,
        "repetitions": args.repetitions,
        "timeout_seconds": args.timeout + 60,
        "seed": 42,
    }
    out = HERE / args.out
    out.write_text(json.dumps(config), encoding="utf-8")
    print(f"wrote {out}: {len(cases)} cases x grid {grid}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=["fetch", "ollama", "llama.cpp"])
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--sizes", default="1000,2000,4000,8000", help="approx. excerpt tokens")
    parser.add_argument("--ctx", type=int, default=16384, help="ollama num_ctx for the size sweep")
    parser.add_argument("--ctx-sweep", help="comma list of contexts to shrink through (ollama)")
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--out", default="moby_sweep.json")
    args = parser.parse_args()
    fetch() if args.runtime == "fetch" else build(args)


if __name__ == "__main__":
    main()
