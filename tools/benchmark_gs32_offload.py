#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure exact input/output lengths through a local streaming completions API."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if not __package__:
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import requests  # noqa: E402


def run(args):
    # Ask the running engine to tokenize so prompt counting uses its exact tokenizer.
    response = requests.post(
        args.url.rstrip("/") + "/tokenize",
        json={
            "model": args.model,
            "prompt": "This is a long context inference test. The arithmetic answer is four.\n",
            "add_special_tokens": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    base = response.json()["tokens"]
    tokens = (base * ((args.input_tokens + len(base) - 1) // len(base)))[: args.input_tokens]
    if not tokens or args.output_tokens < 1:
        raise ValueError("Input and output token counts must be positive")
    payload = {
        "model": args.model,
        "prompt": tokens,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started, first, last = time.perf_counter(), None, None
    text, usage, finish = [], None, None
    with requests.post(
        args.url.rstrip("/") + "/v1/completions", json=payload, stream=True, timeout=(15, args.timeout)
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("text"):
                    last = time.perf_counter()
                    first = last if first is None else first
                    text.append(choice["text"])
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - started
    if usage is None or usage["prompt_tokens"] != args.input_tokens or usage["completion_tokens"] != args.output_tokens:
        raise RuntimeError(f"Exact token-count acceptance failed: {usage}")
    result = {
        "status": "PASS",
        "usage": usage,
        "finish_reason": finish,
        "elapsed_s": elapsed,
        "ttft_s": None if first is None else first - started,
        "tpot_s": None
        if first is None or args.output_tokens < 2
        else (elapsed - (first - started)) / (args.output_tokens - 1),
        # Empty text chunks (including filtered special tokens) still consume
        # generation time. Keep text-emission timing separately from full TPOT.
        "text_emission_tpot_s": None
        if first is None or last is None or args.output_tokens < 2
        else (last - first) / (args.output_tokens - 1),
        "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "output_text": "".join(text),
        "prefix_caching": "server configuration must disable prefix caching for uncached acceptance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "output_text"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18881")
    parser.add_argument("--model", default="vq2a8")
    parser.add_argument("--input-tokens", type=int, default=65536)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
