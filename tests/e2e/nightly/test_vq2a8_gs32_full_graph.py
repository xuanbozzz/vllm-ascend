# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-model B1/I32/O4 eager-vs-full-graph acceptance, explicitly opted in.

Two child processes load the same unified checkpoint sequentially. Each runs
three requests with exactly 32 input and 4 output tokens. Named worker RPC
checks that every decode step replays the public full-model ACLGraphWrapper.
These correctness reports are not serving TTFT/TPOT measurements.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

EXECUTIONS = ("fused", "full_graph")
REQUEST_REPEATS = 3
INPUT_TOKENS = 32
OUTPUT_TOKENS = 4
LOGPROB_ATOL = 1e-3
CHILD_TIMEOUT_SECONDS = 1800
MODULE_NAME = "tests.e2e.nightly.test_vq2a8_gs32_full_graph"
GRAPH_STATE_METHOD = "vq2a8_gs32_full_graph_state"
WORKER_EXTENSION = f"{MODULE_NAME}.GS32GraphStateWorkerExtension"
REPO_ROOT = Path(__file__).resolve().parents[3]


class GS32GraphStateWorkerExtension:
    """Read primitive counters through named RPC without function pickling."""

    def vq2a8_gs32_full_graph_state(self):
        runner = self.model_runner
        entries = getattr(runner.model, "concrete_aclgraph_entries", {})
        owners = getattr(runner, "_quantization_validity_buffers", ())
        return {
            "captures": sum(entry.aclgraph is not None for entry in entries.values()),
            "replays": sum(entry.num_replays for entry in entries.values()),
            "validity_owners": len(owners),
        }


def runner_options(execution, library, binary):
    if execution not in EXECUTIONS:
        raise ValueError(f"Unsupported GS32 graph acceptance execution: {execution}")
    graph = execution == "full_graph"
    compilation = {"mode": 0, "cudagraph_mode": "NONE"}
    additional = {
        "vq2a8": {"gs32_execution": execution, "gs32_library": str(library), "gs32_binary": str(binary)},
        "multistream_dsv4_dsa_overlap": False,
        "enable_flashcomm1": False,
        "mix_placement": False,
    }
    if graph:
        compilation = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
            "max_cudagraph_capture_size": 1,
            "cudagraph_num_of_warmups": 1,
        }
        additional["ascend_compilation_config"] = {"enable_npugraph_ex": False}
    return {
        "quantization": "vq2a8",
        "dtype": "bfloat16",
        "load_format": "safetensors",
        "enforce_eager": not graph,
        "compilation_config": compilation,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "uni",
        "worker_extension_cls": WORKER_EXTENSION,
        "async_scheduling": False,
        "max_model_len": 128,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "block_size": 128,
        "kv_cache_memory_bytes": 1024**3,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "generation_config": "vllm",
        "seed": 0,
        "additional_config": additional,
    }


def graph_state(llm):
    responses = llm.collective_rpc(GRAPH_STATE_METHOD)
    assert len(responses) == 1, "B1 TP1 acceptance requires exactly one worker."
    return responses[0]


def check_graph_progress(execution, previous, current, layer_count):
    if execution == "full_graph":
        assert current["validity_owners"] == layer_count
        assert current["captures"] == previous["captures"] == 1, "Expected one persistent full-model B1 graph."
        assert current["replays"] - previous["replays"] >= OUTPUT_TOKENS - 1, (
            "Some decode tokens did not replay the full model graph."
        )
    else:
        assert current == previous == {"captures": 0, "replays": 0, "validity_owners": 0}


def compare_requests(actual, expected):
    assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
    assert actual["token_ids"] == expected["token_ids"], "Full-graph generation changed output tokens."
    assert len(actual["logprobs"]) == len(expected["logprobs"]) == OUTPUT_TOKENS
    for index, (observed, reference) in enumerate(zip(actual["logprobs"], expected["logprobs"], strict=True)):
        assert observed.keys() == reference.keys(), f"Top-logprob token set changed at output {index}."
        for token, want in reference.items():
            got = observed[token]
            assert math.isfinite(got) and math.isfinite(want)
            assert math.isclose(got, want, rel_tol=0, abs_tol=LOGPROB_ATOL), (
                f"Logprob changed at output {index}, token {token}: {got} vs {want}."
            )


def run_variant(model, library, binary, execution, output):
    # All vLLM/NPU initialization stays in the explicitly selected child.
    from vllm import SamplingParams

    from tests.e2e.conftest import VllmRunner

    config = json.loads((model / "config.json").read_text())
    report = {"execution": execution, "requests": [], "scope": "correctness; not serving TTFT/TPOT"}
    with VllmRunner(str(model), **runner_options(execution, library, binary)) as runner:
        llm = runner.model
        initial = graph_state(llm)
        report["graph_before"] = initial
        if execution == "full_graph":
            assert initial["captures"] == 1, "Full-model decode graph was not captured during initialization."
            assert initial["validity_owners"] == len(config["quantization_config"]["layers"])
        tokenizer = llm.get_tokenizer()
        seed_tokens = tokenizer.encode("请用中文简要介绍计算机的工作原理。", add_special_tokens=False)
        assert seed_tokens
        prompt = (seed_tokens * (INPUT_TOKENS // len(seed_tokens) + 1))[:INPUT_TOKENS]
        sampling = SamplingParams(
            temperature=0, max_tokens=OUTPUT_TOKENS, min_tokens=OUTPUT_TOKENS, ignore_eos=True, seed=0, logprobs=5
        )
        previous_graph = initial
        for repeat in range(REQUEST_REPEATS):
            results = llm.generate([{"prompt_token_ids": prompt}], sampling, use_tqdm=False)
            assert len(results) == 1 and results[0].finished and len(results[0].outputs) == 1
            assert len(results[0].prompt_token_ids) == INPUT_TOKENS
            completion = results[0].outputs[0]
            assert len(completion.token_ids) == OUTPUT_TOKENS and completion.logprobs is not None
            assert completion.text.strip()
            assert all(0 <= token < config["vocab_size"] for token in completion.token_ids)
            current_graph = graph_state(llm)
            check_graph_progress(execution, previous_graph, current_graph, len(config["quantization_config"]["layers"]))
            record = {
                "prompt_token_ids": list(results[0].prompt_token_ids),
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "logprobs": [
                    {str(token): value.logprob for token, value in row.items()} for row in completion.logprobs
                ],
                "graph_state": current_graph,
            }
            if repeat:
                compare_requests(record, report["requests"][0])
            report["requests"].append(record)
            previous_graph = current_graph
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            print(
                "VQ2A8_GS32_FULL_GRAPH="
                + json.dumps({"execution": execution, "repeat": repeat, **record}, ensure_ascii=False, allow_nan=False),
                flush=True,
            )
        report["graph_after"] = previous_graph
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def test_gs32_real_model_b1_i32_o4_full_graph_matches_eager(request, tmp_path):
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("library", "--vq2a8-gs32-library"),
            ("binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass a unified GS32 --vq2a8-model and both native artifacts to test real full-model graphs.")
    assert all(options.values()), "Both GS32 native artifact paths are required."
    paths = {name: Path(value).resolve(strict=True) for name, value in options.items()}
    assert paths["library"].is_file() and paths["binary"].is_file()
    config = json.loads((paths["model"] / "config.json").read_text())
    assert config["quantization_config"]["format"] == "ascend_vq2a8_gs32"
    assert (paths["model"] / "model.safetensors.index.json").is_file()
    reports = {}
    for execution in EXECUTIONS:
        output, log = tmp_path / f"{execution}.json", tmp_path / f"{execution}.log"
        command = [sys.executable, "-u", "-m", MODULE_NAME, "--execution", execution, "--output", str(output)]
        for name, path in paths.items():
            command.extend([f"--{name}", str(path)])
        with log.open("w") as stream:
            child = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=CHILD_TIMEOUT_SECONDS,
            )
        assert child.returncode == 0, f"{execution} failed; see {log}\n{log.read_text()[-16000:]}"
        reports[execution] = json.loads(output.read_text())
    assert len(reports["fused"]["requests"]) == len(reports["full_graph"]["requests"]) == REQUEST_REPEATS
    for actual, expected in zip(reports["full_graph"]["requests"], reports["fused"]["requests"], strict=True):
        compare_requests(actual, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--execution", required=True, choices=EXECUTIONS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_variant(args.model, args.library, args.binary, args.execution, args.output)
