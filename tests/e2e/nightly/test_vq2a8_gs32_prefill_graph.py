# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real B1/I32/O4 prefill + decode graph acceptance against eager execution.

Each child loads the same checkpoint, runs two different I32 prompts, exercises
I31/I33 eager-prefill fallback, then replays the original I32 prompt again.
Worker RPC reads actual full-model ACLGraphWrapper entries for shapes 32 and 1;
passing a decode-only graph or silently falling back to eager is a test failure.
This fixed, fresh-prefill test is not a serving TTFT/TPOT benchmark and does not
prove arbitrary prompt lengths, chunked prefill, or prefix-cache graph support.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_full_graph import (
    CHILD_TIMEOUT_SECONDS,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    REPO_ROOT,
    compare_requests,
)
from tests.e2e.nightly.test_vq2a8_gs32_full_graph import runner_options as decode_runner_options

EXECUTIONS = ("fused", "full_graph_prefill")
MODULE_NAME = "tests.e2e.nightly.test_vq2a8_gs32_prefill_graph"
GRAPH_STATE_METHOD = "vq2a8_gs32_prefill_graph_state"
WORKER_EXTENSION = f"{MODULE_NAME}.GS32PrefillGraphStateWorkerExtension"
REQUEST_REPEATS = 5
PROMPT_TEXTS = ("请用中文简要介绍计算机的工作原理。", "请解释植物如何利用阳光制造养分。")


class GS32PrefillGraphStateWorkerExtension:
    """Separate real prefill/decode captures and replays using public entries."""

    def vq2a8_gs32_prefill_graph_state(self):
        runner = self.model_runner
        state = {name: {"captures": 0, "replays": 0} for name in ("prefill", "decode", "other")}
        entries = getattr(runner.model, "concrete_aclgraph_entries", {})
        for entry in entries.values():
            if entry.aclgraph is None:
                continue
            tokens = entry.batch_descriptor.num_tokens
            name = "prefill" if tokens == INPUT_TOKENS else "decode" if tokens == 1 else "other"
            state[name]["captures"] += 1
            state[name]["replays"] += entry.num_replays
        state["validity_owners"] = len(getattr(runner, "_quantization_validity_buffers", ()))
        return state


def runner_options(execution, library, binary):
    if execution not in EXECUTIONS:
        raise ValueError(f"Unsupported GS32 prefill graph execution: {execution}")
    graph = execution == "full_graph_prefill"
    options = decode_runner_options("full_graph" if graph else "fused", library, binary)
    options["worker_extension_cls"] = WORKER_EXTENSION
    options["additional_config"]["vq2a8"]["gs32_execution"] = execution
    if graph:
        options["compilation_config"].update(
            cudagraph_mode="FULL", cudagraph_capture_sizes=[1, INPUT_TOKENS], max_cudagraph_capture_size=INPUT_TOKENS
        )
    return options


def graph_state(llm):
    responses = llm.collective_rpc(GRAPH_STATE_METHOD)
    assert len(responses) == 1, "B1 TP1 prefill acceptance requires exactly one worker."
    return responses[0]


def check_graph_progress(execution, previous, current, layer_count, *, input_tokens=INPUT_TOKENS):
    if execution == "full_graph_prefill":
        assert previous["validity_owners"] == current["validity_owners"] == layer_count
        for phase, minimum_replays in (("prefill", 1), ("decode", OUTPUT_TOKENS - 1)):
            assert previous[phase]["captures"] == current[phase]["captures"] == 1, (
                f"Expected one persistent full-model {phase} graph."
            )
            replays = current[phase]["replays"] - previous[phase]["replays"]
            if phase == "prefill" and input_tokens != INPUT_TOKENS:
                assert replays == 0, "Unsupported prefill length replayed the fixed I32 graph."
            else:
                assert replays >= minimum_replays, f"Request did not replay its full-model {phase} graph."
        assert current["other"] == previous["other"] == {"captures": 0, "replays": 0}
    else:
        empty = {name: {"captures": 0, "replays": 0} for name in ("prefill", "decode", "other")}
        empty["validity_owners"] = 0
        assert current == previous == empty


def build_prompts(tokenizer):
    prompts = []
    for text in PROMPT_TEXTS:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert tokens, "Acceptance prompt must tokenize to a nonempty sequence."
        prompts.append((tokens * ((INPUT_TOKENS + 1) // len(tokens) + 1))[: INPUT_TOKENS + 1])
    first, second = prompts[0][:INPUT_TOKENS], prompts[1][:INPUT_TOKENS]
    assert first != second, "Prefill replay must consume different input tokens."
    return [first, second, first[:-1], prompts[0], first]


def run_variant(model, library, binary, execution, output):
    # vLLM/NPU imports stay in the explicitly opted-in child process.
    from vllm import SamplingParams

    from tests.e2e.conftest import VllmRunner

    config = json.loads((model / "config.json").read_text())
    layer_count = len(config["quantization_config"]["layers"])
    report = {"execution": execution, "requests": [], "scope": "B1 fresh I32 graph correctness; not serving TTFT/TPOT"}
    with VllmRunner(str(model), **runner_options(execution, library, binary)) as runner:
        llm = runner.model
        previous = graph_state(llm)
        report["graph_before"] = previous
        if execution == "full_graph_prefill":
            assert previous["prefill"]["captures"] == previous["decode"]["captures"] == 1, (
                "Both prefill and decode graphs must be captured during startup."
            )
            assert previous["validity_owners"] == layer_count
        prompts = build_prompts(llm.get_tokenizer())
        sampling = SamplingParams(
            temperature=0, max_tokens=OUTPUT_TOKENS, min_tokens=OUTPUT_TOKENS, ignore_eos=True, seed=0, logprobs=5
        )
        for repeat, prompt in enumerate(prompts):
            results = llm.generate([{"prompt_token_ids": prompt}], sampling, use_tqdm=False)
            assert len(results) == 1 and results[0].finished and len(results[0].outputs) == 1
            assert list(results[0].prompt_token_ids) == prompt
            completion = results[0].outputs[0]
            assert len(completion.token_ids) == OUTPUT_TOKENS and completion.logprobs is not None
            assert completion.text.strip()
            assert all(0 <= token < config["vocab_size"] for token in completion.token_ids)
            current = graph_state(llm)
            check_graph_progress(execution, previous, current, layer_count, input_tokens=len(prompt))
            record = {
                "prompt_token_ids": list(results[0].prompt_token_ids),
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "logprobs": [
                    {str(token): value.logprob for token, value in row.items()} for row in completion.logprobs
                ],
                "graph_state": current,
            }
            if repeat == REQUEST_REPEATS - 1:
                compare_requests(record, report["requests"][0])
            report["requests"].append(record)
            previous = current
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            print(
                "VQ2A8_GS32_PREFILL_GRAPH="
                + json.dumps({"execution": execution, "repeat": repeat, **record}, ensure_ascii=False, allow_nan=False),
                flush=True,
            )
        report["graph_after"] = previous
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def test_gs32_real_model_b1_i32_o4_prefill_graph_matches_eager(request, tmp_path):
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("library", "--vq2a8-gs32-library"),
            ("binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass a unified GS32 --vq2a8-model and both native artifacts for prefill graph acceptance.")
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
    assert len(reports["fused"]["requests"]) == len(reports["full_graph_prefill"]["requests"]) == REQUEST_REPEATS
    for actual, expected in zip(reports["full_graph_prefill"]["requests"], reports["fused"]["requests"], strict=True):
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
