# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real-weight GS32 integration; run after the native stage probe passes."""

import json
import time
from pathlib import Path

import pytest


def test_gs32_real_model_b1_i32_o4(request):
    options = {
        name: request.config.getoption(flag)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass --vq2a8-model and both --vq2a8-gs32-* native paths on Ascend950.")
    assert all(options.values()), "Both GS32 native artifact paths are required."
    paths = {name: str(Path(path).resolve(strict=True)) for name, path in options.items()}
    config = json.loads((Path(paths["model"]) / "config.json").read_text())
    assert config["quantization_config"]["format"] == "ascend_vq2a8_gs32"

    # Import only after the real-model opt-in, never emulate NPU execution.
    from vllm import LLM, SamplingParams

    model = paths.pop("model")
    execution = request.config.getoption("--vq2a8-gs32-execution")
    paths["gs32_execution"] = execution
    llm = LLM(
        model=model,
        quantization="vq2a8",
        dtype="bfloat16",
        load_format="safetensors",
        enforce_eager=True,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="uni",
        async_scheduling=False,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        block_size=128,
        kv_cache_memory_bytes=1024**3,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        generation_config="vllm",
        seed=0,
        additional_config={"vq2a8": paths, "multistream_dsv4_dsa_overlap": False},
    )
    tokens = llm.get_tokenizer().encode("请用中文简要介绍计算机的工作原理。", add_special_tokens=False)
    assert tokens
    prompt = {"prompt_token_ids": (tokens * (32 // len(tokens) + 1))[:32]}
    sampling = SamplingParams(temperature=0, max_tokens=4, min_tokens=4, ignore_eos=True, seed=0)
    previous = None
    for repeat in range(3):
        start = time.perf_counter()
        result = llm.generate([prompt], sampling, use_tqdm=False)[0]
        output = result.outputs[0]
        assert result.finished and len(result.prompt_token_ids) == 32
        assert len(output.token_ids) == 4 and output.text.strip()
        assert all(0 <= token < config["vocab_size"] for token in output.token_ids)
        if previous is not None:
            assert list(output.token_ids) == previous
        previous = list(output.token_ids)
        print(
            "VQ2A8_GS32_REAL_MODEL="
            + json.dumps(
                {
                    "repeat": repeat,
                    "execution": execution,
                    "input_tokens": 32,
                    "output_tokens": previous,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "text": output.text,
                    "scope": "generation smoke, not model quality or serving TPOT",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
