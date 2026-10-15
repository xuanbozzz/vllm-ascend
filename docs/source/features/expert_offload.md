# Exact MoE expert offload: V2 eager

Status: single-worker eager implementation. Asynchronous scheduling remains separate work.

## Pinned implementation baseline

| Input | Commit |
| --- | --- |
| vLLM Ascend main, locked on October 12, 2026 | `1c6ec402ec584f1eac0173b6318539128d74cf7c` |
| Matching vLLM, from `.github/vllm-main-verified.commit` | `8cc9aa5ad350f6f80cf3bebb298d410803ff75a9` |
| Committed feature reference | `6f0fc035c63fead688a262d1dee1dd6ec8dc03e6` |

The implementation was written from the committed feature behavior and the pinned main interfaces. It does not reuse the previous workspace's uncommitted adaptation.

## Configuration and restrictions

Expert offload is disabled by default. Enable the draft using the existing Ascend additional configuration:

```json
{
  "expert_offload_config": {
    "enabled": true,
    "num_device_experts": 32
  }
}
```

The slot count must satisfy `top_k <= num_device_experts < num_experts` for every target routed expert layer. Unknown keys, string booleans, non-integral slot counts, and invalid capacities fail early.

The implementation targets the DeepSeek V4 W4A8 MXFP model contract with group size 32. Acceptance uses the locally stored DeepSeek-V4-Flash-0731 checkpoint. Every routed expert layer must use a supported quantization adapter; shared and dense model weights follow their existing paths.

Required settings are V2, `async_scheduling=False`, and single-worker execution with TP/DP/PP/PCP/DCP sizes equal to one. Use `enforce_eager=True`. Speculative decoding is unsupported. Compiled/piecewise or mixed graphs, expert parallelism, EPLB, dual batch overlap, LoRA, sleep mode, online weight reload, generic CPU weight offload, and PD disaggregation are unsupported. Biased experts and sequence-parallel expert execution are also rejected. Use normal `auto`, `safetensors`, or `pt` checkpoint loading; dummy and preprocessed weight-cache loaders are unsupported.

## Serving example

Expose one idle NPU and install the pinned versions before running:

```bash
bash examples/run_moe_expert_offload.sh /models/DeepSeek-V4-Flash-0731
```

The script enables eager expert offload with 16 resident slots, a fixed 2 GiB KV budget and BF16 KV. Pass extra `vllm serve` arguments to override compatible example defaults. The pinned DeepSeek V4 path uses `--block-size 32`.

## Loading and execution

The routed-expert owner configures its quantization wrapper before weight creation. Enabled layers allocate raw expert Parameters on CPU and retain the original checkpoint loader. A wrapper records accepted expert/shard loads and checks that every weight and scale is present. Registered CPU Parameters own the checkpoint copies; the finalized pinned host store does not retain mmap-backed checkpoint views.

For enabled layers, the quantization wrapper disables upstream automatic device loading. It reuses the original W4A8 MXFP post-processing on a bounded set of resident experts. During loading, it also prepares every logical expert's native weight and scale storage on CPU, converting one expert at a time on NPU and retaining its physical bytes in owned pinned buffers. The complete raw CPU layer is then released, retaining one raw prototype per component for initial pool allocation. It preserves Parameter metadata while reducing the resident expert dimension. The V2 runner adopts these layer bindings after parent model loading, checks the shared pool layout, and allocates one full-expert prefill pool. The pool's persistent device allocation is included in `model_memory_usage` before KV-cache budgeting.

During decode, the planner protects selected resident experts and deterministically assigns misses to free or unselected slots. All four prepared tensor components are transferred together, without runtime post-processing or temporary device bundles. The map is published after transfer completion. A batched host route read replaces scalar device synchronization loops.

On NPU, transfers copy complete physical expert storage blocks directly from pinned CPU buffers into slots with asynchronous ACL host-to-device memcpy. Ordinary tensor slice copies cannot handle the NZ internal weight format on 950DT. The entire batch is validated before any submission, checking device, dtype, shape, strides, format, storage offsets, pinned ownership and capacities. Copies include NZ padding and preserve packed scale layouts. The caller's stream orders copies after previous consumers and is synchronized once per nonempty batch before map publication. Partially submitted batches are drained on failure. CPU tests retain ordinary tensor copies.

Prefill, mixed batches, profiling, and decode batches with more unique experts than slots use the reusable full-expert pool. Phase one requires homogeneous pool specifications, including activation and expert tensor shapes/dtypes. The full pool can consume one complete layer plus format-conversion transients; this implementation does not guarantee that an arbitrary model fits simply because its slot count is reduced.

The kernels receive an explicit weight view and a private all-gather dispatcher with the matching physical expert count. Both GEMMs see the same view. Original routing and prepare/finalize behavior are retained, and temporary communication context is restored in `finally`. Prefill does not rebind resident Parameters or change their addresses.

The manager is owned by the runner. V2 batch metadata determines prefill versus decode. Executions are serialized; a failed copy or forward invalidates the manager so stale or partially replaced experts cannot be used by another request. Worker shutdown drains and closes offload resources while preserving parent shutdown behavior.

## Tests and acceptance

The new CPU tests cover strict configuration, randomized residency conservation, loader completeness and ownership, bundle/view consistency, pool reuse, stable resident addresses, failure invalidation, and lifecycle composition. Hook tests execute isolated bodies from the actual runner, quantization wrapper, and routed-expert source; they do not substitute for importing and exercising the full matching vLLM runner.

In an installed Ascend/vLLM development environment:

```bash
pytest -q tests/ut/expert_offload
pytest -q tests/ut/worker/test_model_runner_v2.py
pytest -q tests/ut/quantization/test_method_adapters.py
pytest -q tests/e2e/pull_request/one_card/expert_offload/test_expert_copy.py
bash format.sh ci
```

The Windows development workspace has CPU Torch and lacks installed vLLM and Torch NPU. Its isolated test launcher bypasses the plugin package initializer and parent UT bootstrap to run dependency-light tests. Full runner UTs and the repository-wide CI command require the matching development environment.

The one-card 950DT tests use synthetic packed MXFP4 experts and real quantization post-processing and GEMM operators. They verify physical expert blocks, both GEMM outputs, resident addresses and full-pool refill addresses. They do not replace full-model checkpoint acceptance.


Prepared native H2D removes per-miss format conversion and temporary device-to-device copies. Asynchronous scheduling, LRC, prediction, MemFabric, EP, and prefill pipelining remain separate contributions.
