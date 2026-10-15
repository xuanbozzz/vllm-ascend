# Exact MoE expert offload: layer-by-layer full-pool eager execution

This first stage supports single-worker DeepSeek V4 W4A8 MXFP, group size 32, with Model Runner V2 and eager execution. Expert offload is disabled by default.

## Configuration

Enable with `--additional-config '{"expert_offload_config":{"enabled":true}}'` and `--enforce-eager --no-async-scheduling`. TP/DP/PP/PCP/DCP sizes must all be one.

Run `bash examples/run_moe_expert_offload.sh /models/DeepSeek-V4-Flash-0731` on one idle NPU. The example reserves 2 GiB for KV and uses BF16 KV; compatible defaults can be overridden with additional arguments.

## Loading and execution

Routed experts retain the existing checkpoint loader and quantization scheme, load raw weights on CPU, validate all accepted expert/shard loads and prepare native weight/scale bytes in owned pinned host buffers. A single temporary expert supplies the NPU physical layout during loading; no per-layer resident slots are allocated. Raw CPU Parameters are retired after preparation, preserving their metadata. One raw prototype per component remains for pool allocation.

The runner adopts the prepared layers, checks homogeneous expert layouts and allocates one shared complete-layer NPU pool. This persistent allocation is included in model-memory accounting before KV budgeting.

Every prefill, chunked-prefill, mixed, decode and profiling execution transfers the current layer's complete expert bundle into that pool, then executes the existing quantized MoE with unchanged logical routing IDs and a matching full-expert dispatcher. Switching layers refills the same addresses; both GEMMs receive one consistent explicit weight view. All NPU copies use complete physical blocks, preserving NZ padding and packed scale strides. Batches are validated before submission and drained before errors propagate.

Execution is serialized. Transfer or forward failure invalidates the manager; shutdown drains and releases host and pool resources while preserving parent cleanup. The pool must fit one complete expert layer plus conversion/workspace transients. Decode copies all experts for every layer and is expected to be slower than the later resident-cache stage.

## Restrictions and validation

Graph execution, speculative decoding, EP/EPLB, async scheduling, DBO, LoRA, sleep, online reload, generic CPU weight offload, disaggregated execution, biased/sequence-parallel experts and heterogeneous pool layouts are rejected. Normal auto/safetensors/pt checkpoint loading is required.

CPU contracts cover loading completeness, full-bundle routing/GEMM consistency across phases and layer switches, pool reuse, lifecycle and injected failures. The one-card 950DT regression checks native full-pool refills. Run `pytest -q tests/ut/expert_offload` and `pytest -q tests/e2e/pull_request/one_card/expert_offload/test_expert_copy.py` in a matching Ascend/vLLM environment.

This stage is a review preview. Local isolated CPU tests do not establish NPU execution, whole-model accuracy or performance; staged full-runner and hardware validation remain required.
