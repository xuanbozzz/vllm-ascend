# VQ2A8 with Model Runner V2, DSpark and expert offload

This integration combines the expert-offload branch at
`fe34a97da31f1a5ad218b039a9a6af8b1d99e32b` with the VQ increment from
`YChange01/vllm-ascend-vq2a8` at `5de0872d6447327cd5623fc173e4f216d2b9c2e7`.
The matching vLLM commit is recorded in `.github/vllm-main-verified.commit`
(`c4df37dfcc7f72ca41cd43801a468526ecf99a2b` for this base).
The older VQ fork's version pins do not replace this base's dependencies.

## Implementation scope

- Ascend950, BF16 model activations, one worker and synchronous scheduling.
- Target routed experts use the unified `ascend_vq2a8` version 1 checkpoint.
  Non-expert linear layers retain the VQ checkpoint's BF16 path.
- DSpark uses an explicitly selected original-format checkpoint, eager drafting,
  fixed verification and `max_num_seqs=1`. VQ draft experts are rejected.
- With expert offload enabled, every routed expert must use a supported adapter:
  VQ2A8 for the target and W4A8 MXFP/group size 32 for the original-format draft.
- The target verification block uses `FULL_DECODE_ONLY`, compilation mode NONE,
  and a capture candidate equal to `num_speculative_tokens + 1`.
- Device route rows are split into at most six rows per native VQ invocation.
  The split is fixed by the captured shape, including padded verification rows.
  Longer prefill batches use the grouped VQ path.
- V1 retains its original small-context VQ support; DSpark/VQ integration is V2 only.

VQ offload copies all twelve stacked payloads, including LUTs, activation orders,
scales, bias correction and RHT signs. Checkpoint logical IDs are distinct from
stored rows and device slots. Native bank pointers remain fixed while serialized
transfers replace the complete slot payload. Full-pool banks and resident banks
share one fresh validity flag per target layer. Callback or validity failures
reject execution output.

VQ pools are keyed by layout and kept separate from the existing MXFP pool.
Sparse hash-layer inventories can require additional layout pools. All pool and
validity allocations are included in model-memory accounting before KV budgeting.
Reducing the slot count does not eliminate full-pool, dense-weight or KV memory.

## Preparing the NPU environment

Install the matching vLLM source and build this repository's full Ascend950
extension in the selected CANN/torch-npu environment. The standalone VQ library
does not supply the full model's HC and attention operators. A stock v0.27.1rc1
vLLM wheel is not established as compatible with this main-based checkout.

## Serving candidate for 64K input and 1K output

```bash
python tools/serve_vq2a8.py \
  --model /models/dsv4-flash-0731-vq2a8 \
  --draft-model /models/dsv4-flash-0731-original \
  --num-speculative-tokens 5 \
  --num-device-experts 84 \
  --max-model-len 66560 \
  --prefill-chunk-size 2048 \
  --kv-cache-mib 2304 \
  --kv-cache-dtype fp8 \
  --dry-run
```

Remove `--dry-run` to load the models. The launcher selects MRV2, chunked prefill,
six-token verification capture and disabled prefix caching. Use `--prefix-caching`
for a separately recorded cached workload. The slot and KV budgets above are
starting candidates, not a measured W2 memory fit. The benchmark client must
request exactly 65536 input tokens and 1024 output tokens with early EOS disabled.
FP8 KV must be validated with the same numerical reference and used consistently
in the W4/W2 comparisons; select `--kv-cache-dtype auto` for the initial dtype check.

## Validation status and remaining gates

Local Windows checks use CPU Torch and isolated contract modules; they do not
validate native operators, complete vLLM integration, logits or throughput.
The NPU environment and checkpoint paths are pending. The following gates remain:

1. Full extension build and import, matching runner/quantization/DSpark UTs.
2. Native route-slot versus grouped dispatch/combine tests:
   `tests/e2e/pull_request/one_card/expert_offload/test_vq2a8_dispatch.py`.
3. Every-field VQ H2D, resident/full-pool bank outputs and capture/replay checks.
4. Same-checkpoint eager/full-pool versus resident/graph numerical comparisons,
   using tolerances fixed before comparison; short/long/short phase transitions.
5. DSpark draft format verification and mixed-format memory accounting.
6. Repeated 65536/1024 benchmarks with fixed prompts, KV configuration, DSpark
   settings and cache-hit policy. Record TTFT, TPOT, acceptance rate, H2D bytes,
   slot hits and allocated/reserved/physical memory peaks.
7. W4 regressions, offload-disabled regressions and `bash format.sh ci`.

No NPU accuracy, long-context compatibility or W2 performance gain is claimed
until those gates have run on the integrated sources.
