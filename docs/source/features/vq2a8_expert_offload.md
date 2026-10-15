# VQ2A8 with Model Runner V2, DSpark and expert offload

This integration combines the expert-offload branch at
`fe34a97da31f1a5ad218b039a9a6af8b1d99e32b` with the VQ increment from
`YChange01/vllm-ascend-vq2a8` at `5de0872d6447327cd5623fc173e4f216d2b9c2e7`.
The single-card GS32 implementation additionally imports native kernels and staged
GS32 payload handling from `a46cf123dc9fa14272fb441744982c04a91df35f` of that VQ fork.
The matching vLLM commit is recorded in `.github/vllm-main-verified.commit`
(`c4df37dfcc7f72ca41cd43801a468526ecf99a2b` for this base).
The older VQ fork's version pins do not replace this base's dependencies.

## Implementation scope

- Ascend950, BF16 model activations, one worker and synchronous scheduling.
- Target routed experts use a unified `ascend_vq2a8` (RHT) or
  `ascend_vq2a8_gs32` version 1 checkpoint, selected by checkpoint metadata.
  Non-expert linear layers retain the VQ checkpoint's BF16 path.
- DSpark uses an explicitly selected original-format checkpoint, eager drafting,
  fixed verification and `max_num_seqs=1`. VQ draft experts are rejected.
- With expert offload enabled, every routed expert must use a supported adapter:
  VQ2A8 for the target and W4A8 MXFP/group size 32 for the original-format draft.
- The target verification block uses `FULL_DECODE_ONLY`, compilation mode NONE,
  and a capture candidate equal to `num_speculative_tokens + 1`.
- RHT device route rows are split into at most six rows per native VQ invocation.
  The split is fixed by the captured shape, including padded verification rows.
  Longer prefill batches use the grouped VQ path.
- V1 retains its original small-context VQ support; DSpark/VQ integration is V2 only.

RHT offload copies all twelve stacked payloads, including LUTs, activation orders,
scales, bias correction and RHT signs. GS32 copies ten payloads: `packed_zn`,
`pair_lut`, `row_scale_bits`, `perm`, and `weight_scale` for both projections. Checkpoint logical IDs are distinct from
stored rows and device slots. Native bank pointers remain fixed while serialized
transfers replace the complete slot payload. Full-pool banks and resident banks
share one fresh validity flag per target layer. Callback or validity failures
reject execution output.

VQ pools are keyed by layout and kept separate from the existing MXFP pool.
Sparse hash-layer inventories can require additional layout pools. All pool and
validity allocations are included in model-memory accounting before KV budgeting.
Reducing the slot count does not eliminate full-pool, dense-weight or KV memory.
Large native host banks use pinned allocations of at most 64 MiB (or one expert,
if larger), with per-expert views retaining the original physical bytes. Raw
checkpoint copies remain pageable until native preparation. A shared 64 MiB
pinned transfer buffer is reserved before locking the banks. If the host pin
budget runs out, subsequent native bytes remain pageable and transfers stage
bounded chunks through that buffer, fencing before reusing it. The fast path
continues to copy pinned banks directly; both paths preserve native expert bytes
and fixed device addresses.

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

## Single-card GS32 with an original-format DSpark draft

Build the mixed kernel and its separate Torch bridge after building the base
Ascend extension. Both are needed: the GS32 bridge supplies projection operators,
while the base extension supplies HC, compressor and attention operators.

```bash
bash csrc/mixed_w2w4_moe_ffn/run_mixed_w2w4_moe_ffn_esl.sh
python tools/build_vq2a8_gs32_stage.py \
  --kernel-binary csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o \
  --cann /usr/local/Ascend/cann-9.2.0
```

The following is a validation candidate; final capacity and latency require a
successful model run. Expose only the authorized physical card and use its
container logical index zero. TP, EP, PP and DP remain one.

```bash
OMP_NUM_THREADS=1 python tools/serve_vq2a8.py \
  --model /models/DeepSeek-V4-Flash-0731-32x32-e4m3-r6diverse-bdnorm-unified \
  --draft-model /models/DeepSeek-V4-Flash-0731 \
  --num-speculative-tokens 5 \
  --gs32-library "$PWD/build/vq2a8-gs32-stage/libvq2a8_gs32_stage.so" \
  --gs32-binary "$PWD/build/vq2a8-gs32-stage/mixed_w2w4_moe_ffn.o" \
  --execution-mode decode_graph \
  --num-device-experts 84 --prefill-chunk-size 2048 \
  --max-model-len 66560 --kv-cache-dtype fp8 --kv-cache-mib 8192 --port 18881
```

The launcher explicitly selects the original draft's quantizer. DeepSeek-V4 FP8
checkpoints require the `deepseek_v4_fp8` backend to retain their MXFP4 experts.
The platform and speculative-config guards preserve the independent draft path
instead of allowing upstream DSpark initialization to substitute the VQ target.
GS32-exported FP32 norm weights remain FP32 when loaded. GS32 serving resolves
`--kv-cache-dtype auto` to FP8 explicitly to match the native A5 attention plan.

GS32 grouped device routing accepts up to 256 routed rows and at most 128 rows
per expert. The capture shape covers the fixed verification token count, including
padding. Larger prefill batches use the checked eager grouped path. The staged
native bank owns projection plans and captured intermediates; full and resident
banks share the runner's fresh validity flag. Current image Triton metadata
compilation fails for the imported route kernel, so the integrated route metadata
uses fixed-shape device tensor operations without host count reads.

For the GS32 single-card engine, native DSA and hidden-state caches use private,
dense per-layer regions within one backing allocation. The planner removes page
padding and recalculates capacity from the sum of the actual layer page sizes.
Compacting pages inside shared-tuple aliases is unsafe: different native strides
can overlap another group's live blocks, including target/draft caches. The
private layout keeps the configured byte budget and prevents that aliasing.
Raw-page KV transfer is incompatible with this layout and is rejected.

The streaming benchmark verifies counts returned by the running engine:

```bash
python tools/benchmark_gs32_offload.py \
  --input-tokens 65536 --output-tokens 1024 \
  --output /validation/logs/gs32-64k-1k.json
```

It disables early EOS for the request and records the prompt hash, TTFT and TPOT.
A successful exact-length run establishes execution and length support, not model
accuracy on a benchmark dataset. Compare numerical outputs separately using the
same checkpoint and predeclared tolerances.

## Validation status

Windows checks cover isolated contracts and source checks. NPU validation uses
Ascend950PR physical card 4, CANN 9.2, Torch 2.10 and torch-npu 2.10.post7.
The available development image has vLLM 0.30.0 and differs from the base's pinned
vLLM commit; results apply to that image and the integrated source tree.

The mixed GS32 kernel and bridge build and register successfully. Real-checkpoint
expert tests cover 1, 6 and 36 routed rows, repeated eviction/reload, all ten stable
payload addresses and bitwise equality of resident graph replay against the full
serial bank. The relevant unit regression has 1085 passes and four skips.
The original-format draft loads alongside the GS32 target, and the six-token
verification graph captures and replays in the full model. Forced pageable-host
staging also passes the nine real-expert eviction/replay comparisons.

On 2026-10-15, the single-card configuration above passed the following sequence
with prefix caching disabled and early EOS disabled:

| Input tokens | Output tokens | Elapsed seconds | TTFT seconds |
| ---: | ---: | ---: | ---: |
| 128 | 16 | 2.97 | 2.43 |
| 32768 | 16 | 71.77 | 71.31 |
| 65536 | 1024 | 169.08 | 143.68 |
| 128 | 16 | 2.88 | 2.39 |

The two short-request texts are identical. The server reports 96,717 effective
KV tokens within the explicit 8 GiB allocation; observed card-4 HBM use during
acceptance was about 49% of 128 GiB. Both model directories were mounted read-only.

Ruff, Python formatting, spelling, Markdown and repository Python source checks
pass. Full `bash format.sh ci` remains incomplete: installing the actionlint hook
environment fails with `RemoteDisconnected` while downloading dependencies.

These runs establish single-card execution and exact-length support for this
checkpoint and environment. They do not establish dataset quality, performance
stability across workloads, full-model eager/graph logit equivalence, or a speedup
over an original W4 target.
