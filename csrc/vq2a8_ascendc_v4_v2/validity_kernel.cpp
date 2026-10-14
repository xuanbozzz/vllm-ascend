// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "validity_launch.h"

namespace vq2a8_ascendc_v4_v2 {
using namespace AscendC;
constexpr uint32_t kValidityScalarBytes = 32;
constexpr uint32_t kValidityMaskBits = 32;
constexpr float kValidityFiniteMaximum = 3.4028234663852886e+38f;

template <HardEvent Event>
__aicore__ inline void ValidityFence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}

// Preserve the original v0.23 fused-checker predicate and single-writer
// schedule. Only the former combined-output scan is absent: this boundary
// belongs to the public FusedMoE expert projection stage, before combine.
class ProjectionValidityKernel {
 public:
  __aicore__ inline void Init(GM_ADDR result) {
    result_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(result));
    pipe_.InitBuffer(inputUb_, kProjectionValidityWidth * sizeof(bfloat16_t));
    pipe_.InitBuffer(floatUb_, kProjectionValidityWidth * sizeof(float));
    pipe_.InitBuffer(scratchUb_, kProjectionValidityWidth * sizeof(float));
    pipe_.InitBuffer(maskUb_, kProjectionValidityWidth / 8);
    pipe_.InitBuffer(resultUb_, kValidityScalarBytes);
    valid_ = true;
  }

  __aicore__ inline void CheckStatus(GM_ADDR pointer, uint32_t groups) {
    GlobalTensor<int32_t> status;
    status.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pointer));
    for (uint32_t row = 0; row < groups; ++row) valid_ = (status.GetValue(row) != 0) && valid_;
  }

  __aicore__ inline void CheckFlag(GM_ADDR pointer) {
    GlobalTensor<uint8_t> flag;
    flag.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(pointer));
    valid_ = (flag.GetValue(0) != 0) && valid_;
  }

  __aicore__ inline void CheckOutput(GM_ADDR pointer, uint32_t rows) {
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(pointer));
    const auto input = inputUb_.Get<bfloat16_t>();
    const auto widened = floatUb_.Get<float>();
    const auto scratch = scratchUb_.Get<float>();
    const auto mask = maskUb_.Get<uint8_t>();
    for (uint32_t row = 0; row < rows; ++row) {
      DataCopy(input, output[row * kProjectionValidityWidth], kProjectionValidityWidth);
      ValidityFence<HardEvent::MTE2_V>();
      // BF16 widening is exact; this changes no model arithmetic. NaNs fail
      // LE and both infinities exceed the largest finite FP32 value.
      Cast(widened, input, RoundMode::CAST_NONE, kProjectionValidityWidth);
      PipeBarrier<PIPE_V>();
      Abs(scratch, widened, kProjectionValidityWidth);
      PipeBarrier<PIPE_V>();
      Compares(mask, scratch, kValidityFiniteMaximum, CMPMODE::LE, kProjectionValidityWidth);
      ValidityFence<HardEvent::V_S>();
      uint32_t allBits = 0xffffffffU;
      for (uint32_t word = 0; word < kProjectionValidityWidth / kValidityMaskBits; ++word) {
        allBits &= maskUb_.Get<uint32_t>().GetValue(word);
      }
      valid_ = (allBits == 0xffffffffU) && valid_;
      ValidityFence<HardEvent::V_MTE2>();
    }
  }

  __aicore__ inline void Finish() {
    resultUb_.Get<uint8_t>().SetValue(0, valid_ ? 1 : 0);
    ValidityFence<HardEvent::S_MTE3>();
    const DataCopyExtParams scalar{1, sizeof(uint8_t), 0, 0, 0};
    DataCopyPad(result_, resultUb_.Get<uint8_t>(), scalar);
    ValidityFence<HardEvent::MTE3_S>();
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> inputUb_, floatUb_, scratchUb_, maskUb_, resultUb_;
  GlobalTensor<uint8_t> result_;
  bool valid_;
};
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_v4_v2_projection_validity(GM_ADDR s0, GM_ADDR s1, GM_ADDR s2, GM_ADDR s3,
                                                                      GM_ADDR s4, GM_ADDR s5, GM_ADDR gate,
                                                                      GM_ADDR down, GM_ADDR routeFlag, GM_ADDR result,
                                                                      uint32_t groups) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ProjectionValidityKernel op;
  op.Init(result);
  op.CheckStatus(s0, groups);
  op.CheckStatus(s1, groups);
  op.CheckStatus(s2, groups);
  op.CheckStatus(s3, groups);
  op.CheckStatus(s4, groups);
  op.CheckStatus(s5, groups);
  op.CheckOutput(gate, groups);
  op.CheckOutput(down, groups);
  op.CheckFlag(routeFlag);
  op.Finish();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchProjectionValidity(void* stream, void* const* statuses, void* const* outputs, void* routeFlag, void* result,
                              uint32_t groups) {
  vq2a8_v4_v2_projection_validity<<<1, nullptr, stream>>>(
      static_cast<GM_ADDR>(statuses[0]), static_cast<GM_ADDR>(statuses[1]), static_cast<GM_ADDR>(statuses[2]),
      static_cast<GM_ADDR>(statuses[3]), static_cast<GM_ADDR>(statuses[4]), static_cast<GM_ADDR>(statuses[5]),
      static_cast<GM_ADDR>(outputs[0]), static_cast<GM_ADDR>(outputs[1]), static_cast<GM_ADDR>(routeFlag),
      static_cast<GM_ADDR>(result), groups);
}
}  // namespace vq2a8_ascendc_v4_v2
