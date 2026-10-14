// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "activation_launch.h"

namespace vq2a8_ascendc_v4_v2 {
using namespace AscendC;
constexpr float kFiniteMaximum = 3.4028234663852886e+38f;
constexpr uint32_t kScalarBytes = 32;
constexpr uint32_t kMaskBitsPerWord = 32;

template <HardEvent Event>
__aicore__ inline void ActivationFence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}

// Compare writes packed bits, so all input checks need only K/32 scalar UB
// reads, not K scalar GM loads. NaN comparisons are false; +/-Inf exceed max.
__aicore__ inline void FiniteMask(const LocalTensor<float>& x, const LocalTensor<float>& scratch,
                                const LocalTensor<uint8_t>& mask, uint32_t width) {
  Abs(scratch, x, width);
  PipeBarrier<PIPE_V>();
  Compares(mask, scratch, kFiniteMaximum, CMPMODE::LE, width);
  PipeBarrier<PIPE_V>();
}

__aicore__ inline bool AllMask(const LocalTensor<uint32_t>& mask, uint32_t width) {
  ActivationFence<HardEvent::V_S>();
  uint32_t result = 0xffffffffU;
  for (uint32_t i = 0; i < width / kMaskBitsPerWord; ++i) result &= mask.GetValue(i);
  return result == 0xffffffffU;
}

template <typename InputT>
class ActivationSignKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR signs, GM_ADDR output,
                            GM_ADDR valid, uint32_t rows, uint32_t width, uint64_t rowStride) {
    rows_ = rows;
    width_ = width;
    rowStride_ = rowStride;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(x));
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    signs_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(signs));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    valid_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valid));
    pipe_.InitBuffer(xUb_, width * sizeof(float));
    if constexpr (sizeof(InputT) != sizeof(float)) {
      pipe_.InitBuffer(xInputUb_, width * sizeof(InputT));
    }
    pipe_.InitBuffer(scaleUb_, width * sizeof(float));
    pipe_.InitBuffer(biasUb_, width * sizeof(float));
    pipe_.InitBuffer(signUb_, width);
    pipe_.InitBuffer(halfUb_, width * sizeof(half));
    pipe_.InitBuffer(signFloatUb_, width * sizeof(float));
    pipe_.InitBuffer(scratchUb_, width * sizeof(float));
    pipe_.InitBuffer(maskUb_, width / 8);
    pipe_.InitBuffer(otherMaskUb_, width / 8);
    pipe_.InitBuffer(statusUb_, kScalarBytes);
  }

  __aicore__ inline void Process() {
    const auto x = xUb_.Get<float>();
    const auto scale = scaleUb_.Get<float>();
    const auto bias = biasUb_.Get<float>();
    const auto sign = signUb_.Get<int8_t>();
    const auto signFloat = signFloatUb_.Get<float>();
    const auto scratch = scratchUb_.Get<float>();
    const auto mask = maskUb_.Get<uint8_t>();
    const auto other = otherMaskUb_.Get<uint8_t>();
    for (uint32_t row = GetBlockIdx(); row < rows_; row += GetBlockNum()) {
      const uint64_t base = uint64_t(row) * width_;
      const uint64_t inputBase = uint64_t(row) * rowStride_;
      if constexpr (sizeof(InputT) == sizeof(float)) {
        DataCopy(x, x_[inputBase], width_);
      } else {
        DataCopy(xInputUb_.Get<InputT>(), x_[inputBase], width_);
      }
      DataCopy(scale, scale_[base], width_);
      DataCopy(bias, bias_[base], width_);
      DataCopy(sign, signs_[base], width_);
      ActivationFence<HardEvent::MTE2_V>();
      if constexpr (sizeof(InputT) != sizeof(float)) {
        // BF16 -> FP32 widens before the same sign multiplication/checks used
        // by the original FP32 path; no reduction or quantizer is changed.
        Cast(x, xInputUb_.Get<InputT>(), RoundMode::CAST_NONE, width_);
        PipeBarrier<PIPE_V>();
      }
      FiniteMask(x, scratch, mask, width_);
      FiniteMask(scale, scratch, other, width_);
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      PipeBarrier<PIPE_V>();
      FiniteMask(bias, scratch, other, width_);
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      PipeBarrier<PIPE_V>();
      // int8 -> half -> float is exact for every int8 value, including -128.
      Cast(halfUb_.Get<half>(), sign, RoundMode::CAST_NONE, width_);
      PipeBarrier<PIPE_V>();
      Cast(signFloat, halfUb_.Get<half>(), RoundMode::CAST_NONE, width_);
      PipeBarrier<PIPE_V>();
      Abs(scratch, signFloat, width_);
      PipeBarrier<PIPE_V>();
      Compares(other, scratch, 1.0f, CMPMODE::EQ, width_);
      PipeBarrier<PIPE_V>();
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      PipeBarrier<PIPE_V>();
      const bool valid = AllMask(maskUb_.Get<uint32_t>(), width_);
      Mul(x, x, signFloat, width_);
      ActivationFence<HardEvent::V_MTE3>();
      DataCopy(output_[base], x, width_);
      statusUb_.Get<int32_t>().SetValue(0, valid ? 1 : 0);
      ActivationFence<HardEvent::S_MTE3>();
      const DataCopyExtParams scalar{1, sizeof(int32_t), 0, 0, 0};
      DataCopyPad(valid_[row], statusUb_.Get<int32_t>(), scalar);
      ActivationFence<HardEvent::MTE3_S>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> xUb_, xInputUb_, scaleUb_, biasUb_, signUb_, halfUb_, signFloatUb_, scratchUb_;
  TBuf<TPosition::VECCALC> maskUb_, otherMaskUb_, statusUb_;
  GlobalTensor<InputT> x_;
  GlobalTensor<float> scale_, bias_, output_;
  GlobalTensor<int8_t> signs_;
  GlobalTensor<int32_t> valid_;
  uint32_t rows_, width_;
  uint64_t rowStride_;
};

using ActivationSignBf16Kernel = ActivationSignKernel<bfloat16_t>;

}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_v4_v2_activation_sign(
    GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR signs, GM_ADDR output, GM_ADDR valid,
    uint32_t rows, uint32_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ActivationSignKernel<float> op;
  op.Init(x, scale, bias, signs, output, valid, rows, width, width);
  op.Process();
}

extern "C" __global__ __aicore__ void vq2a8_v4_v2_activation_sign_strided_fp32(
    GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR signs, GM_ADDR output, GM_ADDR valid,
    uint32_t rows, uint32_t width, uint64_t rowStride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ActivationSignKernel<float> op;
  op.Init(x, scale, bias, signs, output, valid, rows, width, rowStride);
  op.Process();
}

extern "C" __global__ __aicore__ void vq2a8_v4_v2_activation_sign_strided_bf16(
    GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR signs, GM_ADDR output, GM_ADDR valid,
    uint32_t rows, uint32_t width, uint64_t rowStride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ActivationSignBf16Kernel op;
  op.Init(x, scale, bias, signs, output, valid, rows, width, rowStride);
  op.Process();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchActivationSign(void* stream, uint32_t blocks, void* x, void* weightScale, void* weightBias,
                          void* signs, void* output, void* valid, uint32_t rows, uint32_t width) {
  vq2a8_v4_v2_activation_sign<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(x), static_cast<GM_ADDR>(weightScale), static_cast<GM_ADDR>(weightBias),
      static_cast<GM_ADDR>(signs), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid), rows, width);
}

void LaunchActivationSignStrided(void* stream, uint32_t blocks, void* x, void* weightScale, void* weightBias,
                                 void* signs, void* output, void* valid, uint32_t rows, uint32_t width,
                                 uint64_t rowStride, bool inputIsBf16) {
  if (inputIsBf16) {
    vq2a8_v4_v2_activation_sign_strided_bf16<<<blocks, nullptr, stream>>>(
        static_cast<GM_ADDR>(x), static_cast<GM_ADDR>(weightScale), static_cast<GM_ADDR>(weightBias),
        static_cast<GM_ADDR>(signs), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid), rows, width, rowStride);
  } else {
    vq2a8_v4_v2_activation_sign_strided_fp32<<<blocks, nullptr, stream>>>(
        static_cast<GM_ADDR>(x), static_cast<GM_ADDR>(weightScale), static_cast<GM_ADDR>(weightBias),
        static_cast<GM_ADDR>(signs), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid), rows, width, rowStride);
  }
}

}  // namespace vq2a8_ascendc_v4_v2
