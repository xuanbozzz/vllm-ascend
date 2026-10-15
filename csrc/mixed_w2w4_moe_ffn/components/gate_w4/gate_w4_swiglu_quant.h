#ifndef MIXED_MOE_GATE_W4_SWIGLU_QUANT_H
#define MIXED_MOE_GATE_W4_SWIGLU_QUANT_H

#include "kernel_operator.h"

// 仅管理一块AIV私有Post UB；搬运、跨核事件和缓冲生命周期由融合调度器管理。
namespace FfnQuant {
constexpr uint32_t kMaxColumns = 192;
constexpr uint32_t kMaxRows = 128;
constexpr uint32_t kGroupSize = 32;
constexpr uint32_t kMaxGroupsPerRow = kMaxColumns / kGroupSize;
constexpr uint32_t kMaxGroupCount = kMaxRows * kMaxGroupsPerRow;
constexpr uint32_t kB16ElementsPerRepeat = 128;
constexpr uint32_t kQuantElementsPerRepeat = 256;
constexpr uint32_t kGroupsPerQuantRepeat =
    kQuantElementsPerRepeat / kGroupSize;
// Gate -> BF16 H原地复用；Up在SwiGLU全部读完后，前半复用为FP8输出，
// 后续字节复用为workspace。调用方保证workspace紧跟FP8输出且位于Up槽内。
// Scale和inverse都按连续GS32 group紧密保存。继续保留原4KiB槽位，
// 不改变workspace总大小及既有Host/tiling ABI。
constexpr uint32_t kScaleOffset = 0;
constexpr uint32_t kScaleStorageBytes = 4096;
constexpr uint32_t kMaxOffset = kScaleOffset + kScaleStorageBytes;
constexpr uint32_t kInverseOffset = kMaxOffset;
constexpr uint32_t kMaxStorageBytes = 4096;
constexpr uint32_t kRegisterLoadPadding = 256;
constexpr uint32_t kWorkspaceBytes =
    kMaxOffset + kMaxStorageBytes + kRegisterLoadPadding;
static_assert(kMaxGroupCount <= kScaleStorageBytes,
              "compact scale storage is too small");
static_assert(kMaxGroupCount * sizeof(uint16_t) <= kMaxStorageBytes,
              "compact inverse storage is too small");
static_assert(kWorkspaceBytes == 8448, "unexpected quant workspace size");

#ifdef __DAV_C310_VEC__
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr CastTrait kBf16ToFloat0 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait kBf16ToFloat1 = {
    RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait kFloatToBf16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kFloatToFp8_0 = {
    RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kFloatToFp8_1 = {
    RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kFloatToFp8_2 = {
    RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kFloatToFp8_3 = {
    RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__aicore__ inline void Run(__ubuf__ bfloat16_t *gate, __ubuf__ bfloat16_t *up,
                           __ubuf__ uint8_t *workspace, uint32_t rows,
                           uint32_t columns)
{
    if (rows == 0 || columns == 0) {
        return;
    }
    // 调用方保证 rows<=128、columns<=192 且 columns 是32的整数倍。
    // Gate/Up/H/FP8均为紧密ND：[rows,columns]，本函数只调用一次完成整个AIV tile。
    // workspace与原Up重叠是有意设计：第一个VEC_SCOPE消费完全部Up后，后续
    // V指令才会按序写入workspace和FP8输出，因此不会覆盖尚未读取的Up数据。
    __ubuf__ bfloat16_t *h = gate;
    __ubuf__ uint8_t *output = (__ubuf__ uint8_t *)up;
    __ubuf__ uint8_t *scale = workspace + kScaleOffset;
    __ubuf__ uint16_t *maxima = (__ubuf__ uint16_t *)(workspace + kMaxOffset);
    __ubuf__ bfloat16_t *inverse =
        (__ubuf__ bfloat16_t *)(workspace + kInverseOffset);
    const uint32_t element_count = rows * columns;
    const uint32_t total_groups = element_count / kGroupSize;

    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> gate16;
        RegTensor<bfloat16_t> up16;
        RegTensor<bfloat16_t> result16;
        RegTensor<float> gate32;
        RegTensor<float> up32;
        RegTensor<float> temporary;
        RegTensor<float> denominator;
        RegTensor<float> result32;
        RegTensor<float> zero;
        Duplicate(zero, 0.0F);
        const uint16_t repeat =
            static_cast<uint16_t>((element_count + 63) / 64);
        for (uint16_t index = 0; index < repeat; ++index) {
            const uint32_t offset = static_cast<uint32_t>(index) * 64;
            const uint32_t remaining = element_count - offset;
            uint32_t active_count = remaining < 64 ? remaining : 64;
            MaskReg active = UpdateMask<float>(active_count);
            MicroAPI::DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(
                gate16, gate + offset);
            MicroAPI::DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(
                up16, up + offset);
            Cast<float, bfloat16_t, kBf16ToFloat0>(gate32, gate16, active);
            Cast<float, bfloat16_t, kBf16ToFloat0>(up32, up16, active);
            // 基础 SwiGLU，不使用 clamp、top-k 权重或 Step 变体。
            ExpSub(temporary, zero, gate32, active);
            Adds(denominator, temporary, 1.0F, active);
            Div(result32, gate32, denominator, active);
            Mul(result32, result32, up32, active);
            Cast<bfloat16_t, float, kFloatToBf16>(result16, result32, active);
            MicroAPI::DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                h + offset, result16, active);
        }
    }

    __VEC_SCOPE__ {
        RegTensor<uint16_t> even;
        RegTensor<uint16_t> odd;
        RegTensor<uint16_t> pair_max;
        RegTensor<uint16_t> group_max;
        RegTensor<uint16_t> absolute_mask;
        UnalignReg unaligned_store;
        MaskReg all16 = CreateMask<uint16_t, MaskPattern::ALL>();
        Duplicate(absolute_mask, static_cast<uint16_t>(0x7FFF));
        __ubuf__ uint16_t *source = (__ubuf__ uint16_t *)h;
        __ubuf__ uint16_t *compact_maxima = maxima;
        const uint16_t repeat =
            static_cast<uint16_t>((element_count + kQuantElementsPerRepeat - 1) /
                                  kQuantElementsPerRepeat);
        for (uint16_t index = 0; index < repeat; ++index) {
            // DINTLV一次读取256个BF16并生成8个GS32最大值。amax紧密写入，
            // 不再按行丢弃第二行或后续行已经算出的group结果。
            MicroAPI::DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE,
                LoadDist::DIST_DINTLV_B16>(even, odd, source, 256);
            And(even, even, absolute_mask, all16);
            And(odd, odd, absolute_mask, all16);
            Max(pair_max, even, odd, all16);
            ReduceMaxWithDataBlock(group_max, pair_max, all16);
            const uint32_t processed_groups =
                static_cast<uint32_t>(index) * kGroupsPerQuantRepeat;
            const uint32_t remaining_groups =
                total_groups - processed_groups;
            const uint32_t store_groups =
                remaining_groups < kGroupsPerQuantRepeat ?
                    remaining_groups : kGroupsPerQuantRepeat;
            MicroAPI::DataCopyUnAlign<uint16_t,
                PostLiteral::POST_MODE_UPDATE>(
                compact_maxima, group_max, unaligned_store, store_groups);
        }
        MicroAPI::DataCopyUnAlignPost(compact_maxima, unaligned_store, 0);
    }

    __VEC_SCOPE__ {
        RegTensor<uint16_t> maximum;
        RegTensor<uint16_t> exponent;
        RegTensor<uint16_t> mantissa;
        RegTensor<uint16_t> code;
        RegTensor<uint16_t> incremented;
        RegTensor<uint16_t> inv_bits;
        RegTensor<uint16_t> zero;
        RegTensor<uint16_t> one;
        RegTensor<uint16_t> eight;
        RegTensor<uint16_t> nine;
        RegTensor<uint16_t> mantissa_mask;
        RegTensor<uint16_t> threshold;
        RegTensor<uint16_t> zero_code;
        RegTensor<uint16_t> reciprocal_bias;
        RegTensor<uint16_t> finite_max;
        RegTensor<uint16_t> nan_code;
        RegTensor<uint16_t> nan_bf16;
        RegTensor<uint8_t> packed_codes;
        MaskReg above_threshold;
        MaskReg low_exponent;
        MaskReg is_zero;
        MaskReg nonfinite;
        Duplicate(zero, static_cast<uint16_t>(0));
        Duplicate(one, static_cast<uint16_t>(1));
        Duplicate(eight, static_cast<uint16_t>(8));
        Duplicate(nine, static_cast<uint16_t>(9));
        Duplicate(mantissa_mask, static_cast<uint16_t>(0x7F));
        Duplicate(threshold, static_cast<uint16_t>(0x60));
        Duplicate(zero_code, static_cast<uint16_t>(127));
        Duplicate(reciprocal_bias, static_cast<uint16_t>(254));
        Duplicate(finite_max, static_cast<uint16_t>(0x7F7F));
        Duplicate(nan_code, static_cast<uint16_t>(255));
        Duplicate(nan_bf16, static_cast<uint16_t>(0x7FC0));
        const uint16_t repeat = static_cast<uint16_t>(
            (total_groups + kB16ElementsPerRepeat - 1) /
            kB16ElementsPerRepeat);
        for (uint16_t index = 0; index < repeat; ++index) {
            const uint32_t offset =
                static_cast<uint32_t>(index) * kB16ElementsPerRepeat;
            const uint32_t remaining = total_groups - offset;
            uint32_t active_count = remaining < kB16ElementsPerRepeat ?
                remaining : kB16ElementsPerRepeat;
            uint32_t code_count = active_count;
            MaskReg group_mask = UpdateMask<uint16_t>(active_count);
            MaskReg code_mask = UpdateMask<uint8_t>(code_count);
            MicroAPI::DataCopy(maximum, maxima + offset);
            ShiftRights(exponent, maximum, static_cast<int16_t>(7), group_mask);
            And(mantissa, maximum, mantissa_mask, group_mask);
            // 448=1.75*2^8。由 BF16 amax 的指数/尾数精确求
            // ceil_pow2(amax/448)。
            Sub(code, exponent, eight, group_mask);
            Add(incremented, code, one, group_mask);
            Compare<uint16_t, CMPMODE::GT>(
                above_threshold, mantissa, threshold, group_mask);
            Select(code, incremented, code, above_threshold);
            Compare<uint16_t, CMPMODE::LT>(
                low_exponent, exponent, nine, group_mask);
            Select(code, one, code, low_exponent);
            Compare<uint16_t, CMPMODE::EQ>(
                is_zero, maximum, zero, group_mask);
            Select(code, zero_code, code, is_zero);
            Compare<uint16_t, CMPMODE::GT>(
                nonfinite, maximum, finite_max, group_mask);
            Select(code, nan_code, code, nonfinite);
            Sub(inv_bits, reciprocal_bias, code, group_mask);
            ShiftLefts(inv_bits, inv_bits, static_cast<int16_t>(7), group_mask);
            Select(inv_bits, nan_bf16, inv_bits, nonfinite);
            // inverse原位覆盖compact amax；scale code同样连续紧密写入。
            MicroAPI::DataCopy(maxima + offset, inv_bits, group_mask);
            Pack(packed_codes, code);
            MicroAPI::DataCopy(scale + offset, packed_codes, code_mask);
        }
    }

    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> even;
        RegTensor<bfloat16_t> odd;
        RegTensor<bfloat16_t> inv;
        RegTensor<float> even0;
        RegTensor<float> even1;
        RegTensor<float> odd0;
        RegTensor<float> odd1;
        RegTensor<float8_e4m3_t> q0;
        RegTensor<float8_e4m3_t> q1;
        RegTensor<float8_e4m3_t> q2;
        RegTensor<float8_e4m3_t> q3;
        MaskReg all32 = CreateMask<float, MaskPattern::ALL>();
        MaskReg all16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg all8 = CreateMask<uint8_t, MaskPattern::ALL>();
        const uint16_t repeat =
            static_cast<uint16_t>((element_count + kQuantElementsPerRepeat - 1) /
                                  kQuantElementsPerRepeat);
        for (uint16_t index = 0; index < repeat; ++index) {
            const uint32_t input_offset =
                static_cast<uint32_t>(index) * kQuantElementsPerRepeat;
            const uint32_t remaining = element_count - input_offset;
            uint32_t output_count = remaining < kQuantElementsPerRepeat ?
                remaining : kQuantElementsPerRepeat;
            MaskReg output_mask = UpdateMask<uint8_t>(output_count);
            MicroAPI::DataCopy<bfloat16_t, LoadDist::DIST_DINTLV_B16>(
                even, odd, h + input_offset);
            MicroAPI::DataCopy<bfloat16_t, LoadDist::DIST_E2B_B16>(
                inv, inverse + static_cast<uint32_t>(index) *
                    kGroupsPerQuantRepeat);
            // 每轮连续处理256个BF16和对应8个inverse。每个inverse广播到
            // 一个GS32 data block，尾轮无效元素仅参与寄存器计算但不会写回。
            Mul(even, even, inv, all16);
            Mul(odd, odd, inv, all16);
            Cast<float, bfloat16_t, kBf16ToFloat0>(even0, even, all16);
            Cast<float, bfloat16_t, kBf16ToFloat1>(even1, even, all16);
            Cast<float, bfloat16_t, kBf16ToFloat0>(odd0, odd, all16);
            Cast<float, bfloat16_t, kBf16ToFloat1>(odd1, odd, all16);
            Cast<float8_e4m3_t, float, kFloatToFp8_0>(q0, even0, all32);
            Cast<float8_e4m3_t, float, kFloatToFp8_2>(q2, even1, all32);
            Cast<float8_e4m3_t, float, kFloatToFp8_1>(q1, odd0, all32);
            Cast<float8_e4m3_t, float, kFloatToFp8_3>(q3, odd1, all32);
            Or((RegTensor<uint8_t> &)q0, (RegTensor<uint8_t> &)q0,
               (RegTensor<uint8_t> &)q2, all8);
            Or((RegTensor<uint8_t> &)q1, (RegTensor<uint8_t> &)q1,
               (RegTensor<uint8_t> &)q3, all8);
            Or((RegTensor<uint8_t> &)q0, (RegTensor<uint8_t> &)q0,
               (RegTensor<uint8_t> &)q1, all8);
            MicroAPI::DataCopy<uint8_t, StoreDist::DIST_NORM_B8>(
                output + input_offset, (RegTensor<uint8_t> &)q0, output_mask);
        }
    }
}
#endif // __DAV_C310_VEC__
} // namespace FfnQuant
#endif // MIXED_MOE_GATE_W4_SWIGLU_QUANT_H
