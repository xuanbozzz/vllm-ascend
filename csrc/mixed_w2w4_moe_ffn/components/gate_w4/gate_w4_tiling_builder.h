#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gate_w4_post_tiling.h"
#include "gate_w4_tiling.h"

namespace {

static_assert(sizeof(MatFP4TilingData) == 15 * sizeof(uint64_t), "unexpected tiling data layout");
static_assert(kMatFP4W4LutOffset == 128, "W4 LUT offset must remain ABI-compatible");

constexpr uint64_t kKAlignment = 64;
constexpr uint64_t kZnK0 = 16;
constexpr uint64_t kZnN0 = 32;
constexpr uint64_t kMxScaleGroupSize = 32;
// N是Gate+Up总宽；每个N64包含一对Gate32/Up32。
constexpr uint64_t kNAlignment = 2 * kZnN0;
constexpr uint64_t kMAlignment = 16;
constexpr uint64_t kGlobalWideMMax = kMatFP4WideTileM;
constexpr uint64_t kAivKCandidates[] = {512, 256, 128, 64};
constexpr uint64_t kMadKCandidates[] = {256, 128, 64};
constexpr uint64_t kUbCapacity = 256 * 1024;
constexpr uint64_t kL1Capacity = 512 * 1024;
constexpr uint64_t kL0ACapacity = 64 * 1024;
constexpr uint64_t kL0BCapacity = 64 * 1024;
constexpr uint64_t kL0CCapacity = 256 * 1024;

// Performance knob: update this Host-side table after profiling a new N shape.
// N values not listed here use kDefaultWideMStart.
struct WideMStartConfig {
    uint64_t n;
    uint64_t wide_m_start;
};

constexpr uint64_t kDefaultWideMStart = 96;
constexpr WideMStartConfig kWideMStartConfigs[] = {
    {3584, 96},
    {4096, 96},
    {4608, 81},
    {5120, 81},
    {6144, 65},
};

static_assert(kL1Capacity == 2 * kMatFP4L1SlotBytes, "L1 slot layout must cover all L1 memory");
uint64_t CheckedProduct(std::initializer_list<uint64_t> values)
{
    uint64_t result = 1;
    for (uint64_t value : values) {
        if (value != 0 && result > std::numeric_limits<uint64_t>::max() / value) {
            throw std::overflow_error("tensor byte size overflow");
        }
        result *= value;
    }
    return result;
}

uint64_t CheckedSum(std::initializer_list<uint64_t> values)
{
    uint64_t result = 0;
    for (uint64_t value : values) {
        if (result > std::numeric_limits<uint64_t>::max() - value) {
            throw std::overflow_error("tensor byte size overflow");
        }
        result += value;
    }
    return result;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        throw std::logic_error("division by zero while building tiling");
    }
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        throw std::logic_error("zero alignment while building tiling");
    }
    return value / alignment * alignment;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        throw std::logic_error("zero alignment while building tiling");
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : CheckedSum({value, alignment - remainder});
}

uint64_t ComputeNBlock(uint64_t n, uint64_t n_lanes)
{
    return AlignUp(CeilDiv(n, n_lanes), kNAlignment);
}

uint64_t GetWideMStart(uint64_t n)
{
    for (const WideMStartConfig &config : kWideMStartConfigs) {
        if (config.n == n) {
            return config.wide_m_start;
        }
    }
    return kDefaultWideMStart;
}

uint64_t AivUbBytes(uint64_t nbs, uint64_t aiv_kbs)
{
    const uint64_t packed_bytes = CheckedProduct({nbs, aiv_kbs}) / 2;
    const uint64_t decoded_bytes = CheckedProduct({nbs, aiv_kbs, kZnK0 + 1}) / kZnK0;
    const uint64_t input_bytes = CheckedProduct({kMatFP4W4AivInputBufferCount, packed_bytes});
    const uint64_t output_bytes = CheckedProduct({kMatFP4AivOutputBufferCount, decoded_bytes});
    return CheckedSum({input_bytes, output_bytes, kMatFP4W4LutBytes});
}

uint64_t PostUbOffset(uint64_t dequant_bytes)
{
    return AlignUp(dequant_bytes, 32);
}

uint64_t PostUbBytes(uint64_t dequant_bytes)
{
    const uint64_t offset = PostUbOffset(dequant_bytes);
    return offset <= kUbCapacity ? kUbCapacity - offset : 0;
}

uint64_t L1SlotBytes(uint64_t m, uint64_t n, uint64_t k)
{
    const uint64_t data_bytes = CheckedProduct({m + n, k});
    const uint64_t scale_bytes = CheckedProduct({m + n, k}) / kMxScaleGroupSize;
    return CheckedSum({data_bytes, scale_bytes});
}

uint64_t L1SmallBytes(const MatFP4TilingData &tiling)
{
    return CheckedProduct({uint64_t {2}, L1SlotBytes(tiling.small_m_limit, tiling.narrow_nbs,
                                                     tiling.small_aic_kbs)});
}

uint64_t L1WideBytes(const MatFP4TilingData &tiling)
{
    return CheckedProduct({uint64_t {2}, L1SlotBytes(tiling.wide_m_limit, tiling.wide_nbs,
                                                     tiling.middle_aic_kbs)});
}

uint64_t L1Bytes(const MatFP4TilingData &tiling)
{
    return std::max(L1SmallBytes(tiling), L1WideBytes(tiling));
}

uint64_t L0ABytes(const MatFP4TilingData &tiling)
{
    const uint64_t tile_bytes = std::max(CheckedProduct({tiling.small_m_limit, tiling.small_mad_k}),
                                         CheckedProduct({kMatFP4MaxTileM, tiling.middle_mad_k}));
    return CheckedProduct({uint64_t {2}, tile_bytes});
}

uint64_t L0BBytes(const MatFP4TilingData &tiling)
{
    const uint64_t tile_bytes = std::max(CheckedProduct({tiling.narrow_nbs, tiling.small_mad_k}),
                                         CheckedProduct({tiling.wide_nbs, tiling.middle_mad_k}));
    return CheckedProduct({uint64_t {2}, tile_bytes});
}

uint64_t L0CBytes(const MatFP4TilingData &tiling)
{
    const uint64_t narrow_bytes = CheckedProduct(
        {uint64_t {2}, tiling.double_l0c_m, tiling.narrow_nbs, sizeof(float)});
    const uint64_t wide_bytes = CheckedProduct({tiling.wide_m_limit, tiling.wide_nbs, sizeof(float)});
    return std::max(narrow_bytes, wide_bytes);
}

uint64_t AivUbMaxBytes(const MatFP4TilingData &tiling)
{
    return std::max(AivUbBytes(tiling.narrow_nbs, tiling.small_aiv_kbs),
                    AivUbBytes(tiling.wide_nbs, tiling.middle_aiv_kbs));
}

bool IsValidKSchedule(uint64_t k, uint64_t aiv_kbs, uint64_t aic_kbs, uint64_t mad_k)
{
    if (aic_kbs != 2 * aiv_kbs || aic_kbs % mad_k != 0 || aic_kbs % kKAlignment != 0 ||
        aiv_kbs % kZnK0 != 0) {
        return false;
    }
    const uint64_t tail_k = k % aic_kbs;
    return tail_k == 0 ||
        (tail_k % kKAlignment == 0 && tail_k % mad_k == 0 && (tail_k / 2) % kZnK0 == 0);
}

struct MatFP4KSchedule {
    uint64_t aiv_k;
    uint64_t aic_k;
    uint64_t mad_k;
};

struct MatFP4WideSchedule {
    MatFP4KSchedule k;
    uint64_t m_limit;
};

uint64_t ComputeWideMLimit(uint64_t nbs, uint64_t aic_k, uint64_t mad_k)
{
    const uint64_t l1_bytes_per_row = CheckedSum({aic_k, aic_k / kMxScaleGroupSize});
    const uint64_t b1_bytes = CheckedProduct({nbs, l1_bytes_per_row});
    if (b1_bytes >= kMatFP4L1SlotBytes) {
        return 0;
    }
    const uint64_t l1_limit = AlignDown((kMatFP4L1SlotBytes - b1_bytes) / l1_bytes_per_row,
                                        kMAlignment);
    const uint64_t l0a_limit = AlignDown(kL0ACapacity / CheckedProduct({uint64_t {2}, mad_k}),
                                         kMAlignment);
    const uint64_t l0c_limit = AlignDown(
        kL0CCapacity / CheckedProduct({nbs, uint64_t {sizeof(float)}}), kMAlignment);
    return std::min({kGlobalWideMMax, l1_limit, l0a_limit, l0c_limit});
}

MatFP4KSchedule SelectSmallKSchedule(uint64_t k, uint64_t nbs, uint64_t m_limit)
{
    for (uint64_t aiv_k : kAivKCandidates) {
        const uint64_t aic_k = 2 * aiv_k;
        const uint64_t dequant_bytes = AivUbBytes(nbs, aiv_k);
        const uint64_t post_bytes = PostUbBytes(dequant_bytes);
        if (dequant_bytes > kUbCapacity ||
            !FusedPostFits(m_limit, nbs / 2, post_bytes) ||
            L1SlotBytes(m_limit, nbs, aic_k) > kMatFP4L1SlotBytes) {
            continue;
        }
        for (uint64_t mad_k : kMadKCandidates) {
            if (!IsValidKSchedule(k, aiv_k, aic_k, mad_k) ||
                CheckedProduct({uint64_t {2}, nbs, mad_k}) > kL0BCapacity ||
                CheckedProduct({uint64_t {2}, m_limit, mad_k}) > kL0ACapacity) {
                continue;
            }
            return {aiv_k, aic_k, mad_k};
        }
    }
    throw std::invalid_argument("no legal W4 Small AIV/AIC/MMAD K schedule for this K/N/threshold");
}

MatFP4WideSchedule SelectWideSchedule(uint64_t k, uint64_t nbs,
                                      uint64_t small_nbs, uint64_t small_m_limit,
                                      uint64_t small_dequant_bytes)
{
    for (uint64_t aiv_k : kAivKCandidates) {
        const uint64_t aic_k = 2 * aiv_k;
        const uint64_t wide_dequant_bytes = AivUbBytes(nbs, aiv_k);
        const uint64_t dequant_bytes = std::max(small_dequant_bytes, wide_dequant_bytes);
        const uint64_t post_bytes = PostUbBytes(dequant_bytes);
        if (dequant_bytes > kUbCapacity ||
            !FusedPostFits(small_m_limit, small_nbs / 2, post_bytes)) {
            continue;
        }
        for (uint64_t mad_k : kMadKCandidates) {
            if (!IsValidKSchedule(k, aiv_k, aic_k, mad_k) ||
                CheckedProduct({uint64_t {2}, nbs, mad_k}) > kL0BCapacity ||
                CheckedProduct({uint64_t {2}, kMatFP4MaxTileM, mad_k}) > kL0ACapacity) {
                continue;
            }
            const uint64_t hardware_m_limit = ComputeWideMLimit(nbs, aic_k, mad_k);
            const uint64_t post_m_limit = FusedPostMLimit(nbs / 2, post_bytes);
            const uint64_t m_limit = std::min(hardware_m_limit, post_m_limit);
            if (m_limit >= kMAlignment && FusedPostFits(m_limit, nbs / 2, post_bytes)) {
                return {{aiv_k, aic_k, mad_k}, m_limit};
            }
        }
    }
    throw std::invalid_argument("no legal W4 Wide AIV/AIC/MMAD K schedule for this K/N");
}

uint64_t ComputeDoubleL0CM(uint64_t small_m_limit, uint64_t nbs, uint64_t small_aic_k)
{
    if (small_aic_k != 1024) {
        return small_m_limit;
    }
    const uint64_t physical_limit = AlignDown(
        kL0CCapacity / CheckedProduct({uint64_t {2}, nbs, uint64_t {sizeof(float)}}), kMAlignment);
    return std::min(kMatFP4MaxTileM, physical_limit);
}

MatFP4TilingData BuildTiling(uint64_t k, uint64_t n)
{
    if (k == 0 || k % kKAlignment != 0) {
        throw std::invalid_argument("K must be a positive multiple of 64");
    }
    if (n == 0 || n % kNAlignment != 0) {
        throw std::invalid_argument("total Gate+Up width N must be a positive multiple of 64");
    }
    const uint64_t wide_m_start = GetWideMStart(n);
    if (wide_m_start <= kMAlignment || wide_m_start > kMatFP4MaxTileM + 1) {
        throw std::logic_error("Host-side wide_m_start must be in [17, 257]");
    }

    MatFP4TilingData tiling {};
    tiling.narrow_nbs = ComputeNBlock(n, kMatFP4NarrowNCoreCount);
    tiling.wide_nbs = ComputeNBlock(n, kMatFP4WideNCoreCount);
    tiling.small_m_limit = AlignUp(wide_m_start - 1, kMAlignment);
    tiling.n_only_mode = 0;
    tiling.wide_m_start = wide_m_start;

    const MatFP4KSchedule small = SelectSmallKSchedule(k, tiling.narrow_nbs, tiling.small_m_limit);
    const uint64_t small_dequant_bytes = AivUbBytes(tiling.narrow_nbs, small.aiv_k);
    const MatFP4WideSchedule wide = SelectWideSchedule(
        k, tiling.wide_nbs, tiling.narrow_nbs, tiling.small_m_limit,
        small_dequant_bytes);
    tiling.small_aiv_kbs = small.aiv_k;
    tiling.small_aic_kbs = small.aic_k;
    tiling.small_mad_k = small.mad_k;
    tiling.middle_aiv_kbs = wide.k.aiv_k;
    tiling.middle_aic_kbs = wide.k.aic_k;
    tiling.middle_mad_k = wide.k.mad_k;
    tiling.wide_m_limit = wide.m_limit;
    tiling.double_l0c_m = ComputeDoubleL0CM(tiling.small_m_limit, tiling.narrow_nbs,
                                            tiling.small_aic_kbs);
    const uint64_t dequant_bytes = std::max(
        small_dequant_bytes, AivUbBytes(tiling.wide_nbs, tiling.middle_aiv_kbs));
    tiling.post_ub_offset = PostUbOffset(dequant_bytes);
    tiling.post_ub_bytes = PostUbBytes(dequant_bytes);

    if (tiling.narrow_nbs % kNAlignment != 0 || tiling.wide_nbs % kNAlignment != 0 ||
        MatFP4NTileCount(n, tiling.narrow_nbs) > kMatFP4NarrowNCoreCount ||
        MatFP4NTileCount(n, tiling.wide_nbs) > kMatFP4WideNCoreCount) {
        throw std::logic_error("automatic paired-N tiling does not fit the available N lanes");
    }
    const MatFP4NTileDesc narrow_tail = MatFP4GetNTile(
        n, tiling.narrow_nbs, MatFP4NTileCount(n, tiling.narrow_nbs) - 1);
    const MatFP4NTileDesc wide_tail = MatFP4GetNTile(
        n, tiling.wide_nbs, MatFP4NTileCount(n, tiling.wide_nbs) - 1);
    if (narrow_tail.extent % kNAlignment != 0 || wide_tail.extent % kNAlignment != 0) {
        throw std::logic_error("automatic N tail is not Gate32/Up32-pair aligned");
    }
    if (tiling.narrow_nbs / 2 > kFusedColumns ||
        tiling.wide_nbs / 2 > kFusedColumns) {
        throw std::logic_error("automatic N tiling exceeds the fused hidden-N limit");
    }
    if (tiling.double_l0c_m < tiling.small_m_limit) {
        throw std::logic_error("automatic Small L0C buffer is smaller than the Small M capacity");
    }
    if (AivUbMaxBytes(tiling) > tiling.post_ub_offset ||
        tiling.post_ub_offset + tiling.post_ub_bytes != kUbCapacity ||
        tiling.post_ub_offset % 32 != 0 ||
        !FusedPostFits(tiling.small_m_limit, tiling.narrow_nbs / 2,
                       tiling.post_ub_bytes) ||
        !FusedPostFits(tiling.wide_m_limit, tiling.wide_nbs / 2,
                       tiling.post_ub_bytes)) {
        throw std::logic_error("automatic W4 dequant/Post tiling exceeds 256 KiB UB capacity");
    }
    if (L1Bytes(tiling) > kL1Capacity) {
        throw std::logic_error("automatic W4 tiling exceeds 512 KiB L1 capacity");
    }
    if (L0ABytes(tiling) > kL0ACapacity) {
        throw std::logic_error("automatic W4 tiling exceeds 64 KiB L0A capacity");
    }
    if (L0BBytes(tiling) > kL0BCapacity) {
        throw std::logic_error("automatic W4 tiling exceeds 64 KiB L0B capacity");
    }
    if (L0CBytes(tiling) > kL0CCapacity) {
        throw std::logic_error("automatic W4 tiling exceeds 256 KiB L0C capacity");
    }
    return tiling;
}

std::array<uint8_t, kMatFP4W4TilingBufferBytes> BuildTilingBuffer(const MatFP4TilingData &tiling)
{
    // E2M1 codes 0..15 converted directly to equal-valued E4M3 bit patterns.
    constexpr uint8_t kFp4ToFp8[16] = {
        0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
        0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC
    };
    std::array<uint8_t, kMatFP4W4TilingBufferBytes> buffer {};
    std::memcpy(buffer.data(), &tiling, sizeof(tiling));
    for (uint64_t packed = 0; packed < kMatFP4W4LutHalfBytes; ++packed)
    {
        buffer[kMatFP4W4LutOffset + packed] = kFp4ToFp8[packed & 0x0F];
        buffer[kMatFP4W4LutOffset + kMatFP4W4LutHalfBytes + packed] = kFp4ToFp8[packed >> 4];
    }
    return buffer;
}

} // namespace

inline std::vector<uint8_t> BuildGateW4TilingBlob(uint64_t k, uint64_t n)
{
    const MatFP4TilingData tiling = BuildTiling(k, n);
    const auto buffer = BuildTilingBuffer(tiling);
    return std::vector<uint8_t>(buffer.begin(), buffer.end());
}
