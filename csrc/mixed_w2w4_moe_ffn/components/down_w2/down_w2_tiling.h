#ifndef MIXED_MOE_DOWN_W2_TILING_H
#define MIXED_MOE_DOWN_W2_TILING_H

#include <cstdint>

// Host 统一计算后传给 AIC/AIV，避免两侧 tiling 不一致。
struct MatFP4TilingData {
    uint64_t narrow_nbs;       // Small: ceil(N/32 cores)，再按 N32 对齐。
    uint64_t wide_nbs;         // Wide: ceil(N/16 lanes)，再按 N32 对齐。
    uint64_t small_m_limit;    // AlignUp(wide_m_start-1, 16)。
    uint64_t n_only_mode;      // 兼容字段；自动 tiling 固定为 0。
    uint64_t small_aiv_kbs;
    uint64_t small_aic_kbs;
    uint64_t small_mad_k;
    uint64_t middle_aiv_kbs;   // ABI兼容命名；自动 tiling 中表示 Wide AIV K。
    uint64_t middle_aic_kbs;   // ABI兼容命名；自动 tiling 中表示 Wide AIC K。
    uint64_t middle_mad_k;     // ABI兼容命名；自动 tiling 中表示 Wide MMAD K。
    uint64_t wide_m_start;  // 当前 expert 的 M >= 此值时使用 2x16 分核。
    uint64_t wide_m_limit;  // 宽档核内 M tile 容量，按16对齐。
    uint64_t double_l0c_m;  // Small 档每个 L0C buffer 的 M 容量。
};

constexpr uint64_t kMatFP4MaxTileM = 256;
constexpr uint64_t kMatFP4WideTileM = 240;
constexpr uint64_t kMatFP4NarrowNCoreCount = 32;
constexpr uint64_t kMatFP4WideNCoreCount = 16;
constexpr uint64_t kMatFP4L1SlotBytes = 256 * 1024;

enum MatFP4TileMode : uint64_t {
    MAT_FP4_SMALL = 0,
    MAT_FP4_MIDDLE = 1,
    MAT_FP4_WIDE = 2,
};

struct MatFP4TileDesc {
    uint64_t offset;
    uint64_t rows;
    uint64_t mode;
};

struct MatFP4KTileDesc {
    uint64_t offset;
    uint64_t extent;
};

struct MatFP4NTileDesc {
    uint64_t offset;
    uint64_t extent;
};

#ifdef __aicore__
#define MAT_FP4_INLINE __aicore__ inline
#else
#define MAT_FP4_INLINE inline
#endif

MAT_FP4_INLINE bool MatFP4IsWideGroup(uint64_t group_m, uint64_t n_only_mode, uint64_t wide_m_start)
{
    return n_only_mode == 0 && group_m >= wide_m_start;
}

MAT_FP4_INLINE uint64_t MatFP4NTileCount(uint64_t total_n, uint64_t base_n)
{
    return (total_n + base_n - 1) / base_n;
}

MAT_FP4_INLINE MatFP4NTileDesc MatFP4GetNTile(uint64_t total_n, uint64_t base_n, uint64_t tile_id)
{
    const uint64_t offset = tile_id * base_n;
    const uint64_t remaining = total_n - offset;
    return {offset, remaining < base_n ? remaining : base_n};
}

MAT_FP4_INLINE uint64_t MatFP4KTileCount(uint64_t total_k, uint64_t base_k)
{
    return (total_k + base_k - 1) / base_k;
}

MAT_FP4_INLINE MatFP4KTileDesc MatFP4GetKTile(uint64_t total_k, uint64_t base_k, uint64_t tile_id)
{
    const uint64_t offset = tile_id * base_k;
    const uint64_t remaining = total_k - offset;
    return {offset, remaining < base_k ? remaining : base_k};
}

// 将 total 切成接近一半的两块，其中第一块取离 total/2 最近的 16 倍数，距离相同时取较小值。
MAT_FP4_INLINE uint64_t MatFP4SplitNearHalf16(uint64_t total)
{
    const uint64_t lower = total / 32 * 16;
    const uint64_t upper = lower + 16;
    const uint64_t lower_twice = lower * 2;
    const uint64_t upper_twice = upper * 2;
    const uint64_t lower_distance = total >= lower_twice ? total - lower_twice : lower_twice - total;
    const uint64_t upper_distance = total >= upper_twice ? total - upper_twice : upper_twice - total;
    return lower_distance <= upper_distance ? lower : upper;
}

MAT_FP4_INLINE uint64_t MatFP4NarrowTileCount(uint64_t group_m)
{
    return group_m <= kMatFP4MaxTileM ? 1 : 2;
}

MAT_FP4_INLINE MatFP4TileDesc MatFP4GetNarrowTile(uint64_t group_m, uint64_t tile_id, uint64_t small_m_limit)
{
    if (group_m <= kMatFP4MaxTileM) {
        const uint64_t mode = group_m <= small_m_limit ? MAT_FP4_SMALL : MAT_FP4_MIDDLE;
        return {0, group_m, mode};
    }
    const uint64_t first_m = MatFP4SplitNearHalf16(group_m);
    const uint64_t offset = tile_id == 0 ? 0 : first_m;
    const uint64_t rows = tile_id == 0 ? first_m : group_m - first_m;
    const uint64_t mode = rows <= small_m_limit ? MAT_FP4_SMALL : MAT_FP4_MIDDLE;
    return {offset, rows, mode};
}

// 仅沿 N 轴分核时，保留 q-1 个 M256；剩余 256+r 在 r>0 时靠近一半切开。
MAT_FP4_INLINE uint64_t MatFP4NOnlyTileCount(uint64_t group_m)
{
    if (group_m == 0) return 0;
    if (group_m <= kMatFP4MaxTileM) return 1;
    const uint64_t quotient = group_m / kMatFP4MaxTileM;
    const uint64_t remainder = group_m % kMatFP4MaxTileM;
    return quotient - 1 + (remainder == 0 ? 1 : 2);
}

MAT_FP4_INLINE MatFP4TileDesc MatFP4GetNOnlyTile(uint64_t group_m, uint64_t tile_id, uint64_t small_m_limit)
{
    if (group_m <= kMatFP4MaxTileM) {
        const uint64_t mode = group_m <= small_m_limit ? MAT_FP4_SMALL : MAT_FP4_MIDDLE;
        return {0, group_m, mode};
    }

    const uint64_t quotient = group_m / kMatFP4MaxTileM;
    const uint64_t remainder = group_m % kMatFP4MaxTileM;
    const uint64_t full_tile_count = quotient - 1;
    if (tile_id < full_tile_count) {
        return {tile_id * kMatFP4MaxTileM, kMatFP4MaxTileM, MAT_FP4_MIDDLE};
    }

    const uint64_t tail_offset = full_tile_count * kMatFP4MaxTileM;
    const uint64_t tail_m = group_m - tail_offset;
    if (remainder == 0) return {tail_offset, tail_m, MAT_FP4_MIDDLE};

    const uint64_t first_m = MatFP4SplitNearHalf16(tail_m);
    const uint64_t offset = tile_id == full_tile_count ? tail_offset : tail_offset + first_m;
    const uint64_t rows = tile_id == full_tile_count ? first_m : tail_m - first_m;
    const uint64_t mode = rows <= small_m_limit ? MAT_FP4_SMALL : MAT_FP4_MIDDLE;
    return {offset, rows, mode};
}

// 宽档先按 M 轴分成两个 lane；每个 lane 复用原近半切分规则，容量由 host 按 N 指定。
// 默认上限 240 保留 N4096/N3584 的既有 tile 数量和顺序。
MAT_FP4_INLINE uint64_t MatFP4WideTileCount(uint64_t local_m, uint64_t tile_m_limit = kMatFP4WideTileM)
{
    if (local_m == 0) return 0;
    if (local_m <= tile_m_limit) return 1;
    if (local_m <= 2 * tile_m_limit) return 2;
    return local_m / tile_m_limit + 1;
}

MAT_FP4_INLINE MatFP4TileDesc MatFP4GetWideTile(uint64_t local_m, uint64_t tile_id,
                                              uint64_t tile_m_limit = kMatFP4WideTileM)
{
    if (local_m <= tile_m_limit) {
        return {0, local_m, MAT_FP4_WIDE};
    }
    if (local_m <= 2 * tile_m_limit) {
        const uint64_t first_m = MatFP4SplitNearHalf16(local_m);
        if (tile_id == 0) {
            return {0, first_m, MAT_FP4_WIDE};
        }
        return {first_m, local_m - first_m, MAT_FP4_WIDE};
    }

    const uint64_t full_tile_count = local_m / tile_m_limit - 1;
    if (tile_id < full_tile_count) {
        return {tile_id * tile_m_limit, tile_m_limit, MAT_FP4_WIDE};
    }
    const uint64_t tail_offset = full_tile_count * tile_m_limit;
    const uint64_t tail_m = local_m - tail_offset;
    const uint64_t first_m = MatFP4SplitNearHalf16(tail_m);
    if (tile_id == full_tile_count) {
        return {tail_offset, first_m, MAT_FP4_WIDE};
    }
    return {tail_offset + first_m, tail_m - first_m, MAT_FP4_WIDE};
}

#undef MAT_FP4_INLINE

#endif
