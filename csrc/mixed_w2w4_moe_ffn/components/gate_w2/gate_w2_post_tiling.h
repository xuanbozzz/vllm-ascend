#ifndef MIXED_MOE_GATE_W2_POST_TILING_H
#define MIXED_MOE_GATE_W2_POST_TILING_H

#include "gate_w2_tiling.h"

// 后处理直接消费一个完整的 MatMul L0C tile，不改变原 MatMul 的 M/K tiling。
// UB低196KiB由反量化独占，高60KiB由SwiGLU Quant独占。Post区等分为
// 30KiB Gate/H与30KiB Up；Up在SwiGLU完成后复用为FP8输出和量化scratch。
constexpr uint64_t kFusedPostRows = 256;
constexpr uint64_t kFusedSubRows = 128;
constexpr uint64_t kFusedColumns = 192;
constexpr uint64_t kFusedUbBytes = 256 * 1024;
constexpr uint64_t kFusedPostBytes = 60 * 1024;
constexpr uint64_t kFusedPostOffset = kFusedUbBytes - kFusedPostBytes;
constexpr uint64_t kFusedMatrixBytes = kFusedPostBytes / 2;
constexpr uint64_t kFusedGateOffset = 0;
constexpr uint64_t kFusedUpOffset = kFusedGateOffset + kFusedMatrixBytes;
constexpr uint64_t kFusedWorkspaceBytes = 8448;
constexpr uint64_t kFusedMaxQuantBytes = kFusedMatrixBytes / sizeof(uint16_t);
constexpr uint64_t kFusedPostFree = 4;
constexpr uint64_t kFusedPostReady = 5;
static_assert(kFusedPostOffset == 196 * 1024, "unexpected dequant UB capacity");
static_assert(kFusedUpOffset + kFusedMatrixBytes == kFusedPostBytes,
              "unexpected fused post layout");
static_assert(kFusedMaxQuantBytes + kFusedWorkspaceBytes <= kFusedMatrixBytes,
              "quant output and workspace exceed the Up slot");

struct FusedPostTile {
    uint64_t row;      // 当前 L0C tile 内的 M 起点。
    uint64_t column;   // 当前 tile 内的 hidden 列起点，物理 Gate/Up 列数为其两倍。
    uint64_t rows;
    uint64_t columns;
};

#ifdef __aicore__
#define FUSED_INLINE __aicore__ inline
#else
#define FUSED_INLINE inline
#endif

FUSED_INLINE uint64_t FusedPostCount(uint64_t rows, uint64_t paired_columns)
{
    return ((rows + kFusedPostRows - 1) / kFusedPostRows) *
        ((paired_columns / 2 + kFusedColumns - 1) / kFusedColumns);
}

// 将 M/N 后处理遍历展平，让 AIC 和 AIV 使用完全相同的顺序。
FUSED_INLINE FusedPostTile FusedGetPost(uint64_t rows, uint64_t paired_columns, uint64_t id)
{
    const uint64_t hidden = paired_columns / 2;
    const uint64_t nb = (hidden + kFusedColumns - 1) / kFusedColumns;
    const uint64_t row = id / nb * kFusedPostRows;
    const uint64_t column = id % nb * kFusedColumns;
    const uint64_t mr = rows - row;
    const uint64_t nr = hidden - column;
    return {row, column, mr < kFusedPostRows ? mr : kFusedPostRows,
            nr < kFusedColumns ? nr : kFusedColumns};
}

FUSED_INLINE uint64_t FusedMaxSubRows(uint64_t columns)
{
    const uint64_t by_space = kFusedMatrixBytes / (columns * sizeof(uint16_t));
    return by_space < kFusedSubRows ? by_space : kFusedSubRows;
}

FUSED_INLINE uint64_t FusedAlignedMaxSubRows(uint64_t columns)
{
    return FusedMaxSubRows(columns) / 16 * 16;
}

// 保持原M16近半切分。Host会把wide mad_m限制为两个AIV的M16容量之和：
// hidden N128时最大224，N160时最大192，N192时最大160。
FUSED_INLINE uint64_t FusedFirstSubRows(uint64_t rows, uint64_t columns)
{
    const uint64_t max_rows = FusedAlignedMaxSubRows(columns);
    if (rows < 32) return rows;
    uint64_t first = MatFP4SplitNearHalf16(rows);
    if (first > max_rows) first = max_rows;
    if (rows - first > max_rows) first = rows - max_rows;
    return first;
}

FUSED_INLINE uint64_t FusedSubRowBegin(uint64_t rows, uint64_t columns, uint64_t subid)
{
    return subid == 0 ? 0 : FusedFirstSubRows(rows, columns);
}

FUSED_INLINE uint64_t FusedSubRows(uint64_t rows, uint64_t columns, uint64_t subid)
{
    const uint64_t first = FusedFirstSubRows(rows, columns);
    return subid == 0 ? first : rows - first;
}

FUSED_INLINE uint64_t FusedWorkspaceOffset(uint64_t rows, uint64_t columns)
{
    // columns是32的整数倍，因此FP8输出字节数天然32B对齐。
    return rows * columns;
}

FUSED_INLINE bool FusedPostFits(uint64_t rows, uint64_t columns)
{
    const uint64_t first = FusedFirstSubRows(rows, columns);
    const uint64_t second = rows - first;
    const uint64_t largest = first > second ? first : second;
    const uint64_t elements = largest * columns;
    return largest <= FusedMaxSubRows(columns) &&
        elements * sizeof(uint16_t) <= kFusedMatrixBytes &&
        FusedWorkspaceOffset(largest, columns) + kFusedWorkspaceBytes <= kFusedMatrixBytes;
}

#undef FUSED_INLINE
#endif
