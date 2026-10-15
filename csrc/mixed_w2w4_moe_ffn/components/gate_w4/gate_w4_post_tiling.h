#ifndef MIXED_MOE_GATE_W4_POST_TILING_H
#define MIXED_MOE_GATE_W4_POST_TILING_H

#include "gate_w4_tiling.h"

// 后处理直接消费一个完整的MatMul L0C tile。W4反量化使用2个packed输入
// buffer和2个decoded输出buffer；Host把其最大占用之后的剩余UB全部划给
// SwiGLUQuant。Post区等分为Gate/H与Up两个slot，Up随后复用为FP8输出和scratch。
constexpr uint64_t kFusedPostRows = 256;
constexpr uint64_t kFusedSubRows = 128;
constexpr uint64_t kFusedColumns = 192;
constexpr uint64_t kFusedUbBytes = 256 * 1024;
constexpr uint64_t kFusedGateOffset = 0;
constexpr uint64_t kFusedWorkspaceBytes = 8448;
constexpr uint64_t kFusedPostFree = 4;
constexpr uint64_t kFusedPostReady = 5;

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

FUSED_INLINE uint64_t FusedMatrixBytes(uint64_t post_bytes)
{
    return post_bytes / 2;
}

FUSED_INLINE uint64_t FusedUpOffset(uint64_t post_bytes)
{
    return FusedMatrixBytes(post_bytes);
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

FUSED_INLINE uint64_t FusedMaxSubRows(uint64_t columns, uint64_t post_bytes)
{
    if (columns == 0) return 0;
    const uint64_t matrix_bytes = FusedMatrixBytes(post_bytes);
    const uint64_t bf16_rows = matrix_bytes / (columns * sizeof(uint16_t));
    const uint64_t quant_rows = matrix_bytes > kFusedWorkspaceBytes ?
        (matrix_bytes - kFusedWorkspaceBytes) / columns : 0;
    uint64_t rows = bf16_rows < quant_rows ? bf16_rows : quant_rows;
    return rows < kFusedSubRows ? rows : kFusedSubRows;
}

FUSED_INLINE uint64_t FusedAlignedMaxSubRows(uint64_t columns, uint64_t post_bytes)
{
    return FusedMaxSubRows(columns, post_bytes) / 16 * 16;
}

FUSED_INLINE uint64_t FusedPostMLimit(uint64_t columns, uint64_t post_bytes)
{
    return 2 * FusedAlignedMaxSubRows(columns, post_bytes);
}

// 保持原M16近半切分；Host将wide_m_limit限制为两个AIV的M16容量之和。
FUSED_INLINE uint64_t FusedFirstSubRows(uint64_t rows, uint64_t columns,
                                        uint64_t post_bytes)
{
    const uint64_t max_rows = FusedAlignedMaxSubRows(columns, post_bytes);
    if (rows < 32) return rows;
    uint64_t first = MatFP4SplitNearHalf16(rows);
    if (first > max_rows) first = max_rows;
    if (rows - first > max_rows) first = rows - max_rows;
    return first;
}

FUSED_INLINE uint64_t FusedSubRowBegin(uint64_t rows, uint64_t columns,
                                       uint64_t post_bytes, uint64_t subid)
{
    return subid == 0 ? 0 : FusedFirstSubRows(rows, columns, post_bytes);
}

FUSED_INLINE uint64_t FusedSubRows(uint64_t rows, uint64_t columns,
                                   uint64_t post_bytes, uint64_t subid)
{
    const uint64_t first = FusedFirstSubRows(rows, columns, post_bytes);
    return subid == 0 ? first : rows - first;
}

FUSED_INLINE uint64_t FusedWorkspaceOffset(uint64_t rows, uint64_t columns)
{
    // columns是32的整数倍，因此FP8输出字节数天然32B对齐。
    return rows * columns;
}

FUSED_INLINE bool FusedPostFits(uint64_t rows, uint64_t columns, uint64_t post_bytes)
{
    const uint64_t first = FusedFirstSubRows(rows, columns, post_bytes);
    const uint64_t second = rows - first;
    const uint64_t largest = first > second ? first : second;
    const uint64_t elements = largest * columns;
    const uint64_t matrix_bytes = FusedMatrixBytes(post_bytes);
    return largest <= FusedMaxSubRows(columns, post_bytes) &&
        elements * sizeof(uint16_t) <= matrix_bytes &&
        FusedWorkspaceOffset(largest, columns) + kFusedWorkspaceBytes <= matrix_bytes;
}

#undef FUSED_INLINE
#endif
