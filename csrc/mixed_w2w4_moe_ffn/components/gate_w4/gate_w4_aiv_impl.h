#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#include "kernel_operator.h"
#include "kernel_tensor.h"
#include "gate_w4_tiling.h"
#include "gate_w4_post_tiling.h"
#include "gate_w4_swiglu_quant.h"
#include "../../mixed_route.h"

using namespace AscendC;
using namespace MicroAPI;

static_assert(kFusedWorkspaceBytes == FfnQuant::kWorkspaceBytes,
              "post workspace size mismatch");
static_assert(kMatFP4W4AivInputBufferCount == 2,
              "fused W4 decoder requires a two-slot input ring");
static_assert(kMatFP4MaxTileM <= kFusedPostRows,
              "one-tile delayed Post requires one M post block per MatMul tile");

#ifdef __DAV_C310_VEC__

class MatFP4 {
private:
    static constexpr uint64_t kInputBufferCount = kMatFP4W4AivInputBufferCount;
    static constexpr uint64_t kOutputBufferCount = kMatFP4AivOutputBufferCount;

    struct PendingPost {
        FusedPostTile post;
        uint64_t global_row;
        uint64_t hidden_column;
        bool valid;
    };

    __aicore__ inline void init_spr()
    {
        set_atomic_none();
        set_pcie_wr_ctrl(0);
        set_pcie_rd_ctrl(0);
    }

    // Packed E2M1 is laid out as zN [N1,K1,K0=16,N0/2=16], with the
    // lower nibble holding the earlier N0 element.  Two B8 gathers expand
    // each 256-Byte K1 fractal to 512 bytes of exact E4M3 values.  The
    // interleaved store leaves the following 32-Byte UB padding row untouched.
    __aicore__ inline void run_fp4_to_fp8_lut(__ubuf__ void *packed, __ubuf__ void *decoded,
                                            uint16_t n1_blocks, uint16_t k1_blocks)
    {
        constexpr uint16_t kPackedBytesPerK1 = 16 * 16;
        constexpr uint16_t kDecodedStridePerK1 = 17 * 32;
        __ubuf__ uint8_t *table_addr = lut_ub;

        __VEC_SCOPE__
        {
            RegTensor<uint8_t> table_low;
            RegTensor<uint8_t> table_high;
            RegTensor<uint8_t> packed_index;
            RegTensor<uint8_t> value_low;
            RegTensor<uint8_t> value_high;
            MaskReg mask_all8 = CreateMask<uint8_t, MaskPattern::ALL>();

            // Load each table once per VF and reuse both registers for all K1 fractals.
            DataCopy(table_low, table_addr);
            DataCopy(table_high, table_addr + kMatFP4W4LutHalfBytes);

            const uint16_t tile_count = n1_blocks * k1_blocks;
            for (uint16_t tile = 0; tile < tile_count; ++tile)
            {
                DataCopy(packed_index, (__ubuf__ uint8_t *)packed + tile * kPackedBytesPerK1);
                Gather(value_low, table_low, packed_index);
                Gather(value_high, table_high, packed_index);
                DataCopy<uint8_t, StoreDist::DIST_INTLV_B8>(
                    (__ubuf__ uint8_t *)decoded + tile * kDecodedStridePerK1,
                    value_low, value_high, mask_all8);
            }
        }
    }

    __aicore__ inline void gm_ub(__ubuf__ void *ub, __gm__ void *gm, uint64_t burst_count, uint64_t burst_bytes,
                                uint64_t destination_stride, uint64_t source_stride)
    {
        uint64_t cfg0 = 0;
        cfg0 |= burst_count << 4;
        cfg0 |= burst_bytes << 25;

        uint64_t cfg1 = source_stride;
        cfg1 |= destination_stride << 40;

        uint64_t loop_size = 1;
        loop_size |= static_cast<uint64_t>(1) << 21;
        set_loop_size_outtoub(loop_size);
        set_loop1_stride_outtoub(0);
        set_loop2_stride_outtoub(0);

        copy_gm_to_ubuf_align_v2((__ubuf__ float *)ub, (__gm__ float *)gm, cfg0, cfg1);
    }

    __aicore__ inline void ub_l1(__cbuf__ void *l1, __ubuf__ void *ub, uint64_t burst_count, uint64_t burst_length,
                                uint64_t destination_stride, uint64_t source_stride)
    {
        uint64_t cfg = 0;
        cfg |= burst_count << 4;
        cfg |= burst_length << 16;
        cfg |= source_stride << 32;
        cfg |= destination_stride << 48;
        copy_ubuf_to_cbuf((__cbuf__ uint8_t *)l1, (__ubuf__ uint8_t *)ub, cfg);
    }

    __aicore__ inline void InitBuffer()
    {
        const uint64_t small_elements = narrow_nbs * small_aiv_kbs;
        const uint64_t wide_elements = wide_nbs * middle_aiv_kbs;
        const uint64_t max_elements = small_elements > wide_elements ? small_elements : wide_elements;
        const uint64_t packed_bytes = max_elements / 2;
        const uint64_t decoded_bytes = max_elements * 17 / 16;
        __ubuf__ uint8_t *ub_base = (__ubuf__ uint8_t *)get_imm(0);

        // W4反量化独占低地址区：packed输入和decoded输出均使用双缓冲。
        packed_w[0] = (__ubuf__ void *)ub_base;
        packed_w[1] = (__ubuf__ void *)(ub_base + packed_bytes);
        decoded_w[0] = (__ubuf__ void *)(ub_base + kInputBufferCount * packed_bytes);
        decoded_w[1] = (__ubuf__ void *)(ub_base +
            kInputBufferCount * packed_bytes + decoded_bytes);
        lut_ub = ub_base + kInputBufferCount * packed_bytes +
                 kOutputBufferCount * decoded_bytes;

        // 与 AIC 共用两个 256 KiB L1 槽，B 起点按当前档位的 A 容量计算。
        l1_base = (__cbuf__ void *)get_imm(0);
    }

    __aicore__ inline void InitLut(__gm__ void *tiling)
    {
        // A single immutable table per AIV is shared by every expert/tile.
        // EVENT_ID3与输入ring的MTE2->V事件0/1以及Post事件2分离。
        __gm__ uint8_t *table_source = (__gm__ uint8_t *)tiling + kMatFP4W4LutOffset;
        gm_ub(lut_ub, table_source, 1, kMatFP4W4LutBytes, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    }

    __aicore__ inline void process_tile(__gm__ uint8_t *group_b, MatFP4NTileDesc n_tile,
                                        uint64_t mode, uint64_t &cnt_k)
    {
        const uint64_t k0 = 16;
        const uint64_t n0 = 32;
        const bool small_tile = mode == MAT_FP4_SMALL;
        const bool wide_tile = mode == MAT_FP4_WIDE;
        const uint64_t tile_n = n_tile.extent;
        const uint64_t n_begin = n_tile.offset;
        const uint64_t aiv_kbs = small_tile ? small_aiv_kbs : middle_aiv_kbs;
        const uint64_t aic_kbs = small_tile ? small_aic_kbs : middle_aic_kbs;
        const uint64_t a1_m_capacity =
            small_tile ? small_m_limit : (wide_tile ? wide_m_limit : kMatFP4MaxTileM);
        const uint64_t a1_bytes = a1_m_capacity * aic_kbs;
        const uint64_t n1_blocks = tile_n / n0;
        const uint64_t kb = MatFP4KTileCount(K, aic_kbs);

        for (uint64_t ki = 0; ki < kb; ++ki) {
            const MatFP4KTileDesc k_tile = MatFP4GetKTile(K, aic_kbs, ki);
            const uint64_t actual_aiv_k = k_tile.extent / 2;
            const uint64_t k0_blocks = actual_aiv_k / k0;
            const uint64_t input_id = cnt_k % kInputBufferCount;
            const uint64_t output_id = cnt_k % kOutputBufferCount;
            event_t input_event = (input_id == 0) ? EVENT_ID0 : EVENT_ID1;
            event_t output_event = (output_id == 0) ? EVENT_ID0 : EVENT_ID1;

            const uint64_t aiv_k_begin =
                k_tile.offset + static_cast<uint64_t>(subid) * actual_aiv_k;
            const uint64_t k0_block_begin = aiv_k_begin / k0;

            // GM->UB：两个AIV搬运相邻K半区，各自使用双缓冲输入ring。
            wait_flag(PIPE_V, PIPE_MTE2, input_event);

            const uint64_t n1_begin = n_begin / n0;
            __gm__ uint8_t *packed_source =
                group_b + (n1_begin * (K / k0) + k0_block_begin) * k0 * (n0 / 2);
            gm_ub(packed_w[input_id], packed_source, n1_blocks, actual_aiv_k * (n0 / 2),
                  actual_aiv_k * (n0 / 2), K * (n0 / 2));

            set_flag(PIPE_MTE2, PIPE_V, input_event);

            // VF等待本轮输入就绪，以及两路decoded输出buffer中的当前一路可用。
            wait_flag(PIPE_MTE2, PIPE_V, input_event);
            wait_flag(PIPE_MTE3, PIPE_V, output_event);

            run_fp4_to_fp8_lut(packed_w[input_id], decoded_w[output_id],
                               static_cast<uint16_t>(n1_blocks), static_cast<uint16_t>(k0_blocks));

            // VF has consumed packed W; decoded output is handed to MTE3.
            set_flag(PIPE_V, PIPE_MTE2, input_event);
            set_flag(PIPE_V, PIPE_MTE3, output_event);
            wait_flag(PIPE_V, PIPE_MTE3, output_event);

            // 等待 AIC 释放对应的 L1B ping-pong buffer。
            wait_intra_block(PIPE_MTE3, output_id * 2);

            // 两个 AIV 写入共享 L1B 的相邻 K1 半区，MTE3 同时去除 UB padding。
            for (uint64_t n1 = 0; n1 < n1_blocks; ++n1) {
                __cbuf__ uint8_t *l1_destination = (__cbuf__ uint8_t *)l1_base +
                    output_id * kMatFP4L1SlotBytes + a1_bytes +
                    (n1 * (2 * k0_blocks) + static_cast<uint64_t>(subid) * k0_blocks) * k0 * n0;
                __ubuf__ uint8_t *ub_source =
                    (__ubuf__ uint8_t *)decoded_w[output_id] + n1 * k0_blocks * (k0 + 1) * n0;
                ub_l1(l1_destination, ub_source, k0_blocks, k0, 0, 1);
            }

            set_flag(PIPE_MTE3, PIPE_V, output_event);
            // 通知 AIC：当前 B 半区已就绪。
            set_intra_block(PIPE_MTE3, output_id * 2 + 1);
            ++cnt_k;
        }
    }

    __aicore__ inline void copy_rows_to_gm(__gm__ uint8_t *destination,
                                          __ubuf__ uint8_t *source,
                                          uint64_t rows, uint64_t row_bytes,
                                          uint64_t destination_pitch,
                                          uint64_t source_pitch)
    {
        if (rows == 0) return;
        copy_ubuf_to_gm_align_v2(destination, source, 0, rows, row_bytes, 0,
                                destination_pitch, source_pitch);
    }

    __aicore__ inline void run_post(FusedPostTile post, uint64_t global_row,
                                    uint64_t hidden_column)
    {
        __ubuf__ uint8_t *post_ub =
            (__ubuf__ uint8_t *)get_imm(0) + post_ub_offset;
        __ubuf__ bfloat16_t *gate =
            (__ubuf__ bfloat16_t *)(post_ub + kFusedGateOffset);
        __ubuf__ bfloat16_t *up =
            (__ubuf__ bfloat16_t *)(post_ub + FusedUpOffset(post_ub_bytes));
        const uint64_t rows = FusedSubRows(
            post.rows, post.columns, post_ub_bytes,
            static_cast<uint64_t>(subid));
        const uint64_t row_begin = global_row + post.row + FusedSubRowBegin(
            post.rows, post.columns, post_ub_bytes,
            static_cast<uint64_t>(subid));
        const uint64_t column_begin = hidden_column + post.column;
        const uint64_t hidden = N / 2;
        const uint64_t workspace_offset = FusedWorkspaceOffset(rows, post.columns);
        __ubuf__ uint8_t *workspace =
            (__ubuf__ uint8_t *)up + workspace_offset;

        wait_intra_block(PIPE_V, kFusedPostReady);
        // 使用普通数字event id，保持与现有buffer_id写法一致。
        wait_flag(PIPE_MTE3, PIPE_V, 2);
        FfnQuant::Run(gate, up, workspace,
                      static_cast<uint32_t>(rows),
                      static_cast<uint32_t>(post.columns));

        set_flag(PIPE_V, PIPE_MTE3, 2);
        wait_flag(PIPE_V, PIPE_MTE3, 2);
        __gm__ uint8_t *h = (__gm__ uint8_t *)h_gm;
        __gm__ uint8_t *scale = (__gm__ uint8_t *)h_scale_gm;
        if (rows != 0) {
            copy_rows_to_gm(h + row_begin * hidden + column_begin,
                            (__ubuf__ uint8_t *)up, rows, post.columns,
                            hidden, post.columns);
            copy_rows_to_gm(scale + row_begin * (hidden / 32) + column_begin / 32,
                            workspace + FfnQuant::kScaleOffset, rows,
                            post.columns / 32, hidden / 32,
                            post.columns / 32);
        }
        set_intra_block(PIPE_MTE3, kFusedPostFree);
        set_flag(PIPE_MTE3, PIPE_V, 2);
    }

    __aicore__ inline void process_fused_tile(__gm__ uint8_t *group_b,
                                              MatFP4NTileDesc n_tile,
                                              MatFP4TileDesc m_tile,
                                              uint64_t group_row,
                                              uint64_t &cnt_k,
                                              PendingPost &pending_post)
    {
        // 当前tile的全部UB->L1反量化先进入MTE3队列，之后才提交上一tile的
        // H/scale写回，避免Post MTE3阻塞当前tile的MatMul输入。
        process_tile(group_b, n_tile, m_tile.mode, cnt_k);
        if (pending_post.valid) {
            run_post(pending_post.post, pending_post.global_row,
                     pending_post.hidden_column);
        }

        pending_post.post = FusedGetPost(m_tile.rows, n_tile.extent, 0);
        pending_post.global_row = group_row + m_tile.offset;
        pending_post.hidden_column = n_tile.offset / 2;
        pending_post.valid = true;
    }

public:
    __aicore__ inline void Init(uint64_t g, uint64_t k, uint64_t n, __gm__ void *group_list, __gm__ void *a,
                               __gm__ void *a_scale, __gm__ void *b, __gm__ void *b_scale,
                               __gm__ void *h, __gm__ void *h_scale, __gm__ void *tiling,
                               __gm__ void *route_mask)
    {
        G = g;
        K = k;
        N = n;
        group_list_gm = group_list;
        a_gm = a;
        a_scale_gm = a_scale;
        b_gm = b;
        b_scale_gm = b_scale;
        h_gm = h;
        h_scale_gm = h_scale;
        route_mask_gm = route_mask;
        __gm__ MatFP4TilingData *tiling_data = (__gm__ MatFP4TilingData *)tiling;
        narrow_nbs = tiling_data->narrow_nbs;
        wide_nbs = tiling_data->wide_nbs;
        small_m_limit = tiling_data->small_m_limit;
        n_only_mode = tiling_data->n_only_mode;
        small_aiv_kbs = tiling_data->small_aiv_kbs;
        small_aic_kbs = tiling_data->small_aic_kbs;
        middle_aiv_kbs = tiling_data->middle_aiv_kbs;
        middle_aic_kbs = tiling_data->middle_aic_kbs;
        wide_m_start = tiling_data->wide_m_start;
        wide_m_limit = tiling_data->wide_m_limit;
        post_ub_offset = tiling_data->post_ub_offset;
        post_ub_bytes = tiling_data->post_ub_bytes;
        coreid = get_block_idx();
        subid = get_subblockid() & 1;
        InitBuffer();
        init_spr();
        InitLut(tiling);
    }

    __aicore__ inline void process()
    {
        // AIC/AIV 使用同一组 M/N 映射，subid 选择相邻 K 半区。
        // 标准循环buffer：循环前只初始化free事件，不预取数据。
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_intra_block(PIPE_MTE3, kFusedPostFree);
        set_flag(PIPE_MTE3, PIPE_V, 2);

        __gm__ uint8_t *b_base = (__gm__ uint8_t *)b_gm;
        __gm__ int64_t *group_list = (__gm__ int64_t *)group_list_gm;
        const uint64_t w2_b_group_bytes = N * K / 4;
        const uint64_t w4_b_group_bytes = N * K / 2;
        uint64_t row_begin = 0;
        uint64_t b_offset = 0;
        PendingPost pending_post {};

        // AIC/AIV 同步跳过空组，event 保护跨组复用的共享 L1B。
        for (uint64_t group_id = 0; group_id < G; ++group_id) {
            const bool route_w4 = ::MixedRouteBit((__gm__ uint32_t *)route_mask_gm, group_id);
            __gm__ uint8_t *group_b = b_base + b_offset;
            b_offset += route_w4 ? w4_b_group_bytes : w2_b_group_bytes;
            const int64_t signed_group_m = group_list[group_id];
            if (signed_group_m <= 0) {
                continue;
            }
            const uint64_t group_m = static_cast<uint64_t>(signed_group_m);
            if (!route_w4) {
                row_begin += group_m;
                continue;
            }
            const bool wide_group = MatFP4IsWideGroup(group_m, n_only_mode, wide_m_start);
            uint64_t cnt_k = 0;
            if (n_only_mode != 0) {
                const uint64_t nb = MatFP4NTileCount(N, narrow_nbs);
                const uint64_t m_tile_count = MatFP4NOnlyTileCount(group_m);
                for (uint64_t ni = static_cast<uint64_t>(coreid); ni < nb; ni += kMatFP4NarrowNCoreCount) {
                    const MatFP4NTileDesc n_tile = MatFP4GetNTile(N, narrow_nbs, ni);
                    for (uint64_t mi = 0; mi < m_tile_count; ++mi) {
                        process_fused_tile(
                            group_b, n_tile,
                            MatFP4GetNOnlyTile(group_m, mi, small_m_limit),
                            row_begin, cnt_k, pending_post);
                    }
                }
            } else if (!wide_group) {
                const uint64_t nb = MatFP4NTileCount(N, narrow_nbs);
                const uint64_t m_tile_count = MatFP4NarrowTileCount(group_m);
                for (uint64_t ni = static_cast<uint64_t>(coreid); ni < nb; ni += kMatFP4NarrowNCoreCount) {
                    const MatFP4NTileDesc n_tile = MatFP4GetNTile(N, narrow_nbs, ni);
                    for (uint64_t mi = 0; mi < m_tile_count; ++mi) {
                        process_fused_tile(
                            group_b, n_tile,
                            MatFP4GetNarrowTile(group_m, mi, small_m_limit),
                            row_begin, cnt_k, pending_post);
                    }
                }
            } else {
                const uint64_t m_lane = static_cast<uint64_t>(coreid) / kMatFP4WideNCoreCount;
                const uint64_t n_lane = static_cast<uint64_t>(coreid) % kMatFP4WideNCoreCount;
                const uint64_t first_local_m = MatFP4SplitNearHalf16(group_m);
                const uint64_t local_m = m_lane == 0 ? first_local_m : group_m - first_local_m;
                const uint64_t local_begin = m_lane == 0 ? 0 : first_local_m;
                const uint64_t nb = MatFP4NTileCount(N, wide_nbs);
                const uint64_t m_tile_count = MatFP4WideTileCount(local_m, wide_m_limit);
                for (uint64_t ni = n_lane; ni < nb; ni += kMatFP4WideNCoreCount) {
                    const MatFP4NTileDesc n_tile = MatFP4GetNTile(N, wide_nbs, ni);
                    for (uint64_t mi = 0; mi < m_tile_count; ++mi) {
                        process_fused_tile(
                            group_b, n_tile,
                            MatFP4GetWideTile(local_m, mi, wide_m_limit),
                            row_begin + local_begin, cnt_k, pending_post);
                    }
                }
            }
            row_begin += group_m;
        }

        if (pending_post.valid) {
            run_post(pending_post.post, pending_post.global_row,
                     pending_post.hidden_column);
        }

        // 每个循环buffer最终保留一个free事件，kernel退出前统一drain。
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, 2);
        wait_intra_block(PIPE_MTE3, 0);
        wait_intra_block(PIPE_MTE3, 2);
        pipe_barrier(PIPE_ALL);
    }

public:
    uint64_t G = 0;
    uint64_t K = 0;
    uint64_t N = 0;
    uint64_t narrow_nbs = 0;
    uint64_t wide_nbs = 0;
    uint64_t small_m_limit = 0;
    uint64_t n_only_mode = 0;
    uint64_t small_aiv_kbs = 0;
    uint64_t small_aic_kbs = 0;
    uint64_t middle_aiv_kbs = 0;
    uint64_t middle_aic_kbs = 0;
    uint64_t wide_m_start = 0;
    uint64_t wide_m_limit = 0;
    uint64_t post_ub_offset = 0;
    uint64_t post_ub_bytes = 0;
    int32_t coreid = 0;
    int32_t subid = 0;

    __gm__ void *group_list_gm = nullptr;
    __gm__ void *a_gm = nullptr;
    __gm__ void *a_scale_gm = nullptr;
    __gm__ void *b_gm = nullptr;
    __gm__ void *b_scale_gm = nullptr;
    __gm__ void *h_gm = nullptr;
    __gm__ void *h_scale_gm = nullptr;
    __gm__ void *route_mask_gm = nullptr;
    __ubuf__ void *packed_w[kInputBufferCount] {};
    __ubuf__ void *decoded_w[kOutputBufferCount] {};
    __ubuf__ uint8_t *lut_ub = nullptr;
    __cbuf__ void *l1_base = nullptr;
};

#ifndef MAT_FP4_COMPONENT_ONLY
extern "C" __global__ __aicore__ void mixed_w2w4_moe_ffn_mix_aiv(uint64_t g, uint64_t k, uint64_t n, __gm__ void *group_list,
                                                       __gm__ void *a, __gm__ void *a_scale, __gm__ void *b,
                                                       __gm__ void *b_scale, __gm__ void *h,
                                                       __gm__ void *h_scale, __gm__ void *tiling,
                                                       __gm__ void *route_mask)
{
    MatFP4 mat;
    mat.Init(g, k, n, group_list, a, a_scale, b, b_scale, h, h_scale, tiling, route_mask);
    mat.process();
}
#endif

#endif
