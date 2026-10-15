#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#include "kernel_operator.h"
#include "kernel_tensor.h"
#include "gate_w2_tiling.h"
#include "gate_w2_post_tiling.h"
#include "gate_w2_swiglu_quant.h"
#include "../../mixed_route.h"

using namespace AscendC;
using namespace MicroAPI;

static_assert(kFusedWorkspaceBytes == FfnQuant::kWorkspaceBytes,
              "post workspace size mismatch");
static_assert(kFusedPostOffset + kFusedPostBytes == kFusedUbBytes,
              "post UB must occupy the top 60 KiB");
static_assert(kFusedMaxQuantBytes + FfnQuant::kWorkspaceBytes <= kFusedMatrixBytes,
              "quant output and workspace exceed the Up slot");
static_assert(kMatFP4MaxTileM <= kFusedPostRows,
              "one-tile delayed Post requires one M post block per MatMul tile");

#ifdef __DAV_C310_VEC__

class MatFP4 {
private:
    static constexpr uint64_t kInputBufferCount = 3;
    static constexpr uint64_t kOutputBufferCount = 2;

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

    // 使用 DIST_UNPACK4_B8 解码动态 KxN tile。
    // 每个 N32xK32 块共享一个 LUT；每个 K1 后保留一行输出 padding，避免 UB bank conflict。
    __aicore__ inline void run_lut(__ubuf__ void *a, __ubuf__ void *table, __ubuf__ void *out,
                                   uint16_t n1_blocks, uint16_t k1_blocks)
    {
        constexpr uint16_t kK0 = 16;
        constexpr uint16_t kPackedN0Bytes = 8;
        constexpr uint16_t kK1BlocksPerLut = 2;
        constexpr uint16_t kPackedBytesPerRepeat = 64;
        constexpr uint16_t kDecodedBytesPerRepeat = 256;
        constexpr uint16_t kRepeatsPerK1 = 2;
        constexpr uint16_t kBankStrideRows = 1;

        __VEC_SCOPE__ {
            MaskReg mask_all32 = CreateMask<uint32_t, MaskPattern::ALL>();
            MaskReg mask_all8 = CreateMask<uint8_t, MaskPattern::ALL>();

            RegTensor<uint16_t> reg_table;
            RegTensor<uint32_t> reg_packed_w32;
            RegTensor<uint32_t> reg_high_nibble;
            RegTensor<uint32_t> reg_high_index;
            RegTensor<uint32_t> reg_index_word;
            RegTensor<uint32_t> reg_index_mask;
            RegTensor<int32_t> reg_shift_right4;
            RegTensor<int32_t> reg_shift_left16;
            RegTensor<uint16_t> reg_value;

            Duplicate(reg_shift_right4, static_cast<int32_t>(4), mask_all32);
            Duplicate(reg_shift_left16, static_cast<int32_t>(16), mask_all32);
            Duplicate(reg_index_mask, static_cast<uint32_t>(0x000F000F), mask_all32);

            for (uint16_t n1 = 0; n1 < n1_blocks; ++n1) {
                for (uint16_t lut_k = 0; lut_k < k1_blocks / kK1BlocksPerLut; ++lut_k) {
                    // 本地 LUT 为 [N/32,K/32,32 Byte]，每个 LUT 对应 2 个 K1 分形。
                    const uint16_t table_index = n1 * (k1_blocks / kK1BlocksPerLut) + lut_k;
                    // Gather 只使用前 16 个 uint16 项，其余读入内容不参与结果，无需 LUT 尾部 padding。
                    DataCopy(reg_table, (__ubuf__ uint16_t *)table + table_index * 16);

                    for (uint16_t local_k1 = 0; local_k1 < kK1BlocksPerLut; ++local_k1) {
                        const uint16_t k1 = lut_k * kK1BlocksPerLut + local_k1;
                        const uint64_t packed_k1_offset =
                            (static_cast<uint64_t>(n1) * k1_blocks + k1) * kK0 * kPackedN0Bytes;
                        const uint64_t output_k1_offset =
                            (static_cast<uint64_t>(n1) * k1_blocks + k1) * (kK0 + kBankStrideRows) * 32;

                        for (uint16_t repeat = 0; repeat < kRepeatsPerK1; ++repeat) {
                            // 每次将 64 个 packed Byte 零扩展为 64 个 uint32 lane。
                            DataCopy<uint32_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                                reg_packed_w32, (__ubuf__ uint32_t *)a +
                                                    (packed_k1_offset + repeat * kPackedBytesPerRepeat) / sizeof(uint32_t));

                            // 在 uint32 的低/高 16 bit 中直接构造两个 pair-LUT 索引。
                            ShiftRight(reg_high_nibble, reg_packed_w32, reg_shift_right4, mask_all32);
                            ShiftLeft(reg_high_index, reg_high_nibble, reg_shift_left16, mask_all32);
                            Or(reg_index_word, reg_packed_w32, reg_high_index, mask_all32);
                            And(reg_index_word, reg_index_word, reg_index_mask, mask_all32);

                            Gather(reg_value, reg_table, (RegTensor<uint16_t> &)reg_index_word);
                            DataCopy((__ubuf__ uint8_t *)out + output_k1_offset + repeat * kDecodedBytesPerRepeat,
                                     (RegTensor<uint8_t> &)reg_value, mask_all8);
                        }
                    }
                }
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
        const uint64_t middle_elements = narrow_nbs * middle_aiv_kbs;
        const uint64_t wide_elements = wide_nbs * middle_aiv_kbs;
        const uint64_t narrow_elements = small_elements > middle_elements ? small_elements : middle_elements;
        const uint64_t max_elements = narrow_elements > wide_elements ? narrow_elements : wide_elements;
        const uint64_t packed_bytes = max_elements / 4;
        const uint64_t decoded_bytes = max_elements * 17 / 16;
        const uint64_t small_lut_count = (small_aiv_kbs / 32) * (narrow_nbs / 32);
        const uint64_t middle_lut_count = (middle_aiv_kbs / 32) * (narrow_nbs / 32);
        const uint64_t wide_lut_count = (middle_aiv_kbs / 32) * (wide_nbs / 32);
        const uint64_t narrow_lut_count = small_lut_count > middle_lut_count ? small_lut_count : middle_lut_count;
        const uint64_t lut_count = narrow_lut_count > wide_lut_count ? narrow_lut_count : wide_lut_count;
        const uint64_t table_bytes = lut_count * 32;
        __ubuf__ uint8_t *ub_base = (__ubuf__ uint8_t *)get_imm(0);

        // packed W 与 LUT 使用3 buffer，decoded输出保持ping-pong双缓冲。
        // 所有反量化buffer都位于低196KiB，不能进入独立Post区。
        a[0] = (__ubuf__ void *)ub_base;
        a[1] = (__ubuf__ void *)(ub_base + packed_bytes);
        a[2] = (__ubuf__ void *)(ub_base + 2 * packed_bytes);
        b[0] = (__ubuf__ void *)(ub_base + kInputBufferCount * packed_bytes);
        b[1] = (__ubuf__ void *)(ub_base + kInputBufferCount * packed_bytes + decoded_bytes);
        table[0] = (__ubuf__ void *)(ub_base + kInputBufferCount * packed_bytes +
                                     kOutputBufferCount * decoded_bytes);
        table[1] = (__ubuf__ void *)((__ubuf__ uint8_t *)table[0] + table_bytes);
        table[2] = (__ubuf__ void *)((__ubuf__ uint8_t *)table[1] + table_bytes);

        // 与 AIC 共用两个 256 KiB L1 槽，B 起点按当前档位的 A 容量计算。
        l1_base = (__cbuf__ void *)get_imm(0);
    }

    __aicore__ inline void process_tile(__gm__ uint8_t *group_b, __gm__ uint8_t *group_table,
                                        MatFP4NTileDesc n_tile, uint64_t mode, uint64_t &cnt_k)
    {
        const uint64_t k0 = 16;
        const uint64_t n0 = 32;
        const uint64_t codebook_k = 32;
        const uint64_t codebook_n = 32;
        const bool small_tile = mode == MAT_FP4_SMALL;
        const bool wide_tile = mode == MAT_FP4_WIDE;
        const uint64_t tile_n = n_tile.extent;
        const uint64_t n_begin = n_tile.offset;
        const uint64_t aic_kbs = small_tile ? small_aic_kbs : middle_aic_kbs;
        const uint64_t a1_m_capacity =
            small_tile ? small_m_limit : (wide_tile ? wide_m_limit : kMatFP4MaxTileM);
        const uint64_t a1_bytes = a1_m_capacity * aic_kbs;
        const uint64_t n1_blocks = tile_n / n0;
        const uint64_t kb = MatFP4KTileCount(K, aic_kbs);
        const uint64_t k_codebook_count = K / codebook_k;

        for (uint64_t ki = 0; ki < kb; ++ki) {
            const MatFP4KTileDesc k_tile = MatFP4GetKTile(K, aic_kbs, ki);
            const uint64_t actual_aiv_k = k_tile.extent / 2;
            const uint64_t k0_blocks = actual_aiv_k / k0;
            const uint64_t codebook_k_blocks = actual_aiv_k / codebook_k;
            const uint64_t input_id = cnt_k % kInputBufferCount;
            const uint64_t output_id = cnt_k & (kOutputBufferCount - 1);
            const uint64_t aiv_k_begin = k_tile.offset + static_cast<uint64_t>(subid) * actual_aiv_k;
            const uint64_t k0_block_begin = aiv_k_begin / k0;
            wait_flag(PIPE_V, PIPE_MTE2, input_id);

            // 两个 AIV 读取同一 N tile 的相邻 K 半区。
            const uint64_t n1_begin = n_begin / n0;
            __gm__ uint8_t *packed_source =
                group_b + (n1_begin * (K / k0) + k0_block_begin) * k0 * (n0 / 4);
            gm_ub(a[input_id], packed_source, n1_blocks, actual_aiv_k * (n0 / 4),
                  actual_aiv_k * (n0 / 4),
                  K * (n0 / 4));

            // LUT 全局排布 [N/32,K/32,32]；每个 N32 内连续搬运当前 K tile 的码表。
            const uint64_t n_group = n_begin / codebook_n;
            const uint64_t codebook_k_block_begin = k0_block_begin / (codebook_k / k0);
            __gm__ uint8_t *table_source =
                group_table + (n_group * k_codebook_count + codebook_k_block_begin) * 32;
            gm_ub(table[input_id], table_source, n1_blocks, codebook_k_blocks * 32, codebook_k_blocks * 32,
                  k_codebook_count * 32);

            set_flag(PIPE_MTE2, PIPE_V, input_id);
            wait_flag(PIPE_MTE2, PIPE_V, input_id);
            wait_flag(PIPE_MTE3, PIPE_V, output_id);

            run_lut(a[input_id], table[input_id], b[output_id], static_cast<uint16_t>(n1_blocks),
                    static_cast<uint16_t>(k0_blocks));

            set_flag(PIPE_V, PIPE_MTE3, output_id);
            // W/table 在 V 完成后即可释放，不需要等待 decoded 输出完成 MTE3。
            set_flag(PIPE_V, PIPE_MTE2, input_id);
            wait_flag(PIPE_V, PIPE_MTE3, output_id);

            // 等待 AIC 释放对应的 L1B ping-pong buffer。
            wait_intra_block(PIPE_MTE3, output_id * 2);

            // 两个 AIV 写入共享 L1B 的相邻 K1 半区，MTE3 同时去除 UB padding。
            for (uint64_t n1 = 0; n1 < n1_blocks; ++n1) {
                __cbuf__ uint8_t *l1_destination = (__cbuf__ uint8_t *)l1_base +
                    output_id * kMatFP4L1SlotBytes + a1_bytes +
                    (n1 * (2 * k0_blocks) + static_cast<uint64_t>(subid) * k0_blocks) * k0 * n0;
                __ubuf__ uint8_t *ub_source =
                    (__ubuf__ uint8_t *)b[output_id] + n1 * k0_blocks * (k0 + 1) * n0;
                ub_l1(l1_destination, ub_source, k0_blocks, k0, 0, 1);
            }

            set_flag(PIPE_MTE3, PIPE_V, output_id);
            // 通知 AIC：当前 B 半区已就绪。
            set_intra_block(PIPE_MTE3, output_id * 2 + 1);
            ++cnt_k;
        }
    }

    __aicore__ inline void copy_rows_to_gm(__gm__ uint8_t *destination, __ubuf__ uint8_t *source,
                                          uint64_t rows, uint64_t row_bytes,
                                          uint64_t destination_pitch, uint64_t source_pitch)
    {
        if (rows == 0) return;
        // 一条指令搬完本后处理块的所有行。两个 pitch 都是相邻行首地址
        // 的字节距离，不是行尾 gap；目的 GM 可以有跨列 tile 的行间空隙。
        // scale 每行仅1~6B有效，UB中按实际groups紧密排列；只写有效字节。
        // CANN9.2 CopyUbufToGmAlignV2 已确认：原生接口先传GM pitch，再传UB pitch。
        copy_ubuf_to_gm_align_v2(destination, source, 0, rows, row_bytes, 0,
                                destination_pitch, source_pitch);
    }

    __aicore__ inline void run_post(FusedPostTile post, uint64_t global_row, uint64_t hidden_column)
    {
        __ubuf__ uint8_t *post_ub =
            (__ubuf__ uint8_t *)get_imm(0) + kFusedPostOffset;
        __ubuf__ bfloat16_t *gate =
            (__ubuf__ bfloat16_t *)(post_ub + kFusedGateOffset);
        __ubuf__ bfloat16_t *up =
            (__ubuf__ bfloat16_t *)(post_ub + kFusedUpOffset);
        const uint64_t rows = FusedSubRows(
            post.rows, post.columns, static_cast<uint64_t>(subid));
        const uint64_t row_begin = global_row + post.row +
            FusedSubRowBegin(post.rows, post.columns, static_cast<uint64_t>(subid));
        const uint64_t column_begin = hidden_column + post.column;
        const uint64_t hidden = N / 2;
        const uint64_t workspace_offset = FusedWorkspaceOffset(rows, post.columns);
        __ubuf__ uint8_t *workspace =
            (__ubuf__ uint8_t *)up + workspace_offset;

        // ready保证延迟一拍的Gate/Up已经写入独立Post区；event 2保证
        // 上一次H/scale写回完成后才由V复用Up/workspace。反量化ring位于
        // 独立的低196KiB，这里不排空它们。
        wait_intra_block(PIPE_V, kFusedPostReady);
        wait_flag(PIPE_MTE3, PIPE_V, 2);
        FfnQuant::Run(gate, up, workspace,
                      static_cast<uint32_t>(rows), static_cast<uint32_t>(post.columns));

        set_flag(PIPE_V, PIPE_MTE3, 2);
        wait_flag(PIPE_V, PIPE_MTE3, 2);
        __gm__ uint8_t *h = (__gm__ uint8_t *)h_gm;
        __gm__ uint8_t *scale = (__gm__ uint8_t *)h_scale_gm;
        if (rows != 0) {
            // FP8输出已复用Up区；H和scale目的地址独立，分别发出一次多行搬运。
            copy_rows_to_gm(h + row_begin * hidden + column_begin,
                            (__ubuf__ uint8_t *)up, rows, post.columns, hidden, post.columns);
            copy_rows_to_gm(scale + row_begin * (hidden / 32) + column_begin / 32,
                            workspace + FfnQuant::kScaleOffset, rows, post.columns / 32,
                            hidden / 32, post.columns / 32);
        }
        // free和完成事件都排在两次GM写回之后。AIV不在这里等待MTE3，
        // 下一tile可以先执行MTE2/V，借其反量化准备阶段隐藏Post写回。
        set_intra_block(PIPE_MTE3, kFusedPostFree);
        set_flag(PIPE_MTE3, PIPE_V, 2);
    }

    __aicore__ inline void process_fused_tile(__gm__ uint8_t *group_b, __gm__ uint8_t *group_table,
                                              MatFP4NTileDesc n_tile, MatFP4TileDesc m_tile,
                                              uint64_t group_row, uint64_t &cnt_k,
                                              PendingPost &pending_post)
    {
        // 先完成当前tile全部K段的反量化。随后才提交上一tile的Post，
        // 因而同一PIPE_MTE3上当前UB->L1一定排在上一tile的H/scale写回之前。
        process_tile(group_b, group_table, n_tile, m_tile.mode, cnt_k);
        if (pending_post.valid) {
            run_post(pending_post.post, pending_post.global_row,
                     pending_post.hidden_column);
        }

        // Host保证M tile<=256且hidden N tile<=192，所以每个MatMul tile
        // 恰好对应一个Post tile；仅保存描述符，不增加UB数据buffer。
        pending_post.post = FusedGetPost(m_tile.rows, n_tile.extent, 0);
        pending_post.global_row = group_row + m_tile.offset;
        pending_post.hidden_column = n_tile.offset / 2;
        pending_post.valid = true;
    }

public:
    __aicore__ inline void Init(uint64_t g, uint64_t k, uint64_t n, __gm__ void *group_list, __gm__ void *a,
                               __gm__ void *a_scale, __gm__ void *b, __gm__ void *table, __gm__ void *quant_scale, __gm__ void *h, __gm__ void *h_scale,
                               __gm__ void *tiling, __gm__ void *route_mask)
    {
        G = g;
        K = k;
        N = n;
        group_list_gm = group_list;
        a_gm = a;
        a_scale_gm = a_scale;
        b_gm = b;
        table_gm = table;
        (void)quant_scale;
        h_gm = h;
        h_scale_gm = h_scale;
        route_mask_gm = route_mask;
        __gm__ MatFP4TilingData *tiling_data = (__gm__ MatFP4TilingData *)tiling;
        narrow_nbs = tiling_data->narrow_nbs;
        wide_nbs = tiling_data->wide_nbs;
        small_m_limit = tiling_data->small_m_limit;
        n_only_mode = tiling_data->n_only_mode;
        wide_m_start = tiling_data->wide_m_start;
        wide_m_limit = tiling_data->wide_m_limit;
        small_aiv_kbs = tiling_data->small_aiv_kbs;
        small_aic_kbs = tiling_data->small_aic_kbs;
        middle_aiv_kbs = tiling_data->middle_aiv_kbs;
        middle_aic_kbs = tiling_data->middle_aic_kbs;
        coreid = get_block_idx();
        subid = get_subblockid() & 1;
        InitBuffer();
        init_spr();
    }

    __aicore__ inline void process()
    {
        // AIC/AIV 使用同一组 M/N 映射，subid 选择相邻 K 半区。
        const uint64_t codebook_k = 32;
        const uint64_t codebook_n = 32;
        const uint64_t n_group_count = N / codebook_n;

        set_flag(PIPE_V, PIPE_MTE2, 0);
        set_flag(PIPE_V, PIPE_MTE2, 1);
        set_flag(PIPE_V, PIPE_MTE2, 2);
        set_flag(PIPE_MTE3, PIPE_V, 0);
        set_flag(PIPE_MTE3, PIPE_V, 1);
        // 独立Post区初始为空。free让AIC可以写入第一个Gate/Up；
        // 完成事件为第一次run_post提供初始token。
        set_intra_block(PIPE_MTE3, kFusedPostFree);
        set_flag(PIPE_MTE3, PIPE_V, 2);

        __gm__ uint8_t *b_base = (__gm__ uint8_t *)b_gm;
        __gm__ uint8_t *table_base = (__gm__ uint8_t *)table_gm;
        __gm__ int64_t *group_list = (__gm__ int64_t *)group_list_gm;
        const uint64_t w2_b_group_bytes = N * K / 4;
        const uint64_t w4_b_group_bytes = N * K / 2;
        const uint64_t w2_table_group_bytes = (K / codebook_k) * n_group_count * 32;
        const uint64_t w4_table_group_bytes = N * K / 32;

        uint64_t row_begin = 0;
        uint64_t b_offset = 0;
        uint64_t table_offset = 0;
        // Pending跨group保存，使单tile expert也能用下一expert的反量化
        // 隐藏上一expert的Post；输出地址完整保存在描述符中。
        PendingPost pending_post {};
        // AIC/AIV 同步跳过空组，event 保护跨组复用的共享 L1B。
        for (uint64_t group_id = 0; group_id < G; ++group_id) {
            const bool route_w4 = ::MixedRouteBit((__gm__ uint32_t *)route_mask_gm, group_id);
            __gm__ uint8_t *group_b = b_base + b_offset;
            __gm__ uint8_t *group_table = table_base + table_offset;
            b_offset += route_w4 ? w4_b_group_bytes : w2_b_group_bytes;
            table_offset += route_w4 ? w4_table_group_bytes : w2_table_group_bytes;
            const int64_t signed_group_m = group_list[group_id];
            if (signed_group_m <= 0) {
                continue;
            }
            const uint64_t group_m = static_cast<uint64_t>(signed_group_m);
            if (route_w4) {
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
                        process_fused_tile(group_b, group_table, n_tile,
                                           MatFP4GetNOnlyTile(group_m, mi, small_m_limit), row_begin,
                                           cnt_k, pending_post);
                    }
                }
            } else if (!wide_group) {
                const uint64_t nb = MatFP4NTileCount(N, narrow_nbs);
                const uint64_t m_tile_count = MatFP4NarrowTileCount(group_m);
                for (uint64_t ni = static_cast<uint64_t>(coreid); ni < nb; ni += kMatFP4NarrowNCoreCount) {
                    const MatFP4NTileDesc n_tile = MatFP4GetNTile(N, narrow_nbs, ni);
                    for (uint64_t mi = 0; mi < m_tile_count; ++mi) {
                        process_fused_tile(group_b, group_table, n_tile,
                                           MatFP4GetNarrowTile(group_m, mi, small_m_limit), row_begin,
                                           cnt_k, pending_post);
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
                        process_fused_tile(group_b, group_table, n_tile,
                                           MatFP4GetWideTile(local_m, mi, wide_m_limit),
                                           row_begin + local_begin, cnt_k, pending_post);
                    }
                }
            }
            row_begin += group_m;
        }

        // 最后一个tile没有后继反量化负责触发，kernel结束前单独完成Post。
        if (pending_post.valid) {
            run_post(pending_post.post, pending_post.global_row,
                     pending_post.hidden_column);
        }

        // 仅在kernel结束时统一排空各自的ring，保证最后一次L1/GM写回完成。
        for (uint64_t input_id = 0; input_id < kInputBufferCount; ++input_id) {
            wait_flag(PIPE_V, PIPE_MTE2, input_id);
        }
        wait_flag(PIPE_MTE3, PIPE_V, 0);
        wait_flag(PIPE_MTE3, PIPE_V, 1);
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
    uint64_t wide_m_start = 0;
    uint64_t wide_m_limit = 0;
    uint64_t small_aiv_kbs = 0;
    uint64_t small_aic_kbs = 0;
    uint64_t middle_aiv_kbs = 0;
    uint64_t middle_aic_kbs = 0;
    int32_t coreid = 0;
    int32_t subid = 0;

    __gm__ void *group_list_gm = nullptr;
    __gm__ void *a_gm = nullptr;
    __gm__ void *a_scale_gm = nullptr;
    __gm__ void *b_gm = nullptr;
    __gm__ void *table_gm = nullptr;
    __gm__ void *h_gm = nullptr;
    __gm__ void *h_scale_gm = nullptr;
    __gm__ void *route_mask_gm = nullptr;
    __ubuf__ void *a[kInputBufferCount] {};
    __ubuf__ void *b[kOutputBufferCount] {};
    __ubuf__ void *table[kInputBufferCount] {};
    __cbuf__ void *l1_base = nullptr;
};

#ifndef MAT_FP4_COMPONENT_ONLY
extern "C" __global__ __aicore__ void mixed_w2w4_moe_ffn_mix_aiv(uint64_t g, uint64_t k, uint64_t n, __gm__ void *group_list,
                                                       __gm__ void *a, __gm__ void *a_scale, __gm__ void *b,
                                                       __gm__ void *table, __gm__ void *quant_scale, __gm__ void *h, __gm__ void *h_scale,
                                                       __gm__ void *tiling, __gm__ void *route_mask)
{
    MatFP4 mat;
    mat.Init(g, k, n, group_list, a, a_scale, b, table, quant_scale, h, h_scale, tiling, route_mask);
    mat.process();
}
#endif

#endif
