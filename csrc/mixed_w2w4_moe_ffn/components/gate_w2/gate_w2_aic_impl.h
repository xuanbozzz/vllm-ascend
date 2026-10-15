#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#include "gate_w2_tiling.h"
#include "gate_w2_post_tiling.h"
#include "../../mixed_route.h"

#ifdef __DAV_C310_CUBE__

class MatFP4 {
private:
    static constexpr uint64_t kMxScaleGroupSize = 32;
    static constexpr uint32_t kMxE8m0OneU32 = 0x7F7F7F7F;
    // Ascend 950 上 L0B_MX 对当前 AIC 可见的映射窗口为 4 KiB：64 KiB L0B / 16。
    static constexpr uint64_t kL0bMxMappedBytes = 4 * 1024;
    static constexpr uint64_t kL0bMxFillN = 128;
    static constexpr uint64_t kL0bMxFillK = 1024;

    __aicore__ inline void gm_l1_nd2nz(__cbuf__ void *l1, __gm__ void *gm, uint64_t actual_rows,
                                      uint64_t aligned_rows, uint64_t reduction, uint64_t source_row_stride)
    {
        uint64_t cfg0 = 0;
        cfg0 |= source_row_stride << 4;
        cfg0 |= actual_rows << 48;

        const uint64_t cfg1 = reduction;
        uint64_t nz_parameter = 1;
        nz_parameter |= static_cast<uint64_t>(1) << 16;
        nz_parameter |= aligned_rows << 32;
        nz_parameter |= aligned_rows << 48;
        set_mte2_nz_para(nz_parameter);

        copy_gm_to_cbuf_multi_nd2nz((__cbuf__ float8_e4m3_t *)l1, (__gm__ float8_e4m3_t *)gm, cfg0, cfg1);
    }

    // A-scale 在 GM 中为 [M,K/64,2] E8M0；相邻两个 E8M0 reinterpret 为一个 BF16 后执行 DN2NZ。
    // source_row_stride_bytes 是相邻两行首地址之间的 Byte stride，不是行尾 gap。
    __aicore__ inline void gm_l1_scale_dn2nz(__cbuf__ void *l1, __gm__ void *gm, uint64_t actual_rows,
                                             uint64_t aligned_rows, uint64_t tile_k,
                                             uint64_t source_row_stride_bytes)
    {
        // GM scale按E8M0视角为[M, K/32]，reinterpret成BF16后为[D=M, N=K/64]。
        const uint64_t scale_n_b16 = tile_k >> 6;

        uint64_t cfg0 = 0;
        // [43:4] loop1_src_stride：相邻D行首地址的Byte stride。
        cfg0 |= source_row_stride_bytes << 4;
        // [63:48] n_value：BF16视角下每行包含的元素数。
        cfg0 |= scale_n_b16 << 48;

        uint64_t cfg1 = 0;
        // [20:0] d_value：DN矩阵的行数，使用实际M，硬件负责补齐到16行。
        cfg1 |= actual_rows;
        // [60:21] loop4_src_stride = 0；[61] smallc0_en = 0。

        uint64_t nz_parameter = 1;
        // [31:16] dstNzNStride：相邻D行在目的NZ矩阵中的间隔，单位C0。
        nz_parameter |= static_cast<uint64_t>(1) << 16;
        // [47:32] dstNzC0Stride：同一D行的相邻K64分形间隔，单位32B。
        nz_parameter |= scale_n_b16 << 32;
        // [63:48] dstNzMatrixStride = 0，本次只搬一个DN矩阵。
        set_mte2_nz_para(nz_parameter);

        copy_gm_to_cbuf_multi_dn2nz((__cbuf__ bfloat16_t *)l1, (__gm__ bfloat16_t *)gm, cfg0, cfg1);

        // aligned_rows仅用于目标空间大小和后续L1->L0A_MX的xStep计算。
        (void)aligned_rows;
    }

    __aicore__ inline void l1_l0a(__ca__ void *l0a, __cbuf__ void *l1, uint64_t row_offset, uint64_t reduction_offset,
                                 uint64_t rows, uint64_t reduction, uint64_t source_row_stride)
    {
        uint64_t cfg0 = 0;
        cfg0 |= row_offset >> 4;
        cfg0 |= (reduction_offset >> 5) << 16;
        cfg0 |= (rows >> 4) << 32;
        cfg0 |= (reduction >> 5) << 40;

        uint64_t cfg1 = source_row_stride >> 4;
        cfg1 |= (rows >> 4) << 16;
        load_cbuf_to_ca((__ca__ float8_e4m3_t *)l0a, (__cbuf__ float8_e4m3_t *)l1, cfg0, cfg1, false);
    }

    __aicore__ inline void l1_l0b(__cb__ void *l0b, __cbuf__ void *l1, uint64_t k_offset, uint64_t n_offset,
                                 uint64_t k_extent, uint64_t n_extent, uint64_t source_k_stride)
    {
        uint64_t cfg0 = 0;
        cfg0 |= k_offset >> 4;
        cfg0 |= (n_offset >> 5) << 16;
        cfg0 |= (k_extent >> 4) << 32;
        cfg0 |= (n_extent >> 5) << 40;

        uint64_t cfg1 = source_k_stride >> 4;
        cfg1 |= (n_extent >> 4) << 16;

        // MTE1 将 L1B zN [K0=16,N0=32] 转为 L0B nZ [K0=32,N0=16]。
        load_cbuf_to_cb((__cb__ float8_e4m3_t *)l0b, (__cbuf__ float8_e4m3_t *)l1, cfg0, cfg1, true);
    }

    __aicore__ inline void l1_l0a_mx(__cbuf__ void *l1mx, __ca__ void *l0a, uint64_t m, uint64_t k,
                                     uint64_t source_k, uint64_t k_offset)
    {
#ifdef test_sync
        return;
#endif
        uint64_t cfg0 = 0;
        // [31:16] Y start：在完整L1 scale tile中的K方向起点，单位32B/K64。
        cfg0 |= (k_offset >> 6) << 16;
        // [39:32] X step：M方向搬运的16行分形数。
        cfg0 |= (m >> 4) << 32;
        // [47:40] Y step：当前MAD在K方向搬运的32B/K64分形数。
        cfg0 |= (k >> 6) << 40;

        // srcStride必须保持完整L1源矩阵的行跨度；dst按当前mad_k紧密排布。
        uint64_t cfg1 = source_k >> 6;
        cfg1 |= (k >> 6) << 16;
        load_cbuf_to_ca_mx(((uint64_t)l0a) >> 4, (__cbuf__ void *)l1mx, cfg0, cfg1);
    }

    __aicore__ inline void l1_l0b_mx(__cbuf__ void *l1mx, __cb__ void *l0b, uint64_t n, uint64_t k)
    {
#ifdef test_sync
        return;
#endif
        // ScaleB 的一个 32B 分形包含 [16, 2] 个 E8M0：
        // X 方向对应 N，每个分形覆盖 16 行；Y 方向对应 K，每个分形覆盖 K64。
        const uint64_t x_step = n >> 4;
        const uint64_t y_step = k >> 6;

        uint64_t cfg0 = 0;
        cfg0 |= x_step << 32;
        cfg0 |= y_step << 40;

        // stride 是相邻两个 X 分形首地址的间距，而不是每行搬运后的额外 gap。
        // 当前 L1 源和 L0B_MX 目的均按紧密矩形排布，因此 row pitch 都等于 y_step。
        const uint64_t src_stride = y_step;
        const uint64_t dst_stride = y_step;
        uint64_t cfg1 = src_stride;
        cfg1 |= dst_stride << 16;
        load_cbuf_to_cb_mx(((uint64_t)l0b) >> 4, (__cbuf__ int8_t *)l1mx, cfg0, cfg1);
    }

    __aicore__ inline void fill_cbuf(__cbuf__ uint32_t *dst, uint64_t block_count, uint32_t value)
    {
#ifdef test_sync
        return;
#endif
        uint64_t cfg0 = 0;
        cfg0 |= 1ULL;  // [14:0] repeat times
        cfg0 |= (block_count & 0x7FFFULL) << 16;  // [30:16] blocks/repeat，单位 32B
        create_cbuf_matrix(dst, static_cast<int64_t>(cfg0), value);
    }

    __aicore__ inline void init_l0b_mx_scale_one()
    {
        // 一次填满完整的 4 KiB L0B_MX 映射窗口。之后无论 L0B 数据使用 b0[0]
        // 还是 b0[1]，其通过 l0b_addr / 16 推导出的 scale 地址都落在这份常驻全 1 区域内。
        // 128 * 1024 / 32 = 4096 Byte；对应 x_step=8、y_step=16。
        fill_cbuf((__cbuf__ uint32_t *)l1_base, kL0bMxMappedBytes / 32, kMxE8m0OneU32);
        set_flag(PIPE_MTE2, PIPE_MTE1, 2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, 2);

        l1_l0b_mx(l1_base, b0[0], kL0bMxFillN, kL0bMxFillK);
        set_flag(PIPE_MTE1, PIPE_M, 2);
        wait_flag(PIPE_MTE1, PIPE_M, 2);
    }

    __aicore__ inline void gm_l1_quant_scale(__cbuf__ uint64_t *l1, __gm__ uint64_t *gm,
                                             uint64_t elements)
    {
        AscendC::GlobalTensor<uint64_t> source;
        source.SetGlobalBuffer(gm);
        AscendC::LocalTensor<uint64_t> destination(
            AscendC::TPosition::A1, static_cast<uint64_t>((uint64_t)l1), elements);
        AscendC::DataCopy(destination, source, elements);
    }

    __aicore__ inline void l1_fb_quant_scale(__cbuf__ uint64_t *l1, uint64_t elements)
    {
        // 覆盖FB前，先等上一个N段/expert的L0C->UB输出消费完旧系数。
        pipe_barrier(PIPE_FIX);
        // L1->FB的burstLen单位为64 Byte；FB地址对齐单位128 Byte不能用于计算长度。
        copy_cbuf_to_fbuf((__fbuf__ uint64_t *)get_imm(0), l1, 1,
                          elements * sizeof(uint64_t) / 64, 0, 0);
        pipe_barrier(PIPE_FIX);
    }

    __aicore__ inline void set_fixpipe_quant_scale(uint64_t element_offset)
    {
        // FB按[当前tile的Gate系数,Up系数]存放；每次N32搬运都要选择对应子段。
        // FPC[15:8]为quant-pre地址，单位128 Byte，其余字段保持原先的0。
        const uint64_t address = element_offset * sizeof(uint64_t);
        set_fpc((address >> 7) << 8);
        pipe_barrier(PIPE_FIX);
    }

    __aicore__ inline void l0c_ub(__ubuf__ bfloat16_t *destination, __cc__ float *source,
                                 uint64_t rows, uint64_t aligned_rows, uint64_t subid,
                                 uint64_t destination_stride)
    {
        // 单目的模式：每次搬一个 N32，显式指定接收 AIV。目的行跨度使用
        // 当前完整post tile的hidden列数，UB中保持紧密ND。
        // F322BF16 沿用原基线的模式16，不使用标量 scale。
        uint64_t cfg0 = uint64_t {32} << 4;
        cfg0 |= rows << 16;
        cfg0 |= destination_stride << 32;
        uint64_t cfg1 = aligned_rows | (subid << 18);
        const uint64_t quant_mode = static_cast<uint64_t>(QuantMode_t::VQF322BF16_PRE);
        cfg1 |= ((quant_mode >> 5) & 1) << 29;
        cfg1 |= (quant_mode & 31) << 34;
        cfg1 |= uint64_t {1} << 43;
        set_loop3_para(1);
        copy_matrix_cc_to_ub(destination, source, cfg0, cfg1);
    }

    __aicore__ inline void copy_post_sub(__cc__ float *source, uint64_t aligned_rows,
                                        FusedPostTile post, uint64_t subid, uint64_t tile_hidden)
    {
        const uint64_t rows = FusedSubRows(post.rows, post.columns, subid);
        if (rows == 0) return;
        const uint64_t row_begin = post.row +
            FusedSubRowBegin(post.rows, post.columns, subid);
        __ubuf__ uint8_t *post_ub =
            (__ubuf__ uint8_t *)get_imm(0) + kFusedPostOffset;
        __ubuf__ bfloat16_t *gate =
            (__ubuf__ bfloat16_t *)(post_ub + kFusedGateOffset);
        __ubuf__ bfloat16_t *up =
            (__ubuf__ bfloat16_t *)(post_ub + kFusedUpOffset);
        // 物理 N 排列为 Gate32,Up32,...；一个配对仍由两个独立 GS32x32 码本解码。
        for (uint64_t column = 0; column < post.columns; column += 32) {
            const uint64_t gate_column = (post.column + column) * 2;
            // L0C 为 NZ：跨一个 N16 分形前进 aligned_rows*16，跨一行前进16。
            __cc__ float *gate_source = source + gate_column * aligned_rows + row_begin * 16;
            __cc__ float *up_source = gate_source + 32 * aligned_rows;
            set_fixpipe_quant_scale(post.column + column);
            l0c_ub(gate + column, gate_source, rows, aligned_rows, subid, post.columns);
            set_fixpipe_quant_scale(tile_hidden + post.column + column);
            l0c_ub(up + column, up_source, rows, aligned_rows, subid, post.columns);
        }
    }

    __aicore__ inline void post_tile(__cc__ void *source, uint64_t rows,
                                    uint64_t aligned_rows, uint64_t paired_columns)
    {
        const uint64_t count = FusedPostCount(rows, paired_columns);
        const uint64_t tile_hidden = paired_columns / 2;
        for (uint64_t id = 0; id < count; ++id) {
            // AIV 已完成本 tile 的全部权重解码和上一tile写回，UB可以覆盖。
            wait_intra_block(PIPE_FIX, kFusedPostFree);
            wait_intra_block(PIPE_FIX, kFusedPostFree + 16);
            const FusedPostTile post = FusedGetPost(rows, paired_columns, id);
            copy_post_sub((__cc__ float *)source, aligned_rows, post, 0, tile_hidden);
            copy_post_sub((__cc__ float *)source, aligned_rows, post, 1, tile_hidden);
            // 零行 AIV 也要收到 ready，保证两个子核按同一顺序归还缓冲。
            set_intra_block(PIPE_FIX, kFusedPostReady);
            set_intra_block(PIPE_FIX, kFusedPostReady + 16);
        }
    }

    __aicore__ inline void mmad(__cc__ void *l0c, __ca__ void *l0a, __cb__ void *l0b, uint64_t rows,
                               uint64_t reduction, uint64_t columns, bool clear_accumulator)
    {
        uint64_t cfg = rows;
        cfg |= reduction << 12;
        cfg |= columns << 24;

        // L1/L0A 将 M 对齐到 16，最终仅写回有效行。
        cfg |= static_cast<uint64_t>(1) << 61;
        cfg |= static_cast<uint64_t>(clear_accumulator) << 63;
        // 当前实现沿用显式 PIPE_M/PIPE_FIX 同步，unit-flag 保持关闭。
        mad_mx((__cc__ float *)l0c, (__ca__ float8_e4m3_t *)l0a, (__cb__ float8_e4m3_t *)l0b, cfg);
    }

    __aicore__ inline void InitBuffer()
    {
        // 两个 256 KiB 槽分别作为 ping/pong，槽内 A/B 分界由当前 tiling 档位决定。
        l1_base = (__cbuf__ void *)get_imm(0);

        const uint64_t small_a0_bytes = small_m_limit * small_mad_k;
        const uint64_t middle_a0_bytes = kMatFP4MaxTileM * middle_mad_k;
        const uint64_t a0_bytes = small_a0_bytes > middle_a0_bytes ? small_a0_bytes : middle_a0_bytes;
        __ca__ uint8_t *l0a_base = (__ca__ uint8_t *)get_imm(0);
        a0[0] = (__ca__ void *)l0a_base;
        a0[1] = (__ca__ void *)(l0a_base + a0_bytes);

        const uint64_t small_b0_bytes = narrow_nbs * small_mad_k;
        const uint64_t middle_b0_bytes = narrow_nbs * middle_mad_k;
        const uint64_t wide_b0_bytes = wide_nbs * middle_mad_k;
        const uint64_t narrow_b0_bytes = small_b0_bytes > middle_b0_bytes ? small_b0_bytes : middle_b0_bytes;
        const uint64_t b0_bytes = narrow_b0_bytes > wide_b0_bytes ? narrow_b0_bytes : wide_b0_bytes;
        __cb__ uint8_t *l0b_base = (__cb__ uint8_t *)get_imm(0);
        b0[0] = (__cb__ void *)l0b_base;
        b0[1] = (__cb__ void *)(l0b_base + b0_bytes);

        // 小档双缓冲、宽档单缓冲；N160/N192 的双缓冲按小档容量预留。
        const uint64_t c0_bytes = double_l0c_m * narrow_nbs * sizeof(float);
        __cc__ uint8_t *l0c_base = (__cc__ uint8_t *)get_imm(0);
        c0[0] = (__cc__ void *)l0c_base;
        c0[1] = (__cc__ void *)(l0c_base + c0_bytes);
    }

    __aicore__ inline void sync_l0c_mode()
    {
        wait_flag(PIPE_FIX, PIPE_M, 0);
        wait_flag(PIPE_FIX, PIPE_M, 1);
        set_flag(PIPE_FIX, PIPE_M, 0);
        set_flag(PIPE_FIX, PIPE_M, 1);
    }

    __aicore__ inline void process_tile(__gm__ uint8_t *group_a, __gm__ uint8_t *group_a_scale,
                                        __gm__ uint64_t *group_quant_scale, MatFP4NTileDesc n_tile,
                                        MatFP4TileDesc tile, bool load_quant_scale,
                                        uint64_t &cnt_c, uint64_t &cnt_k)
    {
        const bool small_tile = tile.mode == MAT_FP4_SMALL;
        const bool wide_tile = tile.mode == MAT_FP4_WIDE;
        const uint64_t n_capacity = wide_tile ? wide_nbs : narrow_nbs;
        const uint64_t tile_n = n_tile.extent;
        const uint64_t aic_kbs = small_tile ? small_aic_kbs : middle_aic_kbs;
        const uint64_t mad_k = small_tile ? small_mad_k : middle_mad_k;
        const uint64_t a1_m_capacity =
            small_tile ? small_m_limit : (wide_tile ? wide_m_limit : kMatFP4MaxTileM);
        const uint64_t a1_bytes = a1_m_capacity * aic_kbs;
        const uint64_t a1_scale_bytes = a1_m_capacity * aic_kbs / kMxScaleGroupSize;
        // L1 scale 起点按容量定位，N128 尾块仅缩小搬运和计算范围。
        const uint64_t b1_bytes = n_capacity * aic_kbs;
        const uint64_t act_m = (tile.rows + 15) / 16 * 16;
        const uint64_t kb = MatFP4KTileCount(K, aic_kbs);
        const bool single_l0c = wide_tile || (n_only_mode != 0 && !small_tile);
        const uint64_t id_c = single_l0c ? 0 : (cnt_c & 1);

        for (uint64_t ki = 0; ki < kb; ++ki) {
            const MatFP4KTileDesc k_tile = MatFP4GetKTile(K, aic_kbs, ki);
            const uint64_t actual_aic_k = k_tile.extent;
            const uint64_t mad_kb = actual_aic_k / mad_k;
            const uint64_t id = cnt_k & 1;
            const bool load_scale_this_k = load_quant_scale && ki == 0;
            const uint64_t quant_event = 2 + id;
            __cbuf__ uint8_t *l1_slot = (__cbuf__ uint8_t *)l1_base + id * kMatFP4L1SlotBytes;
            __cbuf__ void *a1_addr = (__cbuf__ void *)l1_slot;
            __cbuf__ void *b1_addr = (__cbuf__ void *)(l1_slot + a1_bytes);
            __cbuf__ void *a1_scale_addr = (__cbuf__ void *)(l1_slot + a1_bytes + b1_bytes);
            __cbuf__ uint64_t *quant_scale_addr = (__cbuf__ uint64_t *)(l1_slot + a1_bytes + b1_bytes + a1_scale_bytes);
            wait_flag(PIPE_MTE1, PIPE_MTE2, id);

            gm_l1_nd2nz(a1_addr, (__gm__ void *)(group_a + tile.offset * K + k_tile.offset), tile.rows,
                        act_m, actual_aic_k, K);
            __gm__ uint8_t *scale_source = group_a_scale + tile.offset * (K / kMxScaleGroupSize) +
                k_tile.offset / kMxScaleGroupSize;
            gm_l1_scale_dn2nz(a1_scale_addr, (__gm__ void *)scale_source, tile.rows, act_m, actual_aic_k,
                              K / kMxScaleGroupSize);
            if (load_scale_this_k) {
                const uint64_t tile_hidden = tile_n / 2;
                const uint64_t logical_hidden_offset = n_tile.offset / 2;
                gm_l1_quant_scale(quant_scale_addr, group_quant_scale + logical_hidden_offset, tile_hidden);
                gm_l1_quant_scale(quant_scale_addr + tile_hidden, group_quant_scale + N / 2 + logical_hidden_offset, tile_hidden);
                set_flag(PIPE_MTE2, PIPE_FIX, quant_event);
            }
            if (load_scale_this_k) {
                wait_flag(PIPE_MTE2, PIPE_FIX, quant_event);
                l1_fb_quant_scale(quant_scale_addr, tile_n);
                set_flag(PIPE_FIX, PIPE_MTE1, quant_event);
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, id);

            // 等待两个 AIV 完成同一 N tile 的相邻 K 半区。
            wait_intra_block(PIPE_MTE1, id * 2 + 1);
            wait_intra_block(PIPE_MTE1, id * 2 + 1 + 16);
            wait_flag(PIPE_MTE2, PIPE_MTE1, id);

            if (ki == 0) {
                wait_flag(PIPE_FIX, PIPE_M, id_c);
            }

            for (uint64_t ki2 = 0; ki2 < mad_kb; ++ki2) {
                const uint64_t cnt = ki2 & 1;
                wait_flag(PIPE_M, PIPE_MTE1, cnt);

                l1_l0a(a0[cnt], a1_addr, 0, ki2 * mad_k, act_m, mad_k, act_m);
                l1_l0b(b0[cnt], b1_addr, ki2 * mad_k, 0, mad_k, tile_n, actual_aic_k);

                // L1中常驻当前 actual_aic_k scale tile；Y start 选择当前 mad_k 窗口。
                l1_l0a_mx(a1_scale_addr, a0[cnt], act_m, mad_k, actual_aic_k, ki2 * mad_k);

                set_flag(PIPE_MTE1, PIPE_M, cnt);
                wait_flag(PIPE_MTE1, PIPE_M, cnt);

                __cc__ void *c0_addr = c0[id_c];
                const bool clear_accumulator = ki == 0 && ki2 == 0;
                // 操作数按 16 对齐存放，MAD 的 mbs 使用当前 tile 的实际 M。
                this->mmad(c0_addr, a0[cnt], b0[cnt], tile.rows, mad_k, tile_n, clear_accumulator);
                set_flag(PIPE_M, PIPE_MTE1, cnt);

                const bool final_reduction = ki + 1 == kb && ki2 + 1 == mad_kb;
                if (final_reduction) {
                    set_flag(PIPE_M, PIPE_FIX, cnt);
                    wait_flag(PIPE_M, PIPE_FIX, cnt);

                    post_tile(c0_addr, tile.rows, act_m, tile_n);
                }
            }

            if (ki + 1 == kb) {
                set_flag(PIPE_FIX, PIPE_M, id_c);
            }

            if (load_scale_this_k) wait_flag(PIPE_FIX, PIPE_MTE1, quant_event);
            // 释放共享 L1B，允许两个 AIV 覆盖写入。
            set_intra_block(PIPE_MTE1, id * 2);
            set_intra_block(PIPE_MTE1, id * 2 + 16);
            set_flag(PIPE_MTE1, PIPE_MTE2, id);
            ++cnt_k;
        }
        ++cnt_c;
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
        quant_scale_gm = quant_scale;
        route_mask_gm = route_mask;
        // H/scale 由配对 AIV 写回；AIC 保持同一 kernel ABI。
        (void)h;
        (void)h_scale;
        __gm__ MatFP4TilingData *tiling_data = (__gm__ MatFP4TilingData *)tiling;
        narrow_nbs = tiling_data->narrow_nbs;
        wide_nbs = tiling_data->wide_nbs;
        small_m_limit = tiling_data->small_m_limit;
        n_only_mode = tiling_data->n_only_mode;
        wide_m_start = tiling_data->wide_m_start;
        wide_m_limit = tiling_data->wide_m_limit;
        double_l0c_m = tiling_data->double_l0c_m;
        small_aic_kbs = tiling_data->small_aic_kbs;
        small_mad_k = tiling_data->small_mad_k;
        middle_aic_kbs = tiling_data->middle_aic_kbs;
        middle_mad_k = tiling_data->middle_mad_k;
        coreid = get_block_idx();
        InitBuffer();
    }

    __aicore__ inline void process()
    {
        // 每个 AIC 在进入 expert/K 循环前一次性填满 4 KiB L0B_MX 映射窗口。
        init_l0b_mx_scale_one();

        // 初始化两个 AIV 对 L1B ping-pong buffer 的写权限；+16 表示第二个 AIV。
        set_intra_block(PIPE_MTE1, 0);
        set_intra_block(PIPE_MTE1, 2);
        set_intra_block(PIPE_MTE1, 16);
        set_intra_block(PIPE_MTE1, 18);

        set_flag(PIPE_MTE1, PIPE_MTE2, 0);
        set_flag(PIPE_MTE1, PIPE_MTE2, 1);
        set_flag(PIPE_M, PIPE_MTE1, 0);
        set_flag(PIPE_M, PIPE_MTE1, 1);
        for (uint64_t event_id = 0; event_id < 8; ++event_id) {
            set_flag(PIPE_FIX, PIPE_M, event_id);
        }

        __gm__ int64_t *group_list = (__gm__ int64_t *)group_list_gm;
        __gm__ uint8_t *a_base = (__gm__ uint8_t *)a_gm;
        __gm__ uint8_t *a_scale_base = (__gm__ uint8_t *)a_scale_gm;
        uint64_t row_begin = 0;
        bool has_previous_mode = false;
        bool previous_single_l0c = false;

        // group_list 保存非累计 M_i；A_i/C_i 起点为之前各组 M_i 之和。
        for (uint64_t group_id = 0; group_id < G; ++group_id) {
            const int64_t signed_group_m = group_list[group_id];
            if (signed_group_m <= 0) {
                continue;
            }
            const uint64_t group_m = static_cast<uint64_t>(signed_group_m);
            if (::MixedRouteBit((__gm__ uint32_t *)route_mask_gm, group_id)) {
                row_begin += group_m;
                continue;
            }
            const bool wide_group = MatFP4IsWideGroup(group_m, n_only_mode, wide_m_start);
            const bool single_l0c_group = wide_group || (n_only_mode != 0 && group_m > small_m_limit);
            if (has_previous_mode && previous_single_l0c != single_l0c_group) {
                // 单缓冲会覆盖双缓冲区域，切档前等待所有 Fixpipe 完成。
                sync_l0c_mode();
            }
            has_previous_mode = true;
            previous_single_l0c = single_l0c_group;
            uint64_t cnt_c = 0;
            uint64_t cnt_k = 0;
            __gm__ uint8_t *group_a = a_base + row_begin * K;
            __gm__ uint8_t *group_a_scale = a_scale_base + row_begin * (K / kMxScaleGroupSize);
            __gm__ uint64_t *group_quant_scale = (__gm__ uint64_t *)quant_scale_gm + group_id * N;

            if (n_only_mode != 0) {
                const uint64_t nb = MatFP4NTileCount(N, narrow_nbs);
                const uint64_t m_tile_count = MatFP4NOnlyTileCount(group_m);
                for (uint64_t ni = static_cast<uint64_t>(coreid); ni < nb; ni += kMatFP4NarrowNCoreCount) {
                    for (uint64_t mi = 0; mi < m_tile_count; ++mi) {
                        process_tile(group_a, group_a_scale, group_quant_scale, MatFP4GetNTile(N, narrow_nbs, ni),
                                     MatFP4GetNOnlyTile(group_m, mi, small_m_limit), mi == 0, cnt_c, cnt_k);
                    }
                }
            } else if (!wide_group) {
                const uint64_t nb = MatFP4NTileCount(N, narrow_nbs);
                const uint64_t m_tile_count = MatFP4NarrowTileCount(group_m);
                for (uint64_t ni = static_cast<uint64_t>(coreid); ni < nb; ni += kMatFP4NarrowNCoreCount) {
                    for (uint64_t mi = 0; mi < m_tile_count; ++mi) {
                        process_tile(group_a, group_a_scale, group_quant_scale, MatFP4GetNTile(N, narrow_nbs, ni),
                                     MatFP4GetNarrowTile(group_m, mi, small_m_limit), mi == 0, cnt_c, cnt_k);
                    }
                }
            } else {
                const uint64_t m_lane = static_cast<uint64_t>(coreid) / kMatFP4WideNCoreCount;
                const uint64_t n_lane = static_cast<uint64_t>(coreid) % kMatFP4WideNCoreCount;
                const uint64_t first_local_m = MatFP4SplitNearHalf16(group_m);
                const uint64_t local_begin = m_lane == 0 ? 0 : first_local_m;
                const uint64_t local_m = m_lane == 0 ? first_local_m : group_m - first_local_m;
                const uint64_t nb = MatFP4NTileCount(N, wide_nbs);
                const uint64_t m_tile_count = MatFP4WideTileCount(local_m, wide_m_limit);
                for (uint64_t ni = n_lane; ni < nb; ni += kMatFP4WideNCoreCount) {
                    for (uint64_t mi = 0; mi < m_tile_count; ++mi) {
                        MatFP4TileDesc tile = MatFP4GetWideTile(local_m, mi, wide_m_limit);
                        tile.offset += local_begin;
                        process_tile(group_a, group_a_scale, group_quant_scale, MatFP4GetNTile(N, wide_nbs, ni),
                                     tile, mi == 0, cnt_c, cnt_k);
                    }
                }
            }
            row_begin += group_m;
        }

        // Leave all event slots empty before the next embedded component
        // reuses the same hardware resources in this kernel invocation.
        wait_flag(PIPE_MTE1, PIPE_MTE2, 0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, 1);
        wait_flag(PIPE_M, PIPE_MTE1, 0);
        wait_flag(PIPE_M, PIPE_MTE1, 1);
        for (uint64_t event_id = 0; event_id < 8; ++event_id) {
            wait_flag(PIPE_FIX, PIPE_M, event_id);
        }
        wait_intra_block(PIPE_FIX, kFusedPostFree);
        wait_intra_block(PIPE_FIX, kFusedPostFree + 16);
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
    uint64_t double_l0c_m = 0;
    uint64_t small_aic_kbs = 0;
    uint64_t small_mad_k = 0;
    uint64_t middle_aic_kbs = 0;
    uint64_t middle_mad_k = 0;
    int32_t coreid = 0;
    __gm__ void *group_list_gm = nullptr;
    __gm__ void *a_gm = nullptr;
    __gm__ void *a_scale_gm = nullptr;
    __gm__ void *b_gm = nullptr;
    __gm__ void *table_gm = nullptr;
    __gm__ void *quant_scale_gm = nullptr;
    __gm__ void *route_mask_gm = nullptr;
    __cbuf__ void *l1_base = nullptr;
    __ca__ void *a0[2] {};
    __cb__ void *b0[2] {};
    __cc__ void *c0[2] {};
};

#ifndef MAT_FP4_COMPONENT_ONLY
extern "C" __global__ __aicore__ void mixed_w2w4_moe_ffn_mix_aic(uint64_t g, uint64_t k, uint64_t n, __gm__ void *group_list,
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
