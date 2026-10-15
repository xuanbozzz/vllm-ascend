# MixedW2W4MoE 算子使用说明

## 文档信息

| 项目 | 说明 |
| --- | --- |
| 算子名称 | `MixedW2W4MoE` |
| 工程 Kernel 名称 | `mixed_w2w4_moe_ffn` |
| 功能简称 | 混合 W2A8/W4A8 MoE FFN 融合算子 |
| 文档版本 | 1.0 |

## 功能说明

### 接口功能

本算子面向 MoE FFN 场景，在一次算子调用中按专家分组完成以下计算：

1. Gate/Up Grouped Matmul；
2. SwiGLU 激活；
3. 中间结果 MXFP8 动态量化；
4. Down Grouped Matmul；
5. 输出 BF16 结果。

不同专家可以分别选择 W2A8 或 W4A8 权重格式。当前支持以下三种专家路径：

| `is_expert_w4[e]` | `is_down_w4[e]` | 专家路径 |
| :---: | :---: | --- |
| 0 | 0 | W2A8 Gate/Up + W2A8 Down（2+2） |
| 0 | 1 | W2A8 Gate/Up + W4A8 Down（2+4） |
| 1 | 1 | W4A8 Gate/Up + W4A8 Down（4+4） |
| 1 | 0 | 不支持（4+2） |

### 符号定义

| 符号 | 含义 |
| --- | --- |
| `G` | 专家数量 |
| `M_e` | 第 `e` 个专家对应的 token 数 |
| `M` | 总 token 数，`M = Σ M_e` |
| `K` | Gate/Up Matmul 的归约维度 |
| `N` | Gate 与 Up 的总输出宽度 |
| `H` | SwiGLU 输出宽度及 Down Matmul 的归约维度，`H = N/2` |
| `D` | Down Matmul 的输出宽度 |
| `A_e` | 第 `e` 个专家的输入激活，逻辑形状为 `[M_e, K]` |
| `W_e^GU` | 第 `e` 个专家的 Gate/Up 权重，逻辑形状为 `[N, K]` |
| `W_e^D` | 第 `e` 个专家的 Down 权重，逻辑形状为 `[D, H]` |
| `C_e` | 第 `e` 个专家的输出，形状为 `[M_e, D]` |

### 计算过程

对专家 `e`，首先根据 `groupList[e]` 取得对应的连续 token 范围：

```text
row_begin(e) = sum(groupList[0:e])
A_e = A[row_begin(e) : row_begin(e) + groupList[e], :]
```

随后执行：

```text
Z_e = DequantMXFP8(A_e, A_scale_e) · DequantWeight(W_e^GU, gateup_table_e)^T
若 Gate/Up 为 W2：Z_e = Z_e ⊙ scale_upgate[e, :]
[Gate_e, Up_e] = Split(Z_e, axis=-1)
S_e = Swish(Gate_e) ⊙ Up_e
(H_e, H_scale_e) = QuantMXFP8(S_e)
C_e = BF16(
    DequantMXFP8(H_e, H_scale_e)
    · DequantWeight(W_e^D, down_table_e)^T
)
若 Down 为 W2：在转换为 BF16 前按列乘以 scale_down[e, :]
```

其中：

```text
Swish(x) = x / (1 + exp(-x))
```

`H_e` 和 `H_scale_e` 是算子内部中间结果，不属于用户输出。所有专家的 `C_e` 按输入 token 顺序拼接为最终输出 `C`。

## 输入输出说明

### 逻辑接口

```text
C = MixedW2W4MoE(
    groupList,
    K,
    N,
    D,
    A,
    A_scale,
    scale_upgate,
    scale_down,
    gateup_w,
    gateup_table,
    down_w,
    down_table,
    is_expert_w4,
    is_down_w4
)
```

### 标量参数

| 参数 | 输入/输出 | 数据类型 | 说明 |
| --- | :---: | --- | --- |
| `G` | 输入 | `uint64_t` | 专家数量。当前 Host 调用时显式传入；逻辑接口中也可由 `groupList` 长度得到。 |
| `K` | 输入 | `uint64_t` | Gate/Up Matmul 的归约维度。 |
| `N` | 输入 | `uint64_t` | Gate 与 Up 的总输出宽度；SwiGLU 输出宽度为 `N/2`。 |
| `D` | 输入 | `uint64_t` | Down Matmul 的输出宽度。 |

### Tensor 参数

| 参数 | 输入/输出 | 逻辑形状或存储长度 | 数据类型/编码 | 数据格式 | 说明 |
| --- | :---: | --- | --- | --- | --- |
| `groupList` | 输入 | `[G]` | `INT64` | ND | 每个专家的非累计 token 数 `M_e`。允许某个专家的值为 0。 |
| `A` | 输入 | `[M, K]` | MXFP8 E4M3 | ND | 输入激活；每个元素占 1 Byte。 |
| `A_scale` | 输入 | `[M, K/32]` | E8M0，按 `UINT8` 存储 | ND | `A` 的量化 scale，每连续 32 个 `A` 元素共享一个 scale。 |
| `scale_upgate` | 输入 | `[G, N]` | `UINT64`，低 32 位为 FP32 系数 | ND | W2 Gate/Up 的每专家、每输出列系数；W4 专家保留对应行但不读取。 |
| `scale_down` | 输入 | `[G, D]` | `UINT64`，低 32 位为 FP32 系数 | ND | W2 Down 的每专家、每输出列系数；W4 专家保留对应行但不读取。 |
| `gateup_w` | 输入 | G 个专家权重段顺序拼接 | W2 2-bit code 或 MXFP4 E2M1 | 预打包 zN | Gate/Up 权重。每个专家的精度由 `is_expert_w4` 指定。 |
| `gateup_table` | 输入 | G 个专家表段顺序拼接 | W2 E4M3 pair LUT 或 W4 E8M0 scale | 预处理格式 | 与 `gateup_w` 一一对应。W2 使用 GS32×32 pair LUT；W4 使用 group size 32 的 E8M0 scale。 |
| `down_w` | 输入 | G 个专家权重段顺序拼接 | W2 2-bit code 或 MXFP4 E2M1 | 预打包 zN | Down 权重。每个专家的精度由 `is_down_w4` 指定。 |
| `down_table` | 输入 | G 个专家表段顺序拼接 | W2 E4M3 pair LUT 或 W4 E8M0 scale | 预处理格式 | 与 `down_w` 一一对应。 |
| `is_expert_w4` | 输入 | 逻辑 `[G]`；设备侧为 `[ceil(G/32)]` | 逻辑 `BOOL`；设备侧 `UINT32` bit mask | ND | 第 `e` 位为 0 表示 Gate/Up 使用 W2A8，为 1 表示使用 W4A8。 |
| `is_down_w4` | 输入 | 逻辑 `[G]`；设备侧为 `[ceil(G/32)]` | 逻辑 `BOOL`；设备侧 `UINT32` bit mask | ND | 第 `e` 位为 0 表示 Down 使用 W2A8，为 1 表示使用 W4A8。 |
| `C` | 输出 | `[M, D]` | `BFLOAT16` | ND | 最终 Down Matmul 输出。 |

其中 `M = sum(groupList)`。

### `groupList` 语义

本算子的 `groupList` 保存每个专家的 token 数，不是前缀和。

例如：

```text
groupList = [3, 1, 0, 2]
```

则：

| 专家 | token 数 | `A`/`C` 对应行范围 |
| :---: | :---: | --- |
| 0 | 3 | `[0, 3)` |
| 1 | 1 | `[3, 4)` |
| 2 | 0 | 空分组 |
| 3 | 2 | `[4, 6)` |

因此总 token 数为 6，`A` 的形状为 `[6, K]`，`C` 的形状为 `[6, D]`。

### 路由 bit mask

逻辑上的两个布尔数组在设备侧均压缩为：

```text
uint32_t route_mask[ceil(G / 32)]
```

专家 `e` 的路由值位于：

```text
word = e >> 5
bit  = e & 31
value = (route_mask[word] >> bit) & 1
```

最后一个 `uint32_t` 中未使用的高位必须为 0。当前 `gen_data.py` 会根据命令行中的 0/1 列表自动生成 bit mask。

## 权重与量化表组织要求

### 专家拼接顺序

`gateup_w`、`gateup_table`、`down_w` 和 `down_table` 均按专家编号 `0...G-1` 顺序连续拼接。即使某个专家的 `groupList[e]` 为 0，也必须保留该专家对应的权重段和表段。

### 每个专家的数据大小

令 `H = N/2`，则第 `e` 个专家的数据大小如下：

| 数据段 | W2 路径 | W4 路径 |
| --- | ---: | ---: |
| Gate/Up 权重 | `K*N/4` Byte | `K*N/2` Byte |
| Gate/Up table | `K*N/32` Byte | `K*N/32` Byte |
| Down 权重 | `H*D/4` Byte | `H*D/2` Byte |
| Down table | `H*D/32` Byte | `H*D/32` Byte |

Gate/Up 权重的逻辑列顺序为 `[Gate, Up]`。当前输入文件使用预处理后的设备格式；不能直接将普通 FP32/BF16 稠密权重写入 `gateup_w` 或 `down_w`。

Gate/Up 设备权重及其 table 按 N32 粒度采用 `Gate32, Up32, Gate32, Up32, ...` 的顺序排列。使用工程自带的 `scripts/gen_data.py` 时会自动完成该格式转换。

## 约束说明

- `1 <= G <= 256`。
- `K` 必须为正数且是 64 的倍数。
- `N` 必须为正数且是 128 的倍数。
- `D` 必须为正数且是 32 的倍数。
- `groupList`、`is_expert_w4` 和 `is_down_w4` 的逻辑长度必须均为 `G`。
- `groupList[e]` 必须为非负整数，且 `sum(groupList) > 0`。
- 不支持 `is_expert_w4[e] = 1` 且 `is_down_w4[e] = 0` 的 4+2 路径。
- 所有权重和量化表必须按专家编号完整拼接，包括零 token 专家。
- `A`、权重及其 scale/table 中不得包含非法量化编码；E8M0 的 `0xFF` 不应作为有效 scale。
- 当前输出仅为最终 BF16 矩阵 `C`，中间 SwiGLU 结果及其中间 MXFP8 scale 不对外输出。

## 调用说明

### 当前交付形态

当前工程提供基于 Runtime 的 Kernel 直调样例和命令行 Host 程序，尚未封装为 ACLNN 接口。Host 程序负责准备算子运行所需的内部 workspace 和辅助数据；调用者只需按本文准备公开输入及输出空间。

### 环境依赖

- CANN 9.2.0；
- `ccec` 或 `bisheng` 编译器；
- `ld.lld`；
- 支持 C++17 的 Host 编译器；
- Python 3、NumPy 和 `ml_dtypes`；
- 运行性能采集时需要 `msprof`。

建议先设置工具链路径：

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0
```

### 方式一：生成数据、编译、直接运行和验证

以下示例包含 3 个专家，分别使用 2+2、2+4 和 4+4 路径：

```bash
# 1. 生成输入和 CPU Golden
python3 scripts/gen_data.py 3 4096 4096 4096 \
  --group-list 1,1,1 \
  --is-expert-w4 0,0,1 \
  --is-down-w4 0,1,1 \
  --seed 1234

# 2. 编译 Kernel 和 Host
bash run_mixed_w2w4_moe_ffn_esl.sh

# 3. 直接运行
# 参数依次为 G K N D device_id warmup_cycles test_cycles
./mixed_w2w4_moe_ffn_host 3 4096 4096 4096 1 5 100

# 4. 验证最终 BF16 输出
python3 cmp.py 3 4096
```

`mixed_w2w4_moe_ffn_host` 的命令行格式为：

```text
./mixed_w2w4_moe_ffn_host G K N D [device_id=0] [warmup_cycles=0] [test_cycles=1]
```

### 方式二：编译并使用 `msprof` 运行单个 case

输入数据生成完成后，可以执行：

```bash
bash run_mixed_w2w4_moe_ffn_esl.sh 3 4096 4096 4096 1 5 100
```

该命令会先编译，再调用 `run_mixed_w2w4_moe_ffn_case.sh` 通过 `msprof` 运行对应 case。

如果已经完成编译，也可以直接执行：

```bash
bash run_mixed_w2w4_moe_ffn_case.sh 3 4096 4096 4096 1 5 100
```

### 方式三：批量运行配置文件中的 case

在 `mixed_w2w4_moe_ffn_cases.ini` 中配置全局参数和一个或多个 case：

```ini
[global]
ASCEND_TOOLKIT_HOME = /usr/local/Ascend/cann-9.2.0
RESULT_ROOT = results
DEVICE_ID = 1
WARMUP_CYCLES = 5
TEST_CYCLES = 100
SEED = 1234

[case:mixed_00_01_11]
NUM_EXPERTS = 3
DIM_K = 4096
DIM_N = 4096
DIM_D = 4096
GROUP_LIST = 1,1,1
IS_EXPERT_W4 = 0,0,1
IS_DOWN_W4 = 0,1,1
```

然后执行：

```bash
bash run_mixed_w2w4_moe_ffn_cases.sh
```

批量脚本会完成一次编译，并依次对每个 case 执行数据生成、运行和正确性验证。结果保存在 `RESULT_ROOT` 指定的目录中。

## 输入输出文件

当前样例 Host 从工程目录下的 `input/` 读取二进制输入，并将结果写入 `output/`。

| 文件 | 内容 | 期望大小 |
| --- | --- | ---: |
| `input/input_group_list.bin` | 非累计 `INT64` group list | `8*G` Byte |
| `input/input_a.bin` | MXFP8 E4M3 输入激活 | `M*K` Byte |
| `input/input_a_scale.bin` | E8M0 输入 scale | `M*K/32` Byte |
| `input/input_scale_upgate.bin` | W2 Gate/Up 随路输出 scale，按真实专家编号索引 | `8*G*N` Byte |
| `input/input_scale_down.bin` | W2 Down 随路输出 scale，按真实专家编号索引 | `8*G*D` Byte |
| `input/input_gateup_w.bin` | 按专家拼接的 Gate/Up 权重 | `Σ gateup_weight_bytes(e)` |
| `input/input_gateup_table.bin` | 按专家拼接的 Gate/Up table | `G*K*N/32` Byte |
| `input/input_down_w.bin` | 按专家拼接的 Down 权重 | `Σ down_weight_bytes(e)` |
| `input/input_down_table.bin` | 按专家拼接的 Down table | `G*H*D/32` Byte |
| `input/input_is_expert_w4.bin` | Gate/Up 路由 bit mask | `4*ceil(G/32)` Byte |
| `input/input_is_down_w4.bin` | Down 路由 bit mask | `4*ceil(G/32)` Byte |
| `output/output_c.bin` | 设备 BF16 输出 `C` | `2*M*D` Byte |
| `output/golden_c_fp32.bin` | CPU FP32 Golden | `4*M*D` Byte |
| `output/golden_c.bin` | CPU BF16 Golden | `2*M*D` Byte |

Host 会严格检查所有输入文件的实际字节数，文件大小不匹配时终止运行并报告错误。

## 正确性验证

执行：

```bash
python3 cmp.py G D
```

验证脚本会：

- 根据 `input/input_group_list.bin` 计算总 token 数；
- 将 `output/output_c.bin` 按 BF16 解析；
- 将 FP32 Golden 按 BF16 RNE 规则舍入后进行比较；
- 输出整体及各专家的匹配率、最大绝对误差和主要不匹配位置；
- 默认要求匹配率不低于 99.9%。

如需调整阈值或输出数量，可以使用：

```bash
python3 cmp.py G D \
  --rtol 1e-1 \
  --atol 1e-1 \
  --required-match-rate 99.9 \
  --show 20
```

## 性能测试说明

当前性能样例在每个计时循环中重新拷贝 `A`、`A_scale`、Gate/Up 权重及 table、Down 权重及 table，使主要输入重新从 HBM 读取，降低 L2 缓存命中对结果的影响。

进行算子间性能对比时，应保证以下条件一致：

- 使用相同的 `groupList`、shape 和专家路由；
- 使用相同的 warmup/test 次数；
- 使用相同的 H2D 和缓存处理方式；
- 区分 Host 端统计时间与 profiler 中的纯 Kernel 时间；
- 同时检查正确性结果，避免仅比较异常执行路径的时间。
