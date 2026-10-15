#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -ne 0 && $# -ne 7 ]]; then
    echo "Usage:" >&2
    echo "  $0                                    # 只编译" >&2
    echo "  $0 <experts> <K> <N> <D> <device> <warmup> <test>  # 编译后运行一个 case" >&2
    exit 2
fi

ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

if [[ ! -d "${ASCEND_TOOLKIT_HOME}" ]]; then
    echo "ASCEND_TOOLKIT_HOME does not exist: ${ASCEND_TOOLKIT_HOME}" >&2
    exit 1
fi

if [[ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
elif [[ -f "${ASCEND_TOOLKIT_HOME}/bin/setenv.bash" ]]; then
    source "${ASCEND_TOOLKIT_HOME}/bin/setenv.bash"
fi
set -u

PLATFORM_DIR="${ASCEND_TOOLKIT_HOME}/$(uname -m)-linux"
export PATH="${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/runtime/lib64:${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH:-}"

CCEC="${CCEC:-$(command -v ccec 2>/dev/null || command -v bisheng 2>/dev/null || true)}"
CCE_LINKER="${LD_LLD:-${CCE_LINKER:-$(command -v ld.lld 2>/dev/null || command -v lld 2>/dev/null || true)}}"
CXX="${CXX:-g++}"

if [[ -z "${CCEC}" || ! -x "${CCEC}" ]]; then
    echo "Cannot find ccec or bisheng after loading ${ASCEND_TOOLKIT_HOME}." >&2
    echo "Check ASCEND_TOOLKIT_HOME in mixed_w2w4_moe_ffn_cases.ini." >&2
    exit 1
fi
if [[ -z "${CCE_LINKER}" || ! -x "${CCE_LINKER}" ]]; then
    echo "Cannot find ld.lld after loading ${ASCEND_TOOLKIT_HOME}." >&2
    echo "Check ASCEND_TOOLKIT_HOME in mixed_w2w4_moe_ffn_cases.ini." >&2
    exit 1
fi

if [[ ! -d "${PLATFORM_DIR}/ascendc/include/basic_api" ]]; then
    PLATFORM_DIR="${ASCEND_TOOLKIT_HOME}"
fi
if [[ ! -d "${PLATFORM_DIR}/ascendc/include/basic_api" ]]; then
    echo "Cannot find AscendC headers under ${ASCEND_TOOLKIT_HOME}." >&2
    exit 1
fi

COMMON_INCLUDES=(
    "-I${PLATFORM_DIR}/ascendc/include/basic_api"
    "-I${PLATFORM_DIR}/ascendc/include/basic_api/impl"
    "-I${PLATFORM_DIR}/ascendc/include/basic_api/interface"
    "-I${PLATFORM_DIR}/ascendc/include/highlevel_api"
    "-I${PLATFORM_DIR}/ascendc/include/aicore/simt_api"
    "-I${PLATFORM_DIR}/include"
    "-I${PLATFORM_DIR}/asc"
)

# Keep compiler-default scalar and terminal DCache maintenance for chained kernels.
COMMON_CCEC_FLAGS=(
    -c
    -x cce
    -g
    -O3
    -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-function-stack-size=0x4000
    -mllvm -cce-aicore-addr-transform=true
    -mllvm -cce-aicore-or-combine=false
    -mllvm -instcombine-code-sinking=false
    -Xclang -fcce-vf-vl=256
    --cce-auto-sync=off
    -mllvm -cce-aicore-jump-expand=true
    -mllvm -cce-aicore-mask-opt=false
    -ferror-limit=0
    -w
    --cce-simd-vf-fusion=true
    -DENABLE_MOVVP
    -DPERF_API_NORM
    -mllvm -api-deps-filter
)

compile_kernel()
{
    local arch="$1"
    local macro="$2"
    local source_file="$3"
    local object_file="$4"

    "${CCEC}" \
        "${COMMON_CCEC_FLAGS[@]}" \
        "${COMMON_INCLUDES[@]}" \
        "--cce-aicore-arch=${arch}" \
        "-D${macro}" \
        "${source_file}" \
        -o "${object_file}"
}

echo ">>> [1/3] 编译单算子 AIV"
rm -f mixed_w2w4_moe_ffn_mix_aiv.o mixed_w2w4_moe_ffn_mix_aic.o mixed_w2w4_moe_ffn.o mixed_w2w4_moe_ffn_host tiling_*.o
compile_kernel dav-c310-vec __DAV_C310_VEC__ mixed_w2w4_moe_ffn_aiv.cce mixed_w2w4_moe_ffn_mix_aiv.o

echo ">>> [2/3] 编译单算子 AIC 并链接 Kernel"
compile_kernel dav-c310-cube __DAV_C310_CUBE__ mixed_w2w4_moe_ffn_aic.cce mixed_w2w4_moe_ffn_mix_aic.o

LINKER_DRIVER_FLAGS=()
if [[ "$(basename "${CCE_LINKER}")" == "lld" ]]; then
    LINKER_DRIVER_FLAGS=(-flavor gnu)
fi

"${CCE_LINKER}" \
    "${LINKER_DRIVER_FLAGS[@]}" \
    -m aicorelinux \
    -r \
    -Ttext=0 \
    --threads=1 \
    --allow-multiple-definition \
    mixed_w2w4_moe_ffn_mix_aic.o \
    mixed_w2w4_moe_ffn_mix_aiv.o \
    -static \
    -o mixed_w2w4_moe_ffn.o

echo ">>> [3/3] 编译 Host（四套 tiling builder 已头文件化）"
HOST_CXX_FLAGS=(
    -std=c++17 \
    -O2 \
    -g \
    -ffunction-sections \
    -fdata-sections \
    -I"${PROJECT_DIR}" \
    -I"${PLATFORM_DIR}/include" \
    -I"${PLATFORM_DIR}/include/experiment" \
    -I"${PLATFORM_DIR}/include/experiment/runtime"
)

"${CXX}" "${HOST_CXX_FLAGS[@]}" \
    mixed_w2w4_moe_ffn_host.cc \
    -L"${ASCEND_TOOLKIT_HOME}/runtime/lib64" \
    -L"${ASCEND_TOOLKIT_HOME}/lib64" \
    -lruntime \
    -lascendcl \
    -pthread \
    -Wl,--gc-sections \
    -o mixed_w2w4_moe_ffn_host

echo ">>> 编译完成"

if [[ $# -eq 7 ]]; then
    bash "${PROJECT_DIR}/run_mixed_w2w4_moe_ffn_case.sh" "$@"
fi
