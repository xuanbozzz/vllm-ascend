#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -ne 7 ]]; then
    echo "Usage: $0 <num_experts> <K> <N> <D> <device_id> <warmup_iters> <test_iters>" >&2
    exit 2
fi

ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-prof_mixed_w2w4}"

if [[ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
elif [[ -f "${ASCEND_TOOLKIT_HOME}/bin/setenv.bash" ]]; then
    source "${ASCEND_TOOLKIT_HOME}/bin/setenv.bash"
fi
set -u

export RUN_MODE=device
export PROFILE=1
export PATH="${ASCEND_TOOLKIT_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/runtime/lib64:${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${PROJECT_DIR}/mixed_w2w4_moe_ffn_host" ]]; then
    echo "mixed_w2w4_moe_ffn_host does not exist; run 'bash run_mixed_w2w4_moe_ffn_esl.sh' first." >&2
    exit 1
fi
if ! command -v msprof >/dev/null 2>&1; then
    echo "Cannot find msprof." >&2
    exit 1
fi

msprof \
    --application="./mixed_w2w4_moe_ffn_host $*" \
    --output="${PROFILE_OUTPUT}"
