#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

exec python3 "${PROJECT_DIR}/run_mixed_w2w4_moe_ffn_cases.py"
