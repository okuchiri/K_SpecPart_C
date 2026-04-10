#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

mkdir -p /tmp/julia_depot

export JULIA_DEPOT_PATH="/tmp/julia_depot:/home/norising/.julia"
export LD_LIBRARY_PATH="/home/norising/or-tools-9.4/install_make/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export http_proxy=
export https_proxy=
export HTTP_PROXY=
export HTTPS_PROXY=

JULIA_PROJECT_ROOT=${K_SPECPART_JULIA_PROJECT:-"${REPO_ROOT}/julia_ref_env"}

exec julia --project="${JULIA_PROJECT_ROOT}" "${SCRIPT_DIR}/run_julia_specpart.jl" "$@"
