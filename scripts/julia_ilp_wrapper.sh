#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <hypergraph.hgr> <num_parts> <ub_factor>" >&2
  exit 2
fi

HGR_FILE=$1
NUM_PARTS=$2
UB_FACTOR=$3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ILP_CANDIDATES=(
  "/home/norising/K_SpecPart_C/.external_build/ilp_part/ilp_part"
  "/home/norising/K_SpecPart/ilp_partitioner/build/ilp_part"
)
ORTOOLS_LIB_CANDIDATES=(
  "/home/norising/or-tools-9.4/install_make/lib"
  "/home/tool/ortools/install/CentOS7-gcc9/lib64"
  "/home/tool/ortools/install/CentOS7-gcc9/lib"
)
HMETIS_CANDIDATES=(
  "/home/norising/hmetis-1.5-linux/hmetis"
  "/home/fetzfs_projects/SpecPart/K_SpecPart/hmetis"
)
PART_FILE="${HGR_FILE}.part.${NUM_PARTS}"
METHOD_FILE="${PART_FILE}.method"

find_first_existing() {
  local candidate
  for candidate in "$@"; do
    if [ -e "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

find_loader() {
  local candidate
  for candidate in /lib/ld-linux.so.2 /lib32/ld-linux.so.2 /lib/i386-linux-gnu/ld-linux.so.2; do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

REAL_ILP=$(find_first_existing "${ILP_CANDIDATES[@]}" || true)
ORTOOLS_LIB=$(find_first_existing "${ORTOOLS_LIB_CANDIDATES[@]}" || true)
REAL_HMETIS=$(find_first_existing "${HMETIS_CANDIDATES[@]}" || true)
LINUX32_LOADER=$(find_loader || true)
ILP_TIMEOUT_SECONDS=${K_SPECPART_ILP_TIMEOUT_SECONDS:-120}
K_SPECPART_ILP_TIME_LIMIT=${K_SPECPART_ILP_TIME_LIMIT:-${ILP_TIMEOUT_SECONDS}}
export K_SPECPART_ILP_TIME_LIMIT

read_expected_vertices() {
  awk 'NR == 1 { print $2; exit }' "$HGR_FILE"
}

validate_partition_file() {
  local expected_vertices
  expected_vertices=$(read_expected_vertices)
  if [ -z "${expected_vertices}" ] || [ ! -s "${PART_FILE}" ]; then
    return 1
  fi
  awk -v expected="${expected_vertices}" -v num_parts="${NUM_PARTS}" '
    NF != 1 { valid = 0; exit 1 }
    $1 !~ /^-?[0-9]+$/ { valid = 0; exit 1 }
    {
      value = $1 + 0
      if (value < 0 || value >= num_parts) {
        valid = 0
        exit 1
      }
      count += 1
    }
    END {
      if (count != expected) {
        exit 1
      }
    }
  ' "${PART_FILE}"
}

run_hmetis_fallback() {
  local reason=$1
  local hmetis_status=0
  rm -f "${PART_FILE}" "${METHOD_FILE}"
  if [ -z "${REAL_HMETIS}" ]; then
    echo "wrapper fallback failed: ${reason}; hMETIS not found" >&2
    exit 1
  fi

  echo "wrapper fallback: ${reason}; switching to hMETIS" >&2
  if [ -x "${REAL_HMETIS}" ]; then
    { ( "${REAL_HMETIS}" "${HGR_FILE}" "${NUM_PARTS}" "${UB_FACTOR}" 10 1 1 1 0 0 ) >/dev/null 2>&1; } 2>/dev/null || hmetis_status=$?
  elif [ -n "${LINUX32_LOADER}" ]; then
    { ( "${LINUX32_LOADER}" "${REAL_HMETIS}" "${HGR_FILE}" "${NUM_PARTS}" "${UB_FACTOR}" 10 1 1 1 0 0 ) >/dev/null 2>&1; } 2>/dev/null || hmetis_status=$?
  else
    echo "wrapper fallback failed: hMETIS exists but is not directly executable and no 32-bit loader was found" >&2
    exit 1
  fi

  if validate_partition_file; then
    printf '%s\n' 'hmetis' > "${METHOD_FILE}"
    exit 0
  fi

  echo "wrapper fallback failed: hMETIS exited with status ${hmetis_status} or produced an invalid partition file" >&2
  exit 1
}

rm -f "$PART_FILE"
rm -f "$METHOD_FILE"

if [ -z "${REAL_ILP}" ]; then
  run_hmetis_fallback "ilp_part executable not found"
fi

if [ -z "${ORTOOLS_LIB}" ]; then
  ILP_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
else
  ILP_LD_LIBRARY_PATH="${ORTOOLS_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

run_ilp_once() {
  if command -v timeout >/dev/null 2>&1 && [ "${ILP_TIMEOUT_SECONDS}" -gt 0 ] 2>/dev/null; then
    LD_LIBRARY_PATH="${ILP_LD_LIBRARY_PATH}" \
      timeout --signal=TERM "${ILP_TIMEOUT_SECONDS}s" \
      "$REAL_ILP" "$HGR_FILE" "$NUM_PARTS" "$UB_FACTOR"
  else
    LD_LIBRARY_PATH="${ILP_LD_LIBRARY_PATH}" \
      "$REAL_ILP" "$HGR_FILE" "$NUM_PARTS" "$UB_FACTOR"
  fi
}

if run_ilp_once; then
  if validate_partition_file; then
    printf '%s\n' 'ilp' > "${METHOD_FILE}"
    exit 0
  fi
  run_hmetis_fallback "ilp_part completed but produced an invalid partition file"
fi

run_hmetis_fallback "ilp_part timed out or exited with a non-zero status"
