#!/usr/bin/env bash

set -euo pipefail

HMETIS_CANDIDATES=(
  "/home/norising/hmetis-1.5-linux/hmetis"
  "/home/fetzfs_projects/SpecPart/K_SpecPart/hmetis"
)

find_hmetis() {
  local candidate
  for candidate in "${HMETIS_CANDIDATES[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_linux32_loader() {
  local candidate
  for candidate in /lib/ld-linux.so.2 /lib32/ld-linux.so.2 /lib/i386-linux-gnu/ld-linux.so.2; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

REAL_HMETIS=$(find_hmetis || true)
if [[ -z "${REAL_HMETIS}" ]]; then
  echo "hmetis_wrapper: hMETIS binary not found" >&2
  exit 127
fi

if [[ -x "${REAL_HMETIS}" ]]; then
  exec "${REAL_HMETIS}" "$@"
fi

LINUX32_LOADER=$(find_linux32_loader || true)
if [[ -n "${LINUX32_LOADER}" ]]; then
  exec "${LINUX32_LOADER}" "${REAL_HMETIS}" "$@"
fi

echo "hmetis_wrapper: hMETIS is not executable and no 32-bit loader was found" >&2
exit 126
