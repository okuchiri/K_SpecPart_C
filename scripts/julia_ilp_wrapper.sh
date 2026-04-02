#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <hypergraph.hgr> <num_parts> <ub_factor>" >&2
  exit 2
fi

HGR_FILE=$1
NUM_PARTS=$2
UB_FACTOR=$3

REAL_ILP="/home/norising/K_SpecPart/ilp_partitioner/build/ilp_part"
REAL_HMETIS="/home/fetzfs_projects/SpecPart/K_SpecPart/hmetis"
ORTOOLS_LIB="/home/norising/or-tools-9.4/install_make/lib"
PART_FILE="${HGR_FILE}.part.${NUM_PARTS}"

rm -f "$PART_FILE"

if LD_LIBRARY_PATH="${ORTOOLS_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$REAL_ILP" "$HGR_FILE" "$NUM_PARTS" "$UB_FACTOR"; then
  if [ -s "$PART_FILE" ]; then
    exit 0
  fi
fi

rm -f "$PART_FILE"
exec "$REAL_HMETIS" "$HGR_FILE" "$NUM_PARTS" "$UB_FACTOR" 10 1 1 1 0 0
