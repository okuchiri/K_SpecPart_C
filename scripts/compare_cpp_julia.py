#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CASES_FILE = "scripts/regression_cases.tsv"

CASE_COLUMNS = (
    "name",
    "hypergraph",
    "num_parts",
    "seed",
    "imb",
    "eigvecs",
    "refine_iters",
    "solver_iters",
    "best_solns",
    "ncycles",
    "fixed_file",
    "hint_file",
    "projection_strategy",
)

COMMON_OPTION_FLAGS = {
    "imb": "--imb",
    "eigvecs": "--eigvecs",
    "refine_iters": "--refine-iters",
    "solver_iters": "--solver-iters",
    "best_solns": "--best-solns",
    "ncycles": "--ncycles",
    "fixed_file": "--fixed",
    "hint_file": "--hint",
}

CPP_ONLY_OPTION_FLAGS = {
    "fixed_file": "--fixed-file",
    "hint_file": "--hint-file",
    "projection_strategy": "--projection-strategy",
}

CPP_CUT_RE = re.compile(r"Final cutsize\s+(\d+)")
JULIA_CUT_RE = re.compile(r"FINAL_CUT=(\d+)")


@dataclass
class Case:
    name: str
    hypergraph: Path
    num_parts: int
    seed: int
    imb: str = ""
    eigvecs: str = ""
    refine_iters: str = ""
    solver_iters: str = ""
    best_solns: str = ""
    ncycles: str = ""
    fixed_file: str = ""
    hint_file: str = ""
    projection_strategy: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the C++ and Julia K-SpecPart implementations on the same cases "
            "and compare the resulting partitions with a common evaluator."
        )
    )
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES_FILE,
        help="Tab-separated case file with a header row. Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/kspecpart_cpp_julia_regression",
        help="Directory used for logs, partitions, and summary files.",
    )
    parser.add_argument(
        "--cpp-bin",
        default="build/K_SpecPart",
        help="Path to the C++ executable. Default: %(default)s",
    )
    parser.add_argument(
        "--julia-runner",
        default="scripts/run_julia_specpart.sh",
        help="Path to the Julia wrapper script. Default: %(default)s",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Per-run timeout in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failed case instead of continuing.",
    )
    return parser.parse_args()


def iter_non_comment_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield line


def load_cases(path: Path) -> list[Case]:
    rows = list(iter_non_comment_lines(path))
    if not rows:
        raise ValueError(f"case file is empty: {path}")

    reader = csv.DictReader(rows, delimiter="\t")
    missing = [column for column in ("name", "hypergraph", "num_parts", "seed") if column not in reader.fieldnames]
    if missing:
        raise ValueError(f"case file is missing required columns: {', '.join(missing)}")

    cases: list[Case] = []
    for row in reader:
        name = (row.get("name") or "").strip()
        hypergraph = (row.get("hypergraph") or "").strip()
        if not name or not hypergraph:
            raise ValueError("each case must define non-empty name and hypergraph columns")
        values = {column: (row.get(column) or "").strip() for column in CASE_COLUMNS}
        cases.append(
            Case(
                name=values["name"],
                hypergraph=Path(values["hypergraph"]),
                num_parts=int(values["num_parts"]),
                seed=int(values["seed"]),
                imb=values["imb"],
                eigvecs=values["eigvecs"],
                refine_iters=values["refine_iters"],
                solver_iters=values["solver_iters"],
                best_solns=values["best_solns"],
                ncycles=values["ncycles"],
                fixed_file=values["fixed_file"],
                hint_file=values["hint_file"],
                projection_strategy=values["projection_strategy"],
            )
        )
    return cases


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_hypergraph(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        header = [int(token) for token in handle.readline().split()]
        if len(header) < 2:
            raise ValueError(f"invalid hypergraph header in {path}")

        num_hyperedges = header[0]
        num_vertices = header[1]
        wt_type = header[2] if len(header) > 2 else 0

        eptr = [0]
        eind: list[int] = []
        hwts = [1] * num_hyperedges

        for edge in range(num_hyperedges):
            line = handle.readline()
            if line == "":
                raise ValueError(f"unexpected EOF while reading hyperedges from {path}")
            tokens = [int(token) for token in line.split()]
            if not tokens:
                eptr.append(len(eind))
                continue
            start = 0
            if wt_type not in (0, 10):
                hwts[edge] = tokens[0]
                start = 1
            for vertex in tokens[start:]:
                eind.append(vertex - 1)
            eptr.append(len(eind))

        vwts = [1] * num_vertices
        vertex = 0
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if vertex >= num_vertices:
                break
            vwts[vertex] = int(stripped)
            vertex += 1

    return {
        "num_vertices": num_vertices,
        "num_hyperedges": num_hyperedges,
        "eptr": eptr,
        "eind": eind,
        "vwts": vwts,
        "hwts": hwts,
    }


def read_partition(path: Path) -> list[int]:
    partition: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                partition.append(int(stripped))
    return partition


def evaluate_partition(hypergraph: dict[str, object], num_parts: int, partition: list[int]) -> tuple[int, list[int]]:
    num_vertices = int(hypergraph["num_vertices"])
    num_hyperedges = int(hypergraph["num_hyperedges"])
    eptr = list(hypergraph["eptr"])
    eind = list(hypergraph["eind"])
    vwts = list(hypergraph["vwts"])
    hwts = list(hypergraph["hwts"])

    if len(partition) != num_vertices:
        raise ValueError(f"partition length {len(partition)} does not match hypergraph vertices {num_vertices}")

    cutsize = 0
    balance = [0] * num_parts

    for edge in range(num_hyperedges):
        start = eptr[edge]
        end = eptr[edge + 1]
        if start >= end:
            continue
        base_part = partition[eind[start]]
        cut = False
        for index in range(start + 1, end):
            if partition[eind[index]] != base_part:
                cut = True
                break
        if cut:
            cutsize += hwts[edge]

    for vertex, part in enumerate(partition):
        if 0 <= part < num_parts:
            balance[part] += vwts[vertex]

    return cutsize, balance


def normalize_partition_labels(partition: list[int]) -> list[int]:
    label_map: dict[int, int] = {}
    normalized: list[int] = []
    next_label = 0
    for label in partition:
        if label not in label_map:
            label_map[label] = next_label
            next_label += 1
        normalized.append(label_map[label])
    return normalized


def parse_reported_cut(text: str, regex: re.Pattern[str]) -> int | None:
    matches = regex.findall(text)
    return int(matches[-1]) if matches else None


def build_cpp_command(cpp_bin: Path, case: Case, output_path: Path) -> list[str]:
    command = [
        str(cpp_bin),
        "--hypergraph",
        str(case.hypergraph),
        "--num-parts",
        str(case.num_parts),
        "--seed",
        str(case.seed),
        "--output",
        str(output_path),
    ]

    for column, flag in COMMON_OPTION_FLAGS.items():
        value = getattr(case, column)
        if value:
            command.extend([CPP_ONLY_OPTION_FLAGS.get(column, flag), value])

    if case.projection_strategy:
        command.extend([CPP_ONLY_OPTION_FLAGS["projection_strategy"], case.projection_strategy])

    return command


def build_julia_command(julia_runner: Path, case: Case, output_path: Path) -> tuple[list[str], list[str]]:
    command = [
        str(julia_runner),
        "--hypergraph",
        str(case.hypergraph),
        "--num-parts",
        str(case.num_parts),
        "--seed",
        str(case.seed),
        "--output",
        str(output_path),
    ]

    notes: list[str] = []
    for column, flag in COMMON_OPTION_FLAGS.items():
        value = getattr(case, column)
        if value:
            command.extend([flag, value])

    if case.projection_strategy and case.projection_strategy.lower() != "lda":
        notes.append(f"Julia wrapper always uses lda; ignored projection_strategy={case.projection_strategy}")

    return command, notes


def run_and_log(command: list[str], log_path: Path, timeout_seconds: int) -> dict[str, object]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        runtime = time.monotonic() - started
        output = completed.stdout or ""
        log_path.write_text(
            "$ " + shlex.join(command) + "\n\n" + output,
            encoding="utf-8",
        )
        return {
            "exit_code": completed.returncode,
            "runtime_seconds": runtime,
            "timed_out": False,
            "output": output,
        }
    except subprocess.TimeoutExpired as exc:
        runtime = time.monotonic() - started
        output = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(
            "$ " + shlex.join(command) + "\n\nTIMEOUT after "
            + str(timeout_seconds)
            + " seconds\n\n"
            + output,
            encoding="utf-8",
        )
        return {
            "exit_code": None,
            "runtime_seconds": runtime,
            "timed_out": True,
            "output": output,
        }


def format_balance(balance: list[int] | None) -> str:
    if balance is None:
        return ""
    return "[" + ", ".join(str(value) for value in balance) + "]"


def append_note(notes: list[str], message: str) -> None:
    if message and message not in notes:
        notes.append(message)


def run_case(
    case: Case,
    cpp_bin: Path,
    julia_runner: Path,
    out_dir: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    cpp_partition_path = case_dir / "cpp.part"
    julia_partition_path = case_dir / "julia.part"
    cpp_log_path = case_dir / "cpp.log"
    julia_log_path = case_dir / "julia.log"

    notes: list[str] = []

    cpp_command = build_cpp_command(cpp_bin, case, cpp_partition_path)
    julia_command, julia_notes = build_julia_command(julia_runner, case, julia_partition_path)
    notes.extend(julia_notes)

    cpp_run = run_and_log(cpp_command, cpp_log_path, timeout_seconds)
    julia_run = run_and_log(julia_command, julia_log_path, timeout_seconds)

    result: dict[str, object] = {
        "name": case.name,
        "hypergraph": str(case.hypergraph),
        "num_parts": case.num_parts,
        "seed": case.seed,
        "projection_strategy": case.projection_strategy or "lda",
        "cpp_exit_code": cpp_run["exit_code"],
        "julia_exit_code": julia_run["exit_code"],
        "cpp_timed_out": cpp_run["timed_out"],
        "julia_timed_out": julia_run["timed_out"],
        "cpp_runtime_seconds": round(float(cpp_run["runtime_seconds"]), 3),
        "julia_runtime_seconds": round(float(julia_run["runtime_seconds"]), 3),
        "cpp_reported_cut": parse_reported_cut(str(cpp_run["output"]), CPP_CUT_RE),
        "julia_reported_cut": parse_reported_cut(str(julia_run["output"]), JULIA_CUT_RE),
        "cpp_eval_cut": None,
        "julia_eval_cut": None,
        "cpp_balance": None,
        "julia_balance": None,
        "same_eval_cut": False,
        "same_partition_exact": False,
        "same_partition_normalized": False,
        "partition_length_match": False,
        "notes": notes,
    }

    hypergraph = read_hypergraph(case.hypergraph)

    cpp_partition: list[int] | None = None
    julia_partition: list[int] | None = None

    if cpp_partition_path.is_file():
        cpp_partition = read_partition(cpp_partition_path)
        cpp_cut, cpp_balance = evaluate_partition(hypergraph, case.num_parts, cpp_partition)
        result["cpp_eval_cut"] = cpp_cut
        result["cpp_balance"] = cpp_balance
        if result["cpp_reported_cut"] is not None and result["cpp_reported_cut"] != cpp_cut:
            append_note(notes, "C++ reported cut differs from evaluator cut")
    else:
        append_note(notes, "C++ partition file missing")

    if julia_partition_path.is_file():
        julia_partition = read_partition(julia_partition_path)
        julia_cut, julia_balance = evaluate_partition(hypergraph, case.num_parts, julia_partition)
        result["julia_eval_cut"] = julia_cut
        result["julia_balance"] = julia_balance
        if result["julia_reported_cut"] is not None and result["julia_reported_cut"] != julia_cut:
            append_note(notes, "Julia reported cut differs from evaluator cut")
    else:
        append_note(notes, "Julia partition file missing")

    if cpp_partition is not None and julia_partition is not None:
        result["partition_length_match"] = len(cpp_partition) == len(julia_partition)
        result["same_partition_exact"] = cpp_partition == julia_partition
        result["same_partition_normalized"] = (
            normalize_partition_labels(cpp_partition) == normalize_partition_labels(julia_partition)
        )

    if result["cpp_eval_cut"] is not None and result["julia_eval_cut"] is not None:
        result["same_eval_cut"] = result["cpp_eval_cut"] == result["julia_eval_cut"]
        result["eval_cut_delta"] = int(result["cpp_eval_cut"]) - int(result["julia_eval_cut"])
    else:
        result["eval_cut_delta"] = None

    result["notes"] = notes
    return result


def write_tsv(results: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "name",
        "hypergraph",
        "num_parts",
        "seed",
        "projection_strategy",
        "cpp_exit_code",
        "julia_exit_code",
        "cpp_timed_out",
        "julia_timed_out",
        "cpp_runtime_seconds",
        "julia_runtime_seconds",
        "cpp_reported_cut",
        "julia_reported_cut",
        "cpp_eval_cut",
        "julia_eval_cut",
        "eval_cut_delta",
        "same_eval_cut",
        "same_partition_exact",
        "same_partition_normalized",
        "partition_length_match",
        "cpp_balance",
        "julia_balance",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["cpp_balance"] = format_balance(result.get("cpp_balance"))
            row["julia_balance"] = format_balance(result.get("julia_balance"))
            row["notes"] = " | ".join(result.get("notes", []))
            writer.writerow(row)


def write_markdown(results: list[dict[str, object]], path: Path) -> None:
    lines = [
        "| Case | k | Seed | C++ cut | Julia cut | Delta | Same cut | Same partition | C++ time (s) | Julia time (s) | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            "| {name} | {num_parts} | {seed} | {cpp_eval_cut} | {julia_eval_cut} | {eval_cut_delta} | {same_eval_cut} | {same_partition_normalized} | {cpp_runtime_seconds} | {julia_runtime_seconds} | {notes} |".format(
                name=result["name"],
                num_parts=result["num_parts"],
                seed=result["seed"],
                cpp_eval_cut="" if result["cpp_eval_cut"] is None else result["cpp_eval_cut"],
                julia_eval_cut="" if result["julia_eval_cut"] is None else result["julia_eval_cut"],
                eval_cut_delta="" if result["eval_cut_delta"] is None else result["eval_cut_delta"],
                same_eval_cut="yes" if result["same_eval_cut"] else "no",
                same_partition_normalized="yes" if result["same_partition_normalized"] else "no",
                cpp_runtime_seconds=result["cpp_runtime_seconds"],
                julia_runtime_seconds=result["julia_runtime_seconds"],
                notes="; ".join(result.get("notes", [])),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = repo_root / cases_path

    cpp_bin = Path(args.cpp_bin)
    if not cpp_bin.is_absolute():
        cpp_bin = repo_root / cpp_bin

    julia_runner = Path(args.julia_runner)
    if not julia_runner.is_absolute():
        julia_runner = repo_root / julia_runner

    ensure_file(cases_path, "case file")
    ensure_file(cpp_bin, "C++ executable")
    ensure_file(julia_runner, "Julia runner")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path)
    results: list[dict[str, object]] = []

    for case in cases:
        if not case.hypergraph.is_file():
            raise FileNotFoundError(f"hypergraph not found for case {case.name}: {case.hypergraph}")
        result = run_case(case, cpp_bin, julia_runner, out_dir, args.timeout_seconds)
        results.append(result)
        print(
            f"[{case.name}] cpp_cut={result['cpp_eval_cut']} "
            f"julia_cut={result['julia_eval_cut']} "
            f"same_cut={result['same_eval_cut']} "
            f"same_partition={result['same_partition_normalized']}"
        )
        if args.stop_on_failure and (
            result["cpp_exit_code"] not in (0, None)
            or result["julia_exit_code"] not in (0, None)
            or result["cpp_timed_out"]
            or result["julia_timed_out"]
        ):
            break

    write_tsv(results, out_dir / "summary.tsv")
    write_markdown(results, out_dir / "summary.md")
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Summary written to {out_dir / 'summary.tsv'}")
    print(f"Markdown summary written to {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
