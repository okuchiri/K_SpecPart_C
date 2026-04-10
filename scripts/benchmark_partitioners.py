#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from compare_cpp_julia import (
    CPP_CUT_RE,
    JULIA_CUT_RE,
    evaluate_partition,
    parse_reported_cut,
    read_hypergraph,
    read_partition,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KAHYPAR_BINARY = REPO_ROOT / "third_party/KaHyPar/build/kahypar/application/KaHyPar"
DEFAULT_KAHYPAR_PRESET = REPO_ROOT / "third_party/KaHyPar/config/cut_kKaHyPar_sea20.ini"


@dataclass(frozen=True)
class BenchmarkCase:
    hypergraph: Path
    num_parts: int
    imb: int
    seed: int


@dataclass
class RunResult:
    tool: str
    repeat: int
    seed: int
    status: str
    runtime_seconds: float | None
    cutsize: int | None
    balance: list[int] | None
    partition_file: str | None
    log_file: str
    command: list[str] | None = None
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark K-SpecPart C++, Julia, hMETIS, and KaHyPar on one or more "
            "hypergraphs across multiple (k, imbalance) settings."
        )
    )
    parser.add_argument("--hypergraph", help="Single input hypergraph path")
    parser.add_argument(
        "--hypergraphs",
        nargs="*",
        default=[],
        help="Additional input hypergraph paths",
    )
    parser.add_argument(
        "--hypergraphs-file",
        default="",
        help="Optional file containing one hypergraph path per line",
    )
    parser.add_argument("--num-parts", type=int, help="Single number of parts")
    parser.add_argument(
        "--num-parts-list",
        default="",
        help="Comma-separated numbers of parts, for example: 2,3,4",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base seed used by C++/Julia/KaHyPar")
    parser.add_argument(
        "--seed-list",
        default="",
        help="Comma-separated case seeds. Default uses --seed only",
    )
    parser.add_argument("--imb", type=int, default=2, help="Single imbalance parameter")
    parser.add_argument(
        "--imb-list",
        default="",
        help="Comma-separated imbalance parameters, for example: 2,5,10",
    )
    parser.add_argument("--refine-iters", type=int, default=2, help="Refinement iterations for C++/Julia")
    parser.add_argument("--eigvecs", type=int, default=2, help="Eigenvectors for C++/Julia")
    parser.add_argument("--solver-iters", type=int, default=40, help="LOBPCG iterations for C++/Julia")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per tool")
    parser.add_argument(
        "--repeat-seed-mode",
        choices=("fixed", "increment"),
        default="fixed",
        help=(
            "How seeds change across repeats. 'fixed' measures nondeterminism under the same seed; "
            "'increment' uses seed, seed+1, seed+2, ..."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Per-run timeout")
    parser.add_argument("--out-dir", default="/tmp/kspecpart_benchmarks", help="Benchmark output directory")
    parser.add_argument("--cpp-bin", default="build/K_SpecPart", help="C++ executable")
    parser.add_argument("--julia-runner", default="scripts/run_julia_specpart.sh", help="Julia runner script")
    parser.add_argument("--hmetis-wrapper", default="scripts/hmetis_wrapper.sh", help="hMETIS wrapper script")
    parser.add_argument(
        "--shared-initial-hint",
        action="store_true",
        help=(
            "Generate one shared hMETIS initial hint per repeat and pass it to both C++ and Julia. "
            "Useful for isolating implementation differences from hMETIS nondeterminism."
        ),
    )
    parser.add_argument(
        "--kahypar-binary",
        default="",
        help="KaHyPar executable. Default auto-detects the locally built original KaHyPar first",
    )
    parser.add_argument(
        "--kahypar-preset",
        default="",
        help="KaHyPar preset ini file. Default uses third_party/KaHyPar/config/cut_kKaHyPar_sea20.ini when available",
    )
    parser.add_argument(
        "--kahypar-objective",
        default="cut",
        help="KaHyPar objective. Default: %(default)s",
    )
    parser.add_argument(
        "--kahypar-mode",
        default="direct",
        help="KaHyPar partitioning mode. Default: %(default)s",
    )
    parser.add_argument(
        "--kahypar-cmd-template",
        default=os.environ.get("K_SPECPART_KAHYPAR_CMD_TEMPLATE", ""),
        help=(
            "Optional KaHyPar command template. Available placeholders: "
            "{hypergraph} {num_parts} {imb} {epsilon} {seed} {partition} {partition_dir} {preset}. "
            "If the template does not write to {partition}, the script also checks KaHyPar's default output filename."
        ),
    )
    parser.add_argument("--skip-cpp", action="store_true")
    parser.add_argument("--skip-julia", action="store_true")
    parser.add_argument("--skip-hmetis", action="store_true")
    parser.add_argument("--skip-kahypar", action="store_true")

    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    if args.hypergraph is None and not args.hypergraphs and not args.hypergraphs_file:
        parser.error("at least one hypergraph must be provided via --hypergraph, --hypergraphs, or --hypergraphs-file")

    if args.num_parts is None and not args.num_parts_list:
        parser.error("at least one k must be provided via --num-parts or --num-parts-list")

    return args


def parse_csv(raw: str) -> list[str]:
    values: list[str] = []
    for token in raw.replace("\n", ",").split(","):
        stripped = token.strip()
        if stripped:
            values.append(stripped)
    return values


def ordered_unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def ordered_unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def iter_non_comment_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def resolve_hypergraphs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.hypergraph:
        paths.append(Path(args.hypergraph).expanduser().resolve())
    paths.extend(Path(item).expanduser().resolve() for item in args.hypergraphs)
    if args.hypergraphs_file:
        file_path = Path(args.hypergraphs_file).expanduser().resolve()
        for line in iter_non_comment_lines(file_path):
            candidate = Path(line).expanduser()
            if not candidate.is_absolute():
                candidate = file_path.parent / candidate
            paths.append(candidate.resolve())

    unique = ordered_unique_paths(paths)
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"hypergraph not found: {path}")
    return unique


def resolve_num_parts_values(args: argparse.Namespace) -> list[int]:
    values: list[int] = []
    if args.num_parts is not None:
        values.append(args.num_parts)
    values.extend(int(token) for token in parse_csv(args.num_parts_list))
    values = ordered_unique_ints(values)
    if not values:
        raise ValueError("no num-parts values were resolved")
    if any(value <= 0 for value in values):
        raise ValueError("all num-parts values must be positive")
    return values


def resolve_seed_values(args: argparse.Namespace) -> list[int]:
    values = [args.seed]
    values.extend(int(token) for token in parse_csv(args.seed_list))
    return ordered_unique_ints(values)


def resolve_imb_values(args: argparse.Namespace) -> list[int]:
    values = [args.imb]
    values.extend(int(token) for token in parse_csv(args.imb_list))
    values = ordered_unique_ints(values)
    if any(value < 0 for value in values):
        raise ValueError("all imbalance values must be non-negative")
    return values


def build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    hypergraphs = resolve_hypergraphs(args)
    num_parts_values = resolve_num_parts_values(args)
    seed_values = resolve_seed_values(args)
    imb_values = resolve_imb_values(args)

    cases: list[BenchmarkCase] = []
    for hypergraph in hypergraphs:
        for num_parts in num_parts_values:
            for imb in imb_values:
                for seed in seed_values:
                    cases.append(BenchmarkCase(hypergraph=hypergraph, num_parts=num_parts, imb=imb, seed=seed))
    return cases


def case_name(case: BenchmarkCase, args: argparse.Namespace) -> str:
    suffix = f"{case.hypergraph.stem}.k{case.num_parts}.imb{case.imb}.seed{case.seed}.r{args.repeats}"
    if args.repeat_seed_mode != "fixed":
        suffix += f".seedmode-{args.repeat_seed_mode}"
    if args.shared_initial_hint:
        suffix += ".sharedhint"
    return suffix


def repeat_seed(base_seed: int, repeat: int, seed_mode: str) -> int:
    if seed_mode == "increment":
        return base_seed + repeat - 1
    return base_seed


def run_and_capture(command: list[str], log_path: Path, timeout_seconds: int, cwd: Path | None = None) -> tuple[str, float, str]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
        )
        runtime = time.monotonic() - started
        output = completed.stdout or ""
        log_path.write_text("$ " + shlex.join(command) + "\n\n" + output, encoding="utf-8")
        return ("ok" if completed.returncode == 0 else f"exit_{completed.returncode}", runtime, output)
    except subprocess.TimeoutExpired as exc:
        runtime = time.monotonic() - started
        output = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(
            "$ " + shlex.join(command) + f"\n\nTIMEOUT after {timeout_seconds} seconds\n\n" + output,
            encoding="utf-8",
        )
        return ("timeout", runtime, output)


def evaluate_partition_file(
    hypergraph: dict[str, object],
    num_parts: int,
    partition_path: Path,
) -> tuple[int, list[int]]:
    partition = read_partition(partition_path)
    return evaluate_partition(hypergraph, num_parts, partition)


def maybe_evaluate_partition_file(
    hypergraph: dict[str, object],
    num_parts: int,
    partition_path: Path,
) -> tuple[int | None, list[int] | None, str]:
    try:
        cutsize, balance = evaluate_partition_file(hypergraph, num_parts, partition_path)
        return cutsize, balance, ""
    except ValueError as exc:
        return None, None, str(exc)


def append_note(note: str, extra: str) -> str:
    if not extra:
        return note
    if not note:
        return extra
    return f"{note} | {extra}"


def numeric_stats(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "range": None,
            "mean": None,
            "variance": None,
            "stdev": None,
        }

    values_as_float = [float(value) for value in values]
    minimum = min(values_as_float)
    maximum = max(values_as_float)
    variance = statistics.pvariance(values_as_float) if len(values_as_float) > 1 else 0.0
    stdev = statistics.pstdev(values_as_float) if len(values_as_float) > 1 else 0.0
    return {
        "count": len(values),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "mean": statistics.fmean(values_as_float),
        "variance": variance,
        "stdev": stdev,
    }


def format_kahypar_epsilon(epsilon: float) -> str:
    text = f"{epsilon:.12f}".rstrip("0").rstrip(".")
    return text if text else "0"


def resolve_kahypar_binary(args: argparse.Namespace) -> Path | None:
    if args.kahypar_binary:
        candidate = Path(args.kahypar_binary).expanduser().resolve()
        return candidate if candidate.is_file() else None

    if DEFAULT_KAHYPAR_BINARY.is_file():
        return DEFAULT_KAHYPAR_BINARY

    for executable in ("KaHyPar", "kahypar"):
        found = shutil.which(executable)
        if found:
            candidate = Path(found).resolve()
            if candidate.is_file():
                return candidate

    return None


def resolve_kahypar_preset(args: argparse.Namespace) -> Path | None:
    if args.kahypar_preset:
        candidate = Path(args.kahypar_preset).expanduser().resolve()
        return candidate if candidate.is_file() else None
    return DEFAULT_KAHYPAR_PRESET if DEFAULT_KAHYPAR_PRESET.is_file() else None


def kahypar_default_partition_path(input_path: Path, num_parts: int, epsilon: float, seed: int) -> Path:
    epsilon_text = format_kahypar_epsilon(epsilon)
    return input_path.parent / f"{input_path.name}.part{num_parts}.epsilon{epsilon_text}.seed{seed}.KaHyPar"


def generate_shared_initial_hint(
    args: argparse.Namespace,
    case: BenchmarkCase,
    case_dir: Path,
    repeat: int,
) -> tuple[Path | None, str]:
    actual_seed = repeat_seed(case.seed, repeat, args.repeat_seed_mode)
    stem = f"shared-hint.repeat-{repeat}.seed-{actual_seed}"
    run_dir = case_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.hgr"
    raw_part_path = run_dir / f"input.hgr.part.{case.num_parts}"
    part_path = case_dir / f"{stem}.part"
    log_path = case_dir / f"{stem}.log"
    shutil.copyfile(case.hypergraph, input_path)
    command = [
        args.hmetis_wrapper,
        str(input_path),
        str(case.num_parts),
        str(case.imb),
        "10",
        "1",
        "1",
        "1",
        "0",
        "0",
    ]
    status, _, _ = run_and_capture(command, log_path, args.timeout_seconds)
    if status != "ok":
        return None, f"shared hMETIS hint failed ({status})"
    if not raw_part_path.is_file():
        return None, "shared hMETIS hint partition file missing"
    shutil.copyfile(raw_part_path, part_path)
    note = f"shared hMETIS hint: {part_path.name}"
    if args.repeat_seed_mode != "fixed":
        note = append_note(note, "hMETIS wrapper does not expose an explicit seed argument")
    return part_path, note


def run_cpp(
    args: argparse.Namespace,
    case: BenchmarkCase,
    hypergraph: dict[str, object],
    case_dir: Path,
    repeat: int,
    shared_hint_path: Path | None = None,
    shared_hint_note: str = "",
) -> RunResult:
    actual_seed = repeat_seed(case.seed, repeat, args.repeat_seed_mode)
    stem = f"cpp.repeat-{repeat}.seed-{actual_seed}"
    part_path = case_dir / f"{stem}.part"
    log_path = case_dir / f"{stem}.log"
    command = [
        args.cpp_bin,
        "--hypergraph",
        str(case.hypergraph),
        "--num-parts",
        str(case.num_parts),
        "--seed",
        str(actual_seed),
        "--imb",
        str(case.imb),
        "--refine-iters",
        str(args.refine_iters),
        "--eigvecs",
        str(args.eigvecs),
        "--solver-iters",
        str(args.solver_iters),
        "--output",
        str(part_path),
    ]
    if shared_hint_path is not None:
        command.extend(["--hint-file", str(shared_hint_path)])
    status, runtime, output = run_and_capture(command, log_path, args.timeout_seconds)
    cutsize = None
    balance = None
    note = shared_hint_note
    if status == "ok" and part_path.is_file():
        cutsize, balance, extra_note = maybe_evaluate_partition_file(hypergraph, case.num_parts, part_path)
        note = append_note(note, extra_note)
        if cutsize is None:
            cutsize = parse_reported_cut(output, CPP_CUT_RE)
            if cutsize is not None:
                note = append_note(note, "used reported C++ cutsize")
    elif status == "ok":
        note = append_note(note, "partition file missing")
    return RunResult("cpp", repeat, actual_seed, status, runtime, cutsize, balance, str(part_path) if part_path.exists() else None, str(log_path), command, note)


def run_julia(
    args: argparse.Namespace,
    case: BenchmarkCase,
    hypergraph: dict[str, object],
    case_dir: Path,
    repeat: int,
    shared_hint_path: Path | None = None,
    shared_hint_note: str = "",
) -> RunResult:
    actual_seed = repeat_seed(case.seed, repeat, args.repeat_seed_mode)
    stem = f"julia.repeat-{repeat}.seed-{actual_seed}"
    part_path = case_dir / f"{stem}.part"
    log_path = case_dir / f"{stem}.log"
    command = [
        args.julia_runner,
        "--hypergraph",
        str(case.hypergraph),
        "--num-parts",
        str(case.num_parts),
        "--seed",
        str(actual_seed),
        "--imb",
        str(case.imb),
        "--refine-iters",
        str(args.refine_iters),
        "--eigvecs",
        str(args.eigvecs),
        "--solver-iters",
        str(args.solver_iters),
        "--output",
        str(part_path),
    ]
    if shared_hint_path is not None:
        command.extend(["--hint", str(shared_hint_path)])
    status, runtime, output = run_and_capture(command, log_path, args.timeout_seconds)
    cutsize = None
    balance = None
    note = shared_hint_note
    if status == "ok" and part_path.is_file():
        cutsize, balance, extra_note = maybe_evaluate_partition_file(hypergraph, case.num_parts, part_path)
        note = append_note(note, extra_note)
        if cutsize is None:
            cutsize = parse_reported_cut(output, JULIA_CUT_RE)
            if cutsize is not None:
                note = append_note(note, "used reported Julia cutsize")
    elif status == "ok":
        note = append_note(note, "partition file missing")
    return RunResult("julia", repeat, actual_seed, status, runtime, cutsize, balance, str(part_path) if part_path.exists() else None, str(log_path), command, note)


def run_hmetis(args: argparse.Namespace, case: BenchmarkCase, hypergraph: dict[str, object], case_dir: Path, repeat: int) -> RunResult:
    actual_seed = repeat_seed(case.seed, repeat, args.repeat_seed_mode)
    stem = f"hmetis.repeat-{repeat}.seed-{actual_seed}"
    run_dir = case_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.hgr"
    raw_part_path = run_dir / f"input.hgr.part.{case.num_parts}"
    part_path = case_dir / f"{stem}.part"
    log_path = case_dir / f"{stem}.log"
    shutil.copyfile(case.hypergraph, input_path)
    command = [
        args.hmetis_wrapper,
        str(input_path),
        str(case.num_parts),
        str(case.imb),
        "10",
        "1",
        "1",
        "1",
        "0",
        "0",
    ]
    status, runtime, _ = run_and_capture(command, log_path, args.timeout_seconds)
    cutsize = None
    balance = None
    note = ""
    resolved_part_path: Path | None = None
    if status == "ok" and raw_part_path.is_file():
        shutil.copyfile(raw_part_path, part_path)
        resolved_part_path = part_path
        cutsize, balance, note = maybe_evaluate_partition_file(hypergraph, case.num_parts, part_path)
    elif status == "ok":
        note = "partition file missing"
    if args.repeat_seed_mode != "fixed":
        note = append_note(note, "hMETIS wrapper does not expose an explicit seed argument")
    return RunResult(
        "hmetis",
        repeat,
        actual_seed,
        status,
        runtime,
        cutsize,
        balance,
        str(resolved_part_path) if resolved_part_path is not None and resolved_part_path.exists() else None,
        str(log_path),
        command,
        note,
    )


def run_kahypar(args: argparse.Namespace, case: BenchmarkCase, hypergraph: dict[str, object], case_dir: Path, repeat: int) -> RunResult:
    actual_seed = repeat_seed(case.seed, repeat, args.repeat_seed_mode)
    stem = f"kahypar.repeat-{repeat}.seed-{actual_seed}"
    log_path = case_dir / f"{stem}.log"

    binary = resolve_kahypar_binary(args)
    preset = resolve_kahypar_preset(args)
    if not args.kahypar_cmd_template and (binary is None or preset is None):
        note_parts: list[str] = []
        if binary is None:
            note_parts.append("KaHyPar binary not found")
        if preset is None:
            note_parts.append("KaHyPar preset not found")
        return RunResult("kahypar", repeat, actual_seed, "unavailable", None, None, None, None, str(log_path), None, "; ".join(note_parts))

    run_dir = case_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.hgr"
    part_path = case_dir / f"{stem}.part"
    shutil.copyfile(case.hypergraph, input_path)

    epsilon = max(0.0, case.imb / 100.0)
    if args.kahypar_cmd_template:
        rendered = args.kahypar_cmd_template.format(
            hypergraph=str(input_path),
            num_parts=case.num_parts,
            imb=case.imb,
            epsilon=epsilon,
            seed=actual_seed,
            partition=str(part_path),
            partition_dir=str(run_dir),
            preset=str(preset) if preset is not None else "",
        )
        command = shlex.split(rendered)
        cwd = run_dir
    else:
        command = [
            str(binary),
            "-h",
            input_path.name,
            "-k",
            str(case.num_parts),
            "-e",
            format_kahypar_epsilon(epsilon),
            "-o",
            args.kahypar_objective,
            "-m",
            args.kahypar_mode,
            "-p",
            str(preset),
            "--seed",
            str(actual_seed),
            "-w",
            "true",
            "-q",
            "true",
        ]
        cwd = run_dir

    status, runtime, _ = run_and_capture(command, log_path, args.timeout_seconds, cwd=cwd)
    cutsize = None
    balance = None
    note = ""

    candidate_paths = [part_path, kahypar_default_partition_path(input_path, case.num_parts, epsilon, actual_seed)]
    resolved_part_path: Path | None = None
    for candidate in candidate_paths:
        if candidate.is_file():
            resolved_part_path = candidate
            break

    if status == "ok" and resolved_part_path is not None:
        if resolved_part_path != part_path:
            shutil.copyfile(resolved_part_path, part_path)
            resolved_part_path = part_path
            note = "copied default KaHyPar partition output"
        cutsize, balance, extra_note = maybe_evaluate_partition_file(hypergraph, case.num_parts, resolved_part_path)
        if extra_note:
            note = (note + " | " if note else "") + extra_note
    elif status == "ok":
        note = "partition file missing"

    return RunResult(
        "kahypar",
        repeat,
        actual_seed,
        status,
        runtime,
        cutsize,
        balance,
        str(resolved_part_path) if resolved_part_path is not None and resolved_part_path.exists() else None,
        str(log_path),
        command,
        note,
    )


def summarize(case: BenchmarkCase, args: argparse.Namespace, case_dir: Path, results: list[RunResult]) -> dict[str, object]:
    summary: dict[str, object] = {
        "case": {
            "name": case_name(case, args),
            "hypergraph": str(case.hypergraph),
            "num_parts": case.num_parts,
            "imb": case.imb,
            "seed": case.seed,
            "repeats": args.repeats,
            "repeat_seed_mode": args.repeat_seed_mode,
            "shared_initial_hint": args.shared_initial_hint,
            "refine_iters": args.refine_iters,
            "eigvecs": args.eigvecs,
            "solver_iters": args.solver_iters,
            "artifacts_dir": str(case_dir),
        },
        "runs": [asdict(result) for result in results],
        "tools": {},
    }
    by_tool: dict[str, list[RunResult]] = {}
    for result in results:
        by_tool.setdefault(result.tool, []).append(result)

    for tool, tool_results in by_tool.items():
        ok_results = [result for result in tool_results if result.status == "ok" and result.cutsize is not None]
        cut_values = [result.cutsize for result in ok_results if result.cutsize is not None]
        runtime_values = [result.runtime_seconds for result in ok_results if result.runtime_seconds is not None]
        cut_stats = numeric_stats(cut_values)
        runtime_stats = numeric_stats(runtime_values)
        notes = sorted({result.note for result in tool_results if result.note})
        summary["tools"][tool] = {
            "statuses": [result.status for result in tool_results],
            "status_counts": dict(Counter(result.status for result in tool_results)),
            "seeds": [result.seed for result in tool_results],
            "cutsizes": cut_values,
            "runtimes": runtime_values,
            "notes": notes,
            "total_runs": len(tool_results),
            "successful_runs": len(ok_results),
            "unique_cutsize_count": len(set(cut_values)),
            "best_cutsize": min(cut_values, default=None),
            "worst_cutsize": max(cut_values, default=None),
            "cutsize_mean": cut_stats["mean"],
            "cutsize_variance": cut_stats["variance"],
            "cutsize_stdev": cut_stats["stdev"],
            "cutsize_range": cut_stats["range"],
            "best_runtime_seconds": min(runtime_values, default=None),
            "worst_runtime_seconds": max(runtime_values, default=None),
            "runtime_mean_seconds": runtime_stats["mean"],
            "runtime_variance_seconds": runtime_stats["variance"],
            "runtime_stdev_seconds": runtime_stats["stdev"],
            "runtime_range_seconds": runtime_stats["range"],
        }
    return summary


def format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_case_markdown(summary: dict[str, object], path: Path) -> None:
    case = summary["case"]
    lines = [
        f"# {case['name']}",
        "",
        f"- hypergraph: `{case['hypergraph']}`",
        f"- k: `{case['num_parts']}`",
        f"- imbalance: `{case['imb']}`",
        f"- base seed: `{case['seed']}`",
        f"- repeats: `{case['repeats']}`",
        f"- repeat seed mode: `{case['repeat_seed_mode']}`",
        f"- shared initial hint: `{case['shared_initial_hint']}`",
        f"- refine iters: `{case['refine_iters']}`",
        f"- eigvecs: `{case['eigvecs']}`",
        f"- solver iters: `{case['solver_iters']}`",
        "",
        "## Tool Summary",
        "",
        "| Tool | OK/Total | Unique Cuts | Best Cut | Mean Cut | Cut Stdev | Cut Variance | Worst Cut | Mean Time (s) | Time Stdev (s) | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for tool, data in summary["tools"].items():
        notes = "; ".join(data["notes"])
        lines.append(
            "| "
            + " | ".join(
                [
                    tool,
                    f"{data['successful_runs']}/{data['total_runs']}",
                    format_number(data["unique_cutsize_count"], 0),
                    format_number(data["best_cutsize"], 0),
                    format_number(data["cutsize_mean"]),
                    format_number(data["cutsize_stdev"]),
                    format_number(data["cutsize_variance"]),
                    format_number(data["worst_cutsize"], 0),
                    format_number(data["runtime_mean_seconds"]),
                    format_number(data["runtime_stdev_seconds"]),
                    notes,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Run Details",
            "",
            "| Tool | Repeat | Seed | Status | Cutsize | Runtime (s) | Balance | Note |",
            "| --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )

    for result in summary["runs"]:
        balance = "" if result["balance"] is None else "[" + ", ".join(str(v) for v in result["balance"]) + "]"
        lines.append(
            f"| {result['tool']} | {result['repeat']} | {result['seed']} | {result['status']} | "
            f"{format_number(result['cutsize'], 0)} | {format_number(result['runtime_seconds'])} | "
            f"{balance} | {result['note']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_batch_markdown(summaries: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Benchmark Batch Summary",
        "",
        "| Graph | k | imb | seed | repeats | seed mode | Tool | OK/Total | Best Cut | Mean Cut | Cut Stdev | Worst Cut | Mean Time (s) | Time Stdev (s) | Case Dir |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for summary in summaries:
        case = summary["case"]
        for tool, data in summary["tools"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        Path(case["hypergraph"]).stem,
                        str(case["num_parts"]),
                        str(case["imb"]),
                        str(case["seed"]),
                        str(case["repeats"]),
                        str(case["repeat_seed_mode"]),
                        tool,
                        f"{data['successful_runs']}/{data['total_runs']}",
                        format_number(data["best_cutsize"], 0),
                        format_number(data["cutsize_mean"]),
                        format_number(data["cutsize_stdev"]),
                        format_number(data["worst_cutsize"], 0),
                        format_number(data["runtime_mean_seconds"]),
                        format_number(data["runtime_stdev_seconds"]),
                        str(case["artifacts_dir"]),
                    ]
                )
                + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(
    args: argparse.Namespace,
    case: BenchmarkCase,
    hypergraph: dict[str, object],
    out_dir: Path,
) -> dict[str, object]:
    case_dir = out_dir / case_name(case, args)
    case_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []

    for repeat in range(1, args.repeats + 1):
        shared_hint_path: Path | None = None
        shared_hint_note = ""
        if args.shared_initial_hint and (not args.skip_cpp or not args.skip_julia):
            shared_hint_path, shared_hint_note = generate_shared_initial_hint(args, case, case_dir, repeat)
        if not args.skip_cpp:
            results.append(run_cpp(args, case, hypergraph, case_dir, repeat, shared_hint_path, shared_hint_note))
        if not args.skip_julia:
            results.append(run_julia(args, case, hypergraph, case_dir, repeat, shared_hint_path, shared_hint_note))
        if not args.skip_hmetis:
            results.append(run_hmetis(args, case, hypergraph, case_dir, repeat))
        if not args.skip_kahypar:
            results.append(run_kahypar(args, case, hypergraph, case_dir, repeat))

    summary = summarize(case, args, case_dir, results)
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_case_markdown(summary, case_dir / "summary.md")
    return summary


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args)
    hypergraph_cache: dict[Path, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []

    print(
        f"Running {len(cases)} benchmark case(s) with repeats={args.repeats}, "
        f"repeat_seed_mode={args.repeat_seed_mode}"
    )
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case_name(case, args)}")
        if case.hypergraph not in hypergraph_cache:
            hypergraph_cache[case.hypergraph] = read_hypergraph(case.hypergraph)
        hypergraph = hypergraph_cache[case.hypergraph]
        summary = run_case(args, case, hypergraph, out_dir)
        summaries.append(summary)
        for tool, data in summary["tools"].items():
            print(
                f"  {tool}: ok={data['successful_runs']}/{data['total_runs']} "
                f"best_cut={data['best_cutsize']} mean_cut={format_number(data['cutsize_mean'])} "
                f"cut_stdev={format_number(data['cutsize_stdev'])} "
                f"mean_time={format_number(data['runtime_mean_seconds'])}"
            )

    batch_summary = {"cases": summaries}
    (out_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    write_batch_markdown(summaries, out_dir / "batch_summary.md")

    print(f"Batch artifacts: {out_dir}")
    print(f"Batch markdown summary: {out_dir / 'batch_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
