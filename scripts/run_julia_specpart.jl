#!/usr/bin/env julia

function require_value(args::Vector{String}, index::Int)
    if index > length(args)
        error("missing value for $(args[index - 1])")
    end
    return args[index]
end

mutable struct CliOptions
    hypergraph_file::String
    hypergraph_fixed_file::String
    hint_file::String
    output_file::String
    imb::Int
    num_parts::Int
    eigvecs::Int
    refine_iters::Int
    solver_iters::Int
    best_solns::Int
    ncycles::Int
    seed::Int
end

function parse_args(args::Vector{String})
    options = CliOptions("", "", "", "", 2, 2, 2, 2, 40, 3, 1, 0)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--hypergraph"
            options.hypergraph_file = require_value(args, i + 1)
            i += 2
        elseif arg == "--fixed"
            options.hypergraph_fixed_file = require_value(args, i + 1)
            i += 2
        elseif arg == "--hint"
            options.hint_file = require_value(args, i + 1)
            i += 2
        elseif arg == "--output"
            options.output_file = require_value(args, i + 1)
            i += 2
        elseif arg == "--imb"
            options.imb = parse(Int, require_value(args, i + 1))
            i += 2
        elseif arg == "--num-parts"
            options.num_parts = parse(Int, require_value(args, i + 1))
            i += 2
        elseif arg == "--eigvecs"
            options.eigvecs = parse(Int, require_value(args, i + 1))
            i += 2
        elseif arg == "--refine-iters"
            options.refine_iters = parse(Int, require_value(args, i + 1))
            i += 2
        elseif arg == "--solver-iters"
            options.solver_iters = parse(Int, require_value(args, i + 1))
            i += 2
        elseif arg == "--best-solns"
            options.best_solns = parse(Int, require_value(args, i + 1))
            i += 2
        elseif arg == "--ncycles"
            options.ncycles = parse(Int, require_value(args, i + 1))
            i += 2
        elseif arg == "--seed"
            options.seed = parse(Int, require_value(args, i + 1))
            i += 2
        else
            error("unknown argument: $arg")
        end
    end

    if isempty(options.hypergraph_file)
        error("--hypergraph is required")
    end
    return options
end

function write_partition_file(path::String, partition::Vector{Int})
    open(path, "w") do io
        for part in partition
            println(io, part)
        end
    end
end

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const JULIA_SOURCE_ROOT = "/home/norising/K_SpecPart"
const JULIA_RUNTIME_DIR = joinpath(tempdir(), "kspecpart-julia-ref")

mkpath(JULIA_RUNTIME_DIR)

include(joinpath(JULIA_SOURCE_ROOT, "specpart.jl"))
using .SpecPart

SpecPart.source_dir = JULIA_RUNTIME_DIR
SpecPart.metis_path = JULIA_SOURCE_ROOT
SpecPart.hmetis_path = "/home/fetzfs_projects/SpecPart/K_SpecPart/hmetis "
SpecPart.ilp_path = joinpath(REPO_ROOT, "scripts", "julia_ilp_wrapper.sh") * " "
SpecPart.triton_part_refiner_path = "true "

options = parse_args(ARGS)

partition, cutsize = SpecPart.specpart_run(
    options.hypergraph_file;
    hypergraph_fixed_file = options.hypergraph_fixed_file,
    hint_file = options.hint_file,
    imb = options.imb,
    num_parts = options.num_parts,
    eigvecs = options.eigvecs,
    refine_iters = options.refine_iters,
    solver_iters = options.solver_iters,
    best_solns = options.best_solns,
    ncycles = options.ncycles,
    seed = options.seed,
)

if !isempty(options.output_file)
    write_partition_file(options.output_file, partition)
    println("OUTPUT_FILE=" * options.output_file)
end

println("FINAL_CUT=" * string(cutsize))
println("PART_SIZE=" * string(length(partition)))
