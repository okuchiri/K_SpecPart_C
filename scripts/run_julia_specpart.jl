#!/usr/bin/env julia

using Printf
using LinearAlgebra
using SparseArrays

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

function stage_input_file(path::String, run_dir::String, prefix::String)
    if isempty(path)
        return path
    end
    source = normpath(path)
    destination = joinpath(run_dir, prefix * "-" * basename(source))
    cp(source, destination; force=true)
    return destination
end

function generate_initial_hint_file(hypergraph_file::String, num_parts::Int, imb::Int, run_dir::String)
    hint_base = joinpath(run_dir, "initial_hint.hgr")
    cp(normpath(hypergraph_file), hint_base; force=true)

    runs = 10
    ctype = 1
    rtype = 1
    vcycle = 1
    reconst = 0
    dbglvl = 0
    hmetis_string =
        SpecPart.hmetis_path *
        hint_base * " " *
        string(num_parts) * " " *
        string(imb) * " " *
        string(runs) * " " *
        string(ctype) * " " *
        string(rtype) * " " *
        string(vcycle) * " " *
        string(reconst) * " " *
        string(dbglvl)
    hmetis_command = `sh -c $hmetis_string`
    run(hmetis_command, wait=true)

    hint_file = hint_base * ".part." * string(num_parts)
    if !isfile(hint_file)
        error("failed to generate initial hMETIS hint: " * hint_file)
    end
    return hint_file
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

function lift_partition_to_original(hypergraph,
                                    processed_partition::Vector{Int},
                                    num_parts::Int)
    if length(processed_partition) == hypergraph.num_vertices
        return processed_partition
    end

    (processed_hypergraph,
     original_indices,
     new_indices,
     unused_indices) = SpecPart.isolate_islands(hypergraph)
    if length(processed_partition) != processed_hypergraph.num_vertices
        return processed_partition
    end

    (clusters, cluster_sizes) = SpecPart.island_removal(hypergraph, Int[])
    main_component = findmax(cluster_sizes)[2]
    partition = fill(-1, hypergraph.num_vertices)
    balance = zeros(Int, num_parts)

    for old_vertex in 1:hypergraph.num_vertices
        new_vertex = new_indices[old_vertex]
        if new_vertex == 0
            continue
        end
        part = processed_partition[new_vertex]
        partition[old_vertex] = part
        balance[part + 1] += hypergraph.vwts[old_vertex]
    end

    for component in 1:length(cluster_sizes)
        if component == main_component
            continue
        end

        vertices = Int[]
        fixed_weights = Dict{Int, Int}()
        for vertex in 1:hypergraph.num_vertices
            if clusters[vertex] != component
                continue
            end
            push!(vertices, vertex)
            fixed_part = hypergraph.fixed[vertex]
            if fixed_part >= 0
                fixed_weights[fixed_part] = get(fixed_weights, fixed_part, 0) + hypergraph.vwts[vertex]
            end
        end

        assigned_part = 0
        if length(fixed_weights) == 1
            assigned_part = first(keys(fixed_weights))
        elseif isempty(fixed_weights)
            assigned_part = argmin(balance) - 1
        else
            best_part = first(keys(fixed_weights))
            best_weight = fixed_weights[best_part]
            for (part, weight) in fixed_weights
                if weight > best_weight
                    best_part = part
                    best_weight = weight
                end
            end
            assigned_part = best_part
        end

        for vertex in vertices
            fixed_part = hypergraph.fixed[vertex]
            part = fixed_part >= 0 ? fixed_part : assigned_part
            partition[vertex] = part
            balance[part + 1] += hypergraph.vwts[vertex]
        end
    end

    for vertex in 1:hypergraph.num_vertices
        if partition[vertex] < 0
            fixed_part = hypergraph.fixed[vertex]
            partition[vertex] = fixed_part >= 0 ? fixed_part : 0
        end
    end

    return partition
end

function env_override_or(default_value::String, env_name::String)
    value = get(ENV, env_name, "")
    return isempty(value) ? default_value : value
end

function command_prefix_override_or(default_value::String, env_name::String)
    return rstrip(env_override_or(default_value, env_name)) * " "
end

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const JULIA_SOURCE_ROOT = "/home/norising/K_SpecPart"
const JULIA_RUNTIME_ROOT =
    env_override_or(joinpath(tempdir(), "kspecpart-julia-ref"),
                    "K_SPECPART_JULIA_RUNTIME_ROOT")

mkpath(JULIA_RUNTIME_ROOT)

include(joinpath(JULIA_SOURCE_ROOT, "specpart.jl"))
using .SpecPart

if !isdefined(SpecPart.Laplacians, :degree_matrix)
    @eval SpecPart.Laplacians begin
        function degree_matrix(adj::SparseArrays.SparseMatrixCSC)
            counts = zeros(Float64, size(adj, 1))
            for col in 1:length(adj.colptr)-1
                for idx in adj.colptr[col]:(adj.colptr[col + 1] - 1)
                    counts[adj.rowval[idx]] += 1.0
                end
            end
            return SparseArrays.spdiagm(counts)
        end
    end
end

SpecPart.metis_path = JULIA_SOURCE_ROOT
SpecPart.hmetis_path =
    command_prefix_override_or(joinpath(REPO_ROOT, "scripts", "hmetis_wrapper.sh"),
                               "K_SPECPART_HMETIS_PATH")
SpecPart.ilp_path =
    command_prefix_override_or(joinpath(REPO_ROOT, "scripts", "julia_ilp_wrapper.sh"),
                               "K_SPECPART_ILP_PATH")
SpecPart.triton_part_refiner_path =
    command_prefix_override_or("true", "K_SPECPART_TRITON_PATH")

const DEBUG_OVERLAY_DIR = let
    dir = get(ENV, "K_SPECPART_DEBUG_OVERLAY_DIR",
              get(ENV, "K_SPECPART_DEBUG_EXTERNAL_DIR", ""))
    if isempty(dir)
        nothing
    else
        mkpath(dir)
        dir
    end
end
const OVERLAY_DEBUG_COUNTER = Ref(0)
const OPTIMAL_DEBUG_COUNTER = Ref(0)
const METIS_DEBUG_COUNTER = Ref(0)
const MST_TRACE_COUNTER = Ref(0)
const TREE_GRAPH_DEBUG_COUNTER = Ref(0)
const DEBUG_LOBPCG_DIR = let
    dir = get(ENV, "K_SPECPART_DEBUG_LOBPCG_DIR", "")
    if isempty(dir)
        nothing
    else
        mkpath(dir)
        dir
    end
end
const LOBPCG_DEBUG_COUNTER = Ref(0)
const DEBUG_LOBPCG_STEP_DIR = let
    dir = get(ENV, "K_SPECPART_DEBUG_LOBPCG_STEP_DIR", "")
    if isempty(dir)
        nothing
    else
        mkpath(dir)
        dir
    end
end
const DEBUG_MAIN_RNG_DIR = let
    dir = get(ENV, "K_SPECPART_DEBUG_MAIN_RNG_DIR", "")
    if isempty(dir)
        nothing
    else
        mkpath(dir)
        dir
    end
end

function task_rng_state_tuple()
    t = current_task()
    ntuple(i -> UInt64(getfield(t, Symbol("rngState" * string(i - 1)))), 5)
end

function dump_main_rng_state(label::String)
    if isnothing(DEBUG_MAIN_RNG_DIR)
        return
    end
    open(joinpath(DEBUG_MAIN_RNG_DIR, label * ".txt"), "w") do io
        for word in task_rng_state_tuple()
            println(io, string(word, base=16))
        end
    end
end

function next_overlay_debug_prefix()
    OVERLAY_DEBUG_COUNTER[] += 1
    return joinpath(DEBUG_OVERLAY_DIR, "julia-overlay-round-" * string(OVERLAY_DEBUG_COUNTER[]))
end

function next_optimal_debug_prefix(method::String)
    OPTIMAL_DEBUG_COUNTER[] += 1
    return joinpath(DEBUG_OVERLAY_DIR,
                    "julia-optimal-" * method * "-" * string(OPTIMAL_DEBUG_COUNTER[]))
end

function next_metis_debug_prefix(tree_type::Int)
    METIS_DEBUG_COUNTER[] += 1
    return joinpath(DEBUG_OVERLAY_DIR,
                    "julia-metis-tree-" * string(METIS_DEBUG_COUNTER[]) * ".type-" * string(tree_type))
end

function next_mst_trace_prefix(tree_type::Int)
    MST_TRACE_COUNTER[] += 1
    return joinpath(DEBUG_OVERLAY_DIR,
                    "julia-mst-trace-" * string(MST_TRACE_COUNTER[]) * ".type-" * string(tree_type))
end

function next_tree_graph_debug_prefix(tree_type::Int, label::String)
    TREE_GRAPH_DEBUG_COUNTER[] += 1
    return joinpath(DEBUG_OVERLAY_DIR,
                    "julia-" * label * "-" * string(TREE_GRAPH_DEBUG_COUNTER[]) * ".type-" * string(tree_type))
end

function dump_weighted_graph_debug_artifact(path::String, matrix)
    upper = SparseArrays.triu(SparseMatrixCSC(matrix), 1)
    (ii, jj, vv) = findnz(upper)
    open(path, "w") do io
        println(io, size(matrix, 1), " ", length(vv))
        for idx in eachindex(vv)
            println(io,
                    Int(ii[idx] - 1),
                    " ",
                    Int(jj[idx] - 1),
                    " ",
                    @sprintf("%.17e", Float64(vv[idx])))
        end
    end
end

function dump_optimal_debug_artifacts(method::String,
                                      hgraph_file::String,
                                      partition::Vector{Int},
                                      num_parts::Int)
    if isnothing(DEBUG_OVERLAY_DIR)
        return
    end
    prefix = next_optimal_debug_prefix(method)
    cp(hgraph_file, prefix * ".hgr"; force=true)
    write_partition_file(prefix * ".part." * string(num_parts), partition)
    open(prefix * ".method", "w") do io
        println(io, method)
    end
end

function dump_matrix_debug_artifact(file_name::String, matrix)
    if isnothing(DEBUG_OVERLAY_DIR)
        return
    end
    open(joinpath(DEBUG_OVERLAY_DIR, file_name), "w") do io
        println(io, size(matrix, 1), " ", size(matrix, 2))
        for i in 1:size(matrix, 1)
            for j in 1:size(matrix, 2)
                if j > 1
                    print(io, " ")
                end
                print(io, @sprintf("%.17e", matrix[i, j]))
            end
            print(io, "\n")
        end
    end
end

function next_lobpcg_debug_prefix()
    LOBPCG_DEBUG_COUNTER[] += 1
    return joinpath(DEBUG_LOBPCG_DIR, "julia-lobpcg-" * string(LOBPCG_DEBUG_COUNTER[]))
end

function dump_lobpcg_matrix_artifact(label::String, matrix)
    if isnothing(DEBUG_LOBPCG_DIR)
        return
    end
    prefix = next_lobpcg_debug_prefix()
    open(prefix * "." * label * ".txt", "w") do io
        println(io, size(matrix, 1), " ", size(matrix, 2))
        for i in 1:size(matrix, 1)
            for j in 1:size(matrix, 2)
                if j > 1
                    print(io, " ")
                end
                print(io, @sprintf("%.17e", matrix[i, j]))
            end
            print(io, "\n")
        end
    end
end

function write_dense_debug_artifact(path::String, matrix)
    open(path, "w") do io
        println(io, size(matrix, 1), " ", size(matrix, 2))
        for i in 1:size(matrix, 1)
            for j in 1:size(matrix, 2)
                if j > 1
                    print(io, " ")
                end
                print(io, @sprintf("%.17e", matrix[i, j]))
            end
            print(io, "\n")
        end
    end
end

function dump_lobpcg_step_matrix_artifact(debug_label::String, label::String, matrix)
    if isnothing(DEBUG_LOBPCG_STEP_DIR) || isempty(debug_label)
        return
    end
    write_dense_debug_artifact(joinpath(DEBUG_LOBPCG_STEP_DIR, debug_label * "." * label * ".txt"),
                               Matrix(matrix))
end

function dump_lobpcg_step_upper_triangular_artifact(debug_label::String, label::String, matrix)
    if isnothing(DEBUG_LOBPCG_STEP_DIR) || isempty(debug_label)
        return
    end
    dense = Matrix(matrix)
    for j in axes(dense, 2)
        for i in (j + 1):size(dense, 1)
            dense[i, j] = 0.0
        end
    end
    write_dense_debug_artifact(joinpath(DEBUG_LOBPCG_STEP_DIR, debug_label * "." * label * ".txt"),
                               dense)
end

function dump_lobpcg_step_vector_artifact(debug_label::String, label::String, values)
    if isnothing(DEBUG_LOBPCG_STEP_DIR) || isempty(debug_label)
        return
    end
    path = joinpath(DEBUG_LOBPCG_STEP_DIR, debug_label * "." * label * ".txt")
    open(path, "w") do io
        println(io, length(values), " 1")
        for value in values
            print(io, @sprintf("%.17e", Float64(value)))
            print(io, "\n")
        end
    end
end

function dump_lobpcg_step_index_artifact(debug_label::String, label::String, values)
    if isnothing(DEBUG_LOBPCG_STEP_DIR) || isempty(debug_label)
        return
    end
    path = joinpath(DEBUG_LOBPCG_STEP_DIR, debug_label * "." * label * ".txt")
    open(path, "w") do io
        println(io, length(values), " 1")
        for value in values
            println(io, Int(value))
        end
    end
end

active_mask_as_ints(mask) = [flag ? 1 : 0 for flag in mask]

function cholqr_probe!(blocks,
                       ortho!,
                       A,
                       B,
                       generalized::Bool,
                       debug_label::String,
                       phase_label::String;
                       bs::Int = -1)
    useview = bs != -1
    size_x = useview ? bs : size(blocks.block, 2)
    X = blocks.block
    BX = blocks.B_block
    AX = blocks.A_block

    if useview
        SpecPart.IterativeSolvers.B_mul_X!(blocks, B, size_x)
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".X_pre", @view X[:, 1:size_x])
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".B_pre_ortho", @view BX[:, 1:size_x])
    else
        SpecPart.IterativeSolvers.B_mul_X!(blocks, B)
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".X_pre", X)
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".B_pre_ortho", BX)
    end

    gram_view = @view ortho!.gramVBV[1:size_x, 1:size_x]
    if useview
        mul!(gram_view, adjoint(@view(X[:, 1:size_x])), @view(BX[:, 1:size_x]))
    else
        mul!(gram_view, adjoint(X), BX)
    end
    SpecPart.IterativeSolvers.realdiag!(gram_view)
    dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".gramVBV_pre_chol", gram_view)

    cholf = cholesky!(Hermitian(gram_view))
    R = Matrix(cholf.factors[1:size_x, 1:size_x])
    dump_lobpcg_step_upper_triangular_artifact(debug_label, phase_label * ".cholR", R)
    if useview
        SpecPart.IterativeSolvers.rdiv!(@view(X[:, 1:size_x]), UpperTriangular(R))
        generalized && SpecPart.IterativeSolvers.rdiv!(@view(BX[:, 1:size_x]), UpperTriangular(R))
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".X_post_ortho", @view X[:, 1:size_x])
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".B_post_ortho", @view BX[:, 1:size_x])
        SpecPart.IterativeSolvers.A_mul_X!(blocks, A, size_x)
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".A_post_ortho", @view AX[:, 1:size_x])
    else
        SpecPart.IterativeSolvers.rdiv!(X, UpperTriangular(R))
        generalized && SpecPart.IterativeSolvers.rdiv!(BX, UpperTriangular(R))
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".X_post_ortho", X)
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".B_post_ortho", BX)
        SpecPart.IterativeSolvers.A_mul_X!(blocks, A)
        dump_lobpcg_step_matrix_artifact(debug_label, phase_label * ".A_post_ortho", AX)
    end
end

function run_lobpcg_step_probe(A,
                               B,
                               X0,
                               P,
                               tol::Float64,
                               largest::Bool,
                               debug_label::String)
    if isnothing(DEBUG_LOBPCG_STEP_DIR) || isempty(debug_label)
        return
    end

    iterator = SpecPart.IterativeSolvers.LOBPCGIterator(A, B, largest, copy(X0), P, nothing)
    iterator.constr!(iterator.XBlocks.block, iterator.tempXBlocks.block)
    size_x = size(iterator.XBlocks.block, 2)

    cholqr_probe!(iterator.XBlocks, iterator.ortho!, iterator.A, iterator.B, true, debug_label, "iter1.x")
    SpecPart.IterativeSolvers.block_grams_1x1!(iterator)
    gram1 = Matrix(@view iterator.gramABlock.XAX[1:size_x, 1:size_x])
    dump_lobpcg_step_matrix_artifact(debug_label, "iter1.rr_1x1.gramA", gram1)
    gram1_eig = copy(gram1)
    SpecPart.IterativeSolvers.realdiag!(gram1_eig)
    eig1 = eigen!(Hermitian(gram1_eig))
    dump_lobpcg_step_vector_artifact(debug_label, "iter1.rr_1x1.eigvals", eig1.values)
    dump_lobpcg_step_matrix_artifact(debug_label, "iter1.rr_1x1.eigvecs", eig1.vectors)
    partialsortperm!(@view(iterator.λperm[1:size_x]), eig1.values, 1:size_x; rev=iterator.largest)
    iterator.ritz_values[1:size_x] .= eig1.values[iterator.λperm[1:size_x]]
    iterator.V[1:size_x, 1:size_x] .= eig1.vectors[:, iterator.λperm[1:size_x]]
    dump_lobpcg_step_index_artifact(debug_label, "iter1.rr_1x1.perm", iterator.λperm[1:size_x])
    dump_lobpcg_step_vector_artifact(debug_label, "iter1.rr_1x1.ritz_values", iterator.ritz_values[1:size_x])
    dump_lobpcg_step_matrix_artifact(debug_label,
                                     "iter1.rr_1x1.selected_vectors",
                                     @view(iterator.V[1:size_x, 1:size_x]))
    SpecPart.IterativeSolvers.update_X_P!(iterator, 0, 0)
    SpecPart.IterativeSolvers.residuals!(iterator)
    SpecPart.IterativeSolvers.update_mask!(iterator, tol)
    dump_lobpcg_step_vector_artifact(debug_label, "iter1.residuals", iterator.residuals[1:size_x])
    dump_lobpcg_step_index_artifact(debug_label, "iter1.active_mask", active_mask_as_ints(iterator.activeMask))

    bs = iterator.currentBlockSize[]
    if bs <= 0
        return
    end

    SpecPart.IterativeSolvers.update_active!(iterator.activeMask,
                                             bs,
                                             (iterator.activeRBlocks.block, iterator.RBlocks.block))
    dump_lobpcg_step_matrix_artifact(debug_label, "iter2.r.precond_input", @view(iterator.activeRBlocks.block[:, 1:bs]))
    SpecPart.IterativeSolvers.precond_constr!(iterator.activeRBlocks.block,
                                              iterator.tempXBlocks.block,
                                              bs,
                                              iterator.precond!,
                                              iterator.constr!)
    dump_lobpcg_step_matrix_artifact(debug_label,
                                     "iter2.r.post_precond_constr",
                                     @view(iterator.activeRBlocks.block[:, 1:bs]))
    cholqr_probe!(iterator.activeRBlocks,
                  iterator.ortho!,
                  iterator.A,
                  iterator.B,
                  true,
                  debug_label,
                  "iter2.r";
                  bs=bs)
    SpecPart.IterativeSolvers.block_grams_2x2!(iterator, bs)
    subdim = size_x + bs
    dump_lobpcg_step_matrix_artifact(debug_label,
                                     "iter2.rr_2x2.gramA",
                                     @view(iterator.gramA[1:subdim, 1:subdim]))
    dump_lobpcg_step_matrix_artifact(debug_label,
                                     "iter2.rr_2x2.gramB",
                                     @view(iterator.gramB[1:subdim, 1:subdim]))
    gram2_a = copy(@view iterator.gramA[1:subdim, 1:subdim])
    gram2_b = copy(@view iterator.gramB[1:subdim, 1:subdim])
    SpecPart.IterativeSolvers.realdiag!(gram2_a)
    SpecPart.IterativeSolvers.realdiag!(gram2_b)
    eig2 = eigen!(Hermitian(gram2_a), Hermitian(gram2_b))
    dump_lobpcg_step_vector_artifact(debug_label, "iter2.rr_2x2.eigvals", eig2.values)
    dump_lobpcg_step_matrix_artifact(debug_label, "iter2.rr_2x2.eigvecs", eig2.vectors)
    partialsortperm!(@view(iterator.λperm[1:subdim]), eig2.values, 1:subdim; rev=iterator.largest)
    iterator.ritz_values[1:size_x] .= eig2.values[iterator.λperm[1:size_x]]
    iterator.V[1:subdim, 1:size_x] .= eig2.vectors[:, iterator.λperm[1:size_x]]
    dump_lobpcg_step_index_artifact(debug_label, "iter2.rr_2x2.perm", iterator.λperm[1:subdim])
    dump_lobpcg_step_vector_artifact(debug_label, "iter2.rr_2x2.ritz_values", iterator.ritz_values[1:size_x])
    dump_lobpcg_step_matrix_artifact(debug_label,
                                     "iter2.rr_2x2.selected_vectors",
                                     @view(iterator.V[1:subdim, 1:size_x]))
    SpecPart.IterativeSolvers.update_X_P!(iterator, bs, 0)
    SpecPart.IterativeSolvers.residuals!(iterator)
    SpecPart.IterativeSolvers.update_mask!(iterator, tol)
    dump_lobpcg_step_vector_artifact(debug_label, "iter2.residuals", iterator.residuals[1:size_x])
    dump_lobpcg_step_index_artifact(debug_label, "iter2.active_mask", active_mask_as_ints(iterator.activeMask))
end

function format_lobpcg_scalar(value)
    return @sprintf("%.17e", Float64(value))
end

function format_lobpcg_vector(values)
    if length(values) == 0
        return "[]"
    end
    formatted = String[]
    sizehint!(formatted, length(values))
    for value in values
        push!(formatted, format_lobpcg_scalar(value))
    end
    return "[" * join(formatted, ",") * "]"
end

function append_lobpcg_trace_lines!(lines::Vector{String}, trace, prefix::String = "")
    if isempty(trace)
        push!(lines, prefix * "empty")
        return
    end
    if hasproperty(first(trace), :iteration)
        for state in trace
            max_residual = isempty(state.residual_norms) ? 0.0 : maximum(state.residual_norms)
            push!(lines,
                  prefix *
                  "iter=$(state.iteration) " *
                  "max_residual=$(format_lobpcg_scalar(max_residual)) " *
                  "residual_norms=$(format_lobpcg_vector(state.residual_norms)) " *
                  "ritz_values=$(format_lobpcg_vector(state.ritz_values))")
        end
        return
    end
    for (batch_id, batch_trace) in enumerate(trace)
        push!(lines, prefix * "batch=$(batch_id)")
        append_lobpcg_trace_lines!(lines, batch_trace, prefix * "  ")
    end
end

function format_lobpcg_trace_exact(trace)
    lines = String["LOBPCG exact trace"]
    append_lobpcg_trace_lines!(lines, trace)
    return join(lines, "\n")
end

function solve_eigs_with_debug_x0(hgraph::SpecPart.__hypergraph__,
                                  adj::SparseMatrixCSC,
                                  pindex::SpecPart.__pindex__,
                                  largest::Bool,
                                  nev::Int,
                                  solver_iters::Int;
                                  epsilon::Int = 1,
                                  debug_label::String = "")
    d = ones(hgraph.num_vertices) ./ 1e06
    degs = SpecPart.Laplacians.degree_matrix(adj)
    lap_matrix = spdiagm(d) + degs - adj
    multiplier = ones(size(hgraph.vwts, 2))
    afunc = SpecPart.make_a_func(hgraph, epsilon)
    amap = SpecPart.LinearMaps.LinearMap(afunc, issymmetric=true, hgraph.num_vertices)
    bfunc = SpecPart.make_b_func(hgraph.vwts, pindex, multiplier)
    bmap = SpecPart.LinearMaps.LinearMap(bfunc, hgraph.num_vertices)

    if size(lap_matrix, 1) < 100
        results = SpecPart.IterativeSolvers.lobpcg(lap_matrix, largest, nev + 1, maxiter=solver_iters)
        return results.X[:, 2:nev + 1]
    end

    (pfunc, _) = SpecPart.CombinatorialMultigrid.cmg_preconditioner_lap(lap_matrix)
    X0 = rand(eltype(lap_matrix), size(lap_matrix, 1), nev)
    dump_lobpcg_matrix_artifact("x0", X0)
    run_lobpcg_step_probe(amap,
                          bmap,
                          X0,
                          SpecPart.CombinatorialMultigrid.lPreconditioner(pfunc),
                          1e-40,
                          false,
                          debug_label)
    results = SpecPart.IterativeSolvers.lobpcg(amap,
                                               bmap,
                                               false,
                                               X0;
                                               not_zeros=true,
                                               tol=1e-40,
                                               maxiter=solver_iters,
                                               P=SpecPart.CombinatorialMultigrid.lPreconditioner(pfunc),
                                               log=true)
    dump_lobpcg_matrix_artifact("xfinal", results.X)
    line_log = repeat("=", 60)
    @info "$line_log"
    @info "$results"
    @info "$(results.trace)"
    @info "$(format_lobpcg_trace_exact(results.trace))"
    @info "$line_log"
    return results.X
end

@eval SpecPart begin
    function debug_degrees_aware_prim_mst(g::Graphs.AbstractGraph{U},
                                          degree_threshold::Int,
                                          distmx::AbstractMatrix{T}=Graphs.weights(g),
                                          trace_file::Union{Nothing, String}=nothing) where {T <: Real, U}
        nvg = Graphs.nv(g)
        pq = PriorityQueue{U, T}()
        finished = zeros(Bool, nvg)
        wt = fill(typemax(T), nvg)
        parents = zeros(U, nvg)
        degrees = zeros(U, nvg)
        pq[1] = typemin(T)
        wt[1] = typemin(T)

        io = nothing
        if !isnothing(trace_file)
            io = open(trace_file, "w")
        end

        function write_trace(args...)
            if !isnothing(io)
                println(io, args...)
            end
        end

        iter = 0
        while !isempty(pq)
            v = dequeue!(pq)
            finished[v] = true
            parent = parents[v] == 0 ? -1 : Int(parents[v] - 1)
            write_trace("pop ", iter, " ", Int(v - 1), " ", Float64(wt[v]), " ", parent)
            connection_flag = false
            for u in Graphs.neighbors(g, v)
                if degrees[u] + 1 > degree_threshold && degrees[v] + 1 > degree_threshold
                    continue
                end
                finished[u] && continue

                if wt[u] > distmx[u, v]
                    old_wt = wt[u]
                    wt[u] = distmx[u, v]
                    pq[u] = wt[u]
                    parents[u] = v
                    connection_flag = true
                    write_trace("relax ", iter, " 1 ", Int(v - 1), " ", Int(u - 1), " ",
                                Float64(old_wt), " ", Float64(wt[u]))
                end
            end
            if connection_flag == false
                write_trace("retry ", iter, " ", Int(v - 1))
                for u in Graphs.neighbors(g, v)
                    finished[u] && continue

                    if wt[u] > distmx[u, v]
                        old_wt = wt[u]
                        wt[u] = distmx[u, v]
                        pq[u] = wt[u]
                        parents[u] = v
                        connection_flag = true
                        write_trace("relax ", iter, " 2 ", Int(v - 1), " ", Int(u - 1), " ",
                                    Float64(old_wt), " ", Float64(wt[u]))
                    end
                end
            end
            iter += 1
        end

        edges = [Graphs.Edge{U}(parents[v], v) for v in Graphs.vertices(g) if parents[v] != 0]
        for edge in edges
            write_trace("finish ", Int(edge.dst - 1), " ", Int(edge.src - 1))
        end
        if !isnothing(io)
            close(io)
        end
        return edges
    end

    function construct_tree(g::SimpleWeightedGraphs.SimpleWeightedGraph,
                            X::AbstractArray,
                            tree_type::Int)
        tree = SimpleWeightedGraphs.SimpleGraph(g)
        n = SimpleWeightedGraphs.nv(g)
        tree_matrix = spzeros(n, n)
        if tree_type == 1
            vtx_ids = Vector{Int}(1:n)
            fiedler = X[:, 1]
            sorted_vtx_ids = sortperm(fiedler)
            reverse_map = zeros(Int, n)
            for i in 1:n
                reverse_map[sorted_vtx_ids[i]] = i
            end
            (i, j, w) = findnz(g.weights)
            tree_reordered_mat = sparse(sorted_vtx_ids[i], sorted_vtx_ids[j], w)
            tree_reordered_graph = SimpleWeightedGraphs.SimpleWeightedGraph(tree_reordered_mat)
            lsst = akpw(tree_reordered_graph.weights)
            (i, j, w) = findnz(lsst)
            lsst = sparse(reverse_map[i], reverse_map[j], w)
            tree = SimpleWeightedGraphs.SimpleGraph(lsst)
            tree_matrix = lsst
        elseif tree_type == 2
            reweighted_prefix = nothing
            rawtree_prefix = nothing
            trace_file = nothing
            if !isnothing(Main.DEBUG_OVERLAY_DIR)
                reweighted_prefix = Main.next_tree_graph_debug_prefix(tree_type, "reweighted")
                rawtree_prefix = Main.next_tree_graph_debug_prefix(tree_type, "rawtree")
                trace_file = Main.next_mst_trace_prefix(tree_type) * ".txt"
                Main.dump_weighted_graph_debug_artifact(reweighted_prefix * ".txt", g.weights)
            end
            mst = debug_degrees_aware_prim_mst(g, 10, Graphs.weights(g), trace_file)
            i = zeros(Int, length(mst))
            j = zeros(Int, length(mst))
            w = zeros(length(mst))
            for k in 1:length(mst)
                vpair = mst[k]
                vsrc = vpair.src
                vdst = vpair.dst
                i[k] = vsrc
                j[k] = vdst
                w[k] = g.weights[vsrc, vdst]
            end
            tree = SimpleWeightedGraphs.SimpleGraph(mst)
            tree_matrix = sparse(i, j, w, n, n)
            tree_matrix += tree_matrix'
            if !isnothing(rawtree_prefix)
                Main.dump_weighted_graph_debug_artifact(rawtree_prefix * ".txt", tree_matrix)
            end
        elseif tree_type == 3
            vtxs = sortperm(X)
            i = vtxs[1:end-1]
            j = vtxs[2:end]
            w = abs.(X[j] - X[i])
            w_z = findall(w .== 0.0)
            w[w_z] .= 1e-6
            tree_matrix = sparse(i, j, w, n, n)
            tree_matrix += tree_matrix'
            tree = SimpleWeightedGraphs.SimpleGraph(tree_matrix)
        else
            @warn "Please select correct tree type!"
        end
        return tree, tree_matrix
    end

    function overlay(partitions::Vector, hgraph::__hypergraph__)
        (clusters, clusters_sizes) = hyperedges_removal(partitions, hgraph)
        hgraph_contracted = contract_hypergraph(hgraph, clusters)
        if !isnothing(Main.DEBUG_OVERLAY_DIR)
            prefix = Main.next_overlay_debug_prefix()
            write_hypergraph(hgraph_contracted, prefix * ".hgr")
            Main.write_partition_file(prefix * ".clusters", clusters)
            for i in 1:length(partitions)
                write_partition(partitions[i], prefix * ".input." * string(i - 1) * ".part")
            end
        end
        return hgraph_contracted, clusters
    end

    function METIS_tree_partition(T::SimpleWeightedGraphs.SimpleGraph,
                                  distilled_cuts::__cut_profile__,
                                  hgraph::__hypergraph__,
                                  seed::Int,
                                  metis_path::String,
                                  metis_opts::Int,
                                  num_parts::Int,
                                  ub_factor::Int)
        hwts = hgraph.hwts
        vtx_cuts = distilled_cuts.vtx_cuts
        edge_cuts = distilled_cuts.edge_cuts
        pred = distilled_cuts.pred
        forced_0 = distilled_cuts.forced_0
        forced_1 = distilled_cuts.forced_1
        forced_01 = distilled_cuts.forced_01
        FB0 = distilled_cuts.FB0
        FB1 = distilled_cuts.FB1
        edge_cuts_0 = distilled_cuts.edge_cuts_0
        edge_cuts_1 = distilled_cuts.edge_cuts_1
        nforced_0 = sum(hwts[forced_0])
        nforced_1 = sum(hwts[forced_1])
        nforced_01 = sum(hwts[forced_01])
        exc_0 = zeros(Int, hgraph.num_vertices)
        exc_1 = zeros(Int, hgraph.num_vertices)
        cut_cost = zeros(hgraph.num_vertices)
        T_matrix = sparse(T)

        for i in 1:hgraph.num_vertices
            if pred[i] == i
                cut_cost[i] = 1e09
                continue
            end
            exc_0[i] = edge_cuts[i] + nforced_0 - FB0[i] + edge_cuts_1[i] + nforced_01
            exc_1[i] = edge_cuts[i] + nforced_1 - FB1[i] + edge_cuts_0[i] + nforced_01
            cut_cost[i] = min(exc_0[i], exc_1[i])
            parent = pred[i]
            T_matrix[i, parent] = cut_cost[i]
            T_matrix[parent, i] = cut_cost[i]
        end

        g = SimpleWeightedGraphs.SimpleWeightedGraph(T_matrix)
        gname = build_metis_graph(g, metis_opts)
        if !isnothing(Main.DEBUG_OVERLAY_DIR)
            prefix = Main.next_metis_debug_prefix(metis_opts)
            cp(gname, prefix * ".gr"; force=true)
        end
        metis(metis_path, gname, num_parts, seed, ub_factor, metis_opts)
        pname = gname * ".part." * string(num_parts)
        pfile = open(pname, "r")
        partition = zeros(Int, hgraph.num_vertices)
        partition_i = 0
        for ln in eachline(pname)
            partition_i += 1
            partition[partition_i] = parse(Int, ln)
        end
        close(pfile)
        (cutsize, ~) = golden_evaluator(hgraph, num_parts, partition)
        @info "Cutsize from metis  $cutsize"
        return (partition, cutsize)
    end

    function optimal_partitioner(hmetis_path::String, cplex_path::String, hgraph::__hypergraph__, num_parts::Int, ub_factor::Int)
        partition = zeros(Int, hgraph.num_vertices)
        hgr_file_name = source_dir * "/" * "coarse.hgr"
        write_hypergraph(hgraph, hgr_file_name)

        function read_partition!(target::Vector{Int}, pfile::String)
            f = open(pfile, "r")
            itr = 0
            for ln in eachline(f)
                itr += 1
                target[itr] = parse(Int, ln)
            end
            close(f)
        end

        function run_hmetis!(target::Vector{Int}, local_hgr_name::String)
            runs = 10
            ctype = 1
            rtype = 1
            vcycle = 1
            reconst = 0
            dbglvl = 0
            hmetis_string = hmetis_path * " " * local_hgr_name * " " * string(num_parts) * " " * string(ub_factor) * " " * string(runs) * " " * string(ctype) * " " * string(rtype) * " " * string(vcycle) * " " * string(reconst) * " " * string(dbglvl)
            hmetis_command = `sh -c $hmetis_string`
            run(hmetis_command, wait=true)
            read_partition!(target, local_hgr_name * ".part." * string(num_parts))
        end

        if (hgraph.num_hyperedges < 1500 && num_parts == 2) || (hgraph.num_hyperedges < 300 && num_parts > 2)
            ilp_string = ilp_path * " " * hgr_file_name * " " * string(num_parts) * " " * string(ub_factor)
            ilp_command = `sh -c $ilp_string`
            run(ilp_command, wait=true)

            pfile = hgr_file_name * ".part." * string(num_parts)
            read_partition!(partition, pfile)

            golden = golden_evaluator(hgraph, num_parts, partition)
            cutsize = golden[1]
            balance = golden[2]
            max_balance = Int(ceil(sum(hgraph.vwts) * (((100 / num_parts) + ub_factor) / 100)))
            balanced = maximum(balance) <= max_balance
            println("[wrapper] optimal_partitioner ilp cutsize=", cutsize,
                    " balance=", balance,
                    " max_balance=", max_balance,
                    " balanced=", balanced)

            method = "ilp"
            if !balanced || cutsize == 0
                run_hmetis!(partition, hgr_file_name)
                method = "hmetis"
            end

            Main.dump_optimal_debug_artifacts(method, hgr_file_name, partition, num_parts)

            rm_cmd = `rm $hgr_file_name`
            run(rm_cmd, wait=true)
            rm_cmd = `rm $pfile`
            run(rm_cmd, wait=true)
            return partition
        end

        parallel_runs = 10
        partitions = [zeros(Int, length(partition)) for _ in 1:parallel_runs]
        cutsizes = zeros(Int, parallel_runs)
        @sync Threads.@threads for i in 1:parallel_runs
            local_hgr_name = hgr_file_name * "." * string(i)
            cmd = "cp " * hgr_file_name * " " * local_hgr_name
            run(`sh -c $cmd`, wait=true)
            run_hmetis!(partitions[i], local_hgr_name)
        end
        for i in 1:parallel_runs
            local_hgr_name = hgr_file_name * "." * string(i)
            local_pfile_name = local_hgr_name * ".part." * string(num_parts)
            golden = golden_evaluator(hgraph, num_parts, partitions[i])
            cutsizes[i] = golden[1]
            rm_cmd = `rm $local_hgr_name $local_pfile_name`
            run(rm_cmd, wait=true)
        end
        _, best_cut_idx = findmin(cutsizes)
        Main.dump_optimal_debug_artifacts("hmetis", hgr_file_name, partitions[best_cut_idx], num_parts)
        rm_cmd = `rm $hgr_file_name`
        run(rm_cmd, wait=true)
        return partitions[best_cut_idx]
    end

    function k_way_spectral_refine(hypergraph_file::String,
                                   partition::Vector{Int},
                                   hgraph::__hypergraph__,
                                   metis_path::String,
                                   ub_factor::Int;
                                   num_parts::Int = 2,
                                   eigen_vecs::Int = 1,
                                   cycles::Int = 1,
                                   refine_iters::Int = 2,
                                   solver_iters::Int = 20,
                                   best_solns::Int = 3,
                                   seed::Int = 0)
        line_log = repeat("=", 60)
        @info "$line_log"
        @info "**Solver parameters**"
        @info "$line_log"
        @info "Solver iterations $solver_iters"
        @info "Num vecs $eigen_vecs"
        @info "Tol 1e-40"
        specpart_refined_partition = partition
        processed_hg_name = hypergraph_file * ".processed"
        write_hypergraph(hgraph, processed_hg_name)
        adj_matrix = hypergraph2graph(hgraph, cycles)
        Main.dump_main_rng_state("jl-kway-after-graphification")
        global_partitions = []
        global_cutsizes = Int[]

        for iter in 1:refine_iters
            @info "[specpart] iteration $iter"
            Main.dump_main_rng_state("jl-kway-iter-" * string(iter) * "-start")
            partition_list = [Vector{Int}() for _ in 1:num_parts]
            for i in 1:length(partition)
                push!(partition_list[partition[i] + 1], i)
            end

            hgraph_vec = Vector{__hypergraph__}()
            adj_mat_vec = []
            pindices_vec = Vector{__pindex__}()
            for i in 1:num_parts
                side_0 = partition_list[i]
                side_1 = Int[]
                for j in 1:num_parts
                    if i == j
                        continue
                    end
                    append!(side_1, partition_list[j])
                end
                fixed_vertices = __pindex__(side_0, side_1)
                push!(pindices_vec, fixed_vertices)
                push!(hgraph_vec, hgraph)
                push!(adj_mat_vec, adj_matrix)
            end

            embedding_vec = Vector{Any}(undef, length(pindices_vec))
            @sync Threads.@threads for i in 1:length(pindices_vec)
                if isnothing(Main.DEBUG_LOBPCG_DIR) && isnothing(Main.DEBUG_LOBPCG_STEP_DIR)
                    embedding_vec[i] = solve_eigs(hgraph_vec[i],
                                                  adj_mat_vec[i],
                                                  pindices_vec[i],
                                                  false,
                                                  eigen_vecs,
                                                  solver_iters,
                                                  epsilon=num_parts - 1)
                else
                    embedding_vec[i] = Main.solve_eigs_with_debug_x0(hgraph_vec[i],
                                                                     adj_mat_vec[i],
                                                                     pindices_vec[i],
                                                                     false,
                                                                     eigen_vecs,
                                                                     solver_iters,
                                                                     epsilon=num_parts - 1,
                                                                     debug_label="iter-" * string(iter) * ".block-" * string(i))
                end
            end

            concatenated_evec_matrix = embedding_vec[1]
            for i in 2:length(embedding_vec)
                concatenated_evec_matrix = hcat(concatenated_evec_matrix, embedding_vec[i])
            end

            Main.dump_matrix_debug_artifact("julia-kway-concat-embedding.txt",
                                            concatenated_evec_matrix)
            Main.dump_matrix_debug_artifact("julia-kway-concat-embedding-iter-" * string(iter) * ".txt",
                                            concatenated_evec_matrix)
            reduced_evec_matrix = lda(concatenated_evec_matrix, partition)
            reduced_evec_matrix = Array(reduced_evec_matrix')
            Main.dump_matrix_debug_artifact("julia-kway-embedding-iter-" * string(iter) * ".txt",
                                            reduced_evec_matrix)
            max_capacity = Int(ceil(sum(hgraph.vwts) * ((100 / num_parts) + ub_factor) / 100))
            min_capacity = Int(ceil(sum(hgraph.vwts) * ((100 / num_parts) - ub_factor) / 100))
            fixed_vertices = __pindex__(Int[], Int[])
            partitions_vec = tree_partition(adj_mat_vec[1],
                                            reduced_evec_matrix,
                                            hgraph,
                                            fixed_vertices,
                                            ub_factor,
                                            [min_capacity, max_capacity],
                                            metis_path,
                                            num_parts,
                                            seed,
                                            true)

            partitions = []
            cutsizes = Int[]
            cut_dictionary = Dict{Int, Int}()

            @sync Threads.@threads for i in 1:length(partitions_vec)
                partition_file = "tree_partition" * string(i) * ".part." * string(num_parts)
                write_partition(partitions_vec[i][1], partition_file)
                triton_part_refine(triton_part_refiner_path,
                                   processed_hg_name,
                                   partition_file,
                                   num_parts,
                                   ub_factor,
                                   seed,
                                   i)
            end
            for i in 1:length(partitions_vec)
                partition_file = "tree_partition" * string(i) * ".part." * string(num_parts)
                partitions_vec[i][1] = read_hint_file(partition_file)
                push!(partitions, partitions_vec[i][1])
            end

            rm_cmd = "rm -r *.part." * string(num_parts)
            run(`sh -c $rm_cmd`, wait=true)
            for i in 1:length(partitions)
                (cutsize, balance) = golden_evaluator(hgraph, num_parts, partitions[i])
                push!(cutsizes, cutsize)
                if haskey(cut_dictionary, cutsize) == false
                    push!(cut_dictionary, cutsize => i)
                end
                @info "[specpart] Refined partition $i with cutsize $cutsize $balance"
            end

            unique_partitions = []
            unique_cutsizes = Int[]
            unique_keys = collect(keys(cut_dictionary))
            for i in 1:length(unique_keys)
                key = unique_keys[i]
                partition_id = cut_dictionary[key]
                push!(unique_cutsizes, key)
                push!(unique_partitions, partitions[partition_id])
            end

            sorted_partition_ids = sortperm(unique_cutsizes)
            best_partitions = []
            solns_to_pick = min(best_solns, length(unique_cutsizes))
            for i in 1:solns_to_pick
                push!(best_partitions, unique_partitions[sorted_partition_ids[i]])
                @info "[specpart] partition picked with cutsize $(unique_cutsizes[sorted_partition_ids[i]])"
            end
            push!(best_partitions, partition)
            Main.dump_main_rng_state("jl-kway-iter-" * string(iter) * "-pre-overlay")
            hgraph_contracted, clusters = overlay(best_partitions, hgraph)
            @info "Running optimal attempt partitioning**"
            refined_partition = optimal_partitioner(hmetis_path, ilp_path, hgraph_contracted, num_parts, ub_factor)
            cutsize = golden_evaluator(hgraph_contracted, num_parts, refined_partition)
            @info "specpart cutsize recorded: $cutsize"

            partition_projected = zeros(Int, hgraph.num_vertices)
            for i in 1:length(clusters)
                cid = clusters[i]
                partition_projected[i] = refined_partition[cid]
            end
            specpart_partition_name = processed_hg_name * ".specpart" * ".part." * string(num_parts)
            write_partition(partition_projected, specpart_partition_name)
            triton_part_refine(triton_part_refiner_path,
                               processed_hg_name,
                               specpart_partition_name,
                               num_parts,
                               ub_factor,
                               seed,
                               0)
            partition_projected = read_hint_file(specpart_partition_name)
            cutsize = golden_evaluator(hgraph, num_parts, partition_projected)
            @info "specpart cutsize recorded: $cutsize"
            specpart_refined_partition = partition_projected
            Main.dump_main_rng_state("jl-kway-iter-" * string(iter) * "-post-overlay")
            push!(global_partitions, partition_projected)
            push!(global_cutsizes, cutsize[1])

            if get(ENV, "K_SPECPART_STOP_AFTER_FIRST_OVERLAY", "") == "1"
                cmd = "rm " * processed_hg_name
                run(`sh -c $cmd`, wait=true)
                return partition_projected, cutsize[1]
            end
        end

        @info "[specpart] running final round of overlay"
        push!(global_partitions, partition)
        hgraph_contracted, clusters = overlay(global_partitions, hgraph)
        @info "Running optimal attempt partitioning**"
        specpart_partition = optimal_partitioner(hmetis_path, ilp_path, hgraph_contracted, num_parts, ub_factor)
        cutsize = golden_evaluator(hgraph_contracted, num_parts, specpart_partition)
        @info "[specpart] refined cutsize recorded: $cutsize"
        partition_projected = zeros(Int, hgraph.num_vertices)
        for i in 1:length(clusters)
            cid = clusters[i]
            partition_projected[i] = specpart_partition[cid]
        end
        pre_refined_part = partition_projected
        pre_refined_cut, ~ = golden_evaluator(hgraph, num_parts, pre_refined_part)
        specpart_partition_name = processed_hg_name * ".specpart" * ".part.2"
        write_partition(partition_projected, specpart_partition_name)
        triton_part_refine(triton_part_refiner_path,
                           processed_hg_name,
                           specpart_partition_name,
                           num_parts,
                           ub_factor,
                           seed,
                           0)
        partition_projected = read_hint_file(specpart_partition_name)
        cutsize = golden_evaluator(hgraph, num_parts, partition_projected)
        global_min_cut, idx = findmin(global_cutsizes)
        final_cut = 0
        final_part = Int[]
        if cutsize[1] < global_min_cut
            final_cut = cutsize[1]
            final_part = partition_projected
        else
            final_cut = global_min_cut
            final_part = global_partitions[idx]
        end
        cmd = "rm " * processed_hg_name
        run(`sh -c $cmd`, wait=true)
        post_refined_cut = final_cut
        post_refined_part = final_part
        if pre_refined_cut < post_refined_cut
            @info "specpart cutsize recorded: $pre_refined_cut"
            return pre_refined_part, pre_refined_cut
        else
            @info "specpart cutsize recorded: $post_refined_cut"
            return post_refined_part, post_refined_cut
        end
    end
end

options = parse_args(ARGS)
run_dir = mktempdir(JULIA_RUNTIME_ROOT)
work_dir = joinpath(run_dir, "work")
mkpath(work_dir)
SpecPart.source_dir = work_dir

staged_hypergraph = stage_input_file(options.hypergraph_file, run_dir, "hypergraph")
staged_fixed = stage_input_file(options.hypergraph_fixed_file, run_dir, "fixed")
staged_hint = stage_input_file(options.hint_file, run_dir, "hint")
if isempty(staged_hint)
    staged_hint = generate_initial_hint_file(staged_hypergraph, options.num_parts, options.imb, run_dir)
end

partition, cutsize = cd(run_dir) do
    SpecPart.specpart_run(
        staged_hypergraph;
        hypergraph_fixed_file = staged_fixed,
        hint_file = staged_hint,
        imb = options.imb,
        num_parts = options.num_parts,
        eigvecs = options.eigvecs,
        refine_iters = options.refine_iters,
        solver_iters = options.solver_iters,
        best_solns = options.best_solns,
        ncycles = options.ncycles,
        seed = options.seed,
    )
end

original_hypergraph = SpecPart.read_hypergraph_file(staged_hypergraph, staged_fixed)
partition = lift_partition_to_original(original_hypergraph, partition, options.num_parts)
if length(partition) == original_hypergraph.num_vertices
    cutsize, _ = SpecPart.golden_evaluator(original_hypergraph, options.num_parts, partition)
end

if !isempty(options.output_file)
    write_partition_file(options.output_file, partition)
    println("OUTPUT_FILE=" * options.output_file)
end

println("FINAL_CUT=" * string(cutsize))
println("PART_SIZE=" * string(length(partition)))
