#pragma once

#include <string>
#include <utility>
#include <vector>

namespace kspecpart {

enum class ProjectionStrategy {
    kLda,
    kRandomSigned,
    kAlternatingColumns,
    kLeadingColumns,
};

struct Hypergraph {
    int num_vertices = 0;
    int num_hyperedges = 0;
    std::vector<int> eptr;
    std::vector<int> eind;
    std::vector<int> vptr;
    std::vector<int> vind;
    std::vector<int> fixed;
    std::vector<int> vwts;
    std::vector<int> hwts;
};

struct WeightedGraph {
    int num_vertices = 0;
    std::vector<std::vector<std::pair<int, double>>> adjacency;
    std::vector<double> degrees;
};

struct SpecPartOptions {
    std::string hypergraph_file;
    std::string fixed_file;
    std::string hint_file;
    std::string output_file = "partition.part";
    std::string gpmetis_executable;
    std::string hmetis_executable;
    std::string ilp_partitioner_executable;
    std::string triton_refiner_executable;
    int imb = 2;
    int num_parts = 2;
    int eigvecs = 2;
    int refine_iters = 2;
    int solver_iters = 40;
    int best_solns = 3;
    int ncycles = 1;
    int seed = 0;
    bool log_lobpcg = false;
    ProjectionStrategy projection_strategy = ProjectionStrategy::kLda;
    bool enable_optimal_partitioner = true;
    bool enable_hmetis_partitioner = true;
    bool enable_ilp_partitioner = true;
    bool enable_triton_refiner = true;
    bool enable_metis = true;
    bool hmetis_explicit = false;
    bool ilp_partitioner_explicit = false;
    bool triton_refiner_explicit = false;
    bool gpmetis_explicit = false;
};

struct PartitionResult {
    std::vector<int> partition;
    int cutsize = 0;
    std::vector<int> balance;
};

}  // namespace kspecpart
