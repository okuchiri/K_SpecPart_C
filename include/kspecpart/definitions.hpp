#pragma once

#include <string>
#include <utility>
#include <vector>

namespace kspecpart {

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
    int imb = 2;
    int num_parts = 2;
    int eigvecs = 2;
    int refine_iters = 2;
    int solver_iters = 40;
    int best_solns = 3;
    int ncycles = 1;
    int seed = 0;
};

struct PartitionResult {
    std::vector<int> partition;
    int cutsize = 0;
    std::vector<int> balance;
};

}  // namespace kspecpart
