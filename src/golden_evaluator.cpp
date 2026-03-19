#include "kspecpart/golden_evaluator.hpp"

#include <vector>

namespace kspecpart {

PartitionResult evaluate_partition(const Hypergraph& hypergraph, int num_parts, const std::vector<int>& partition) {
    PartitionResult result;
    result.partition = partition;
    result.balance.assign(num_parts, 0);

    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        const int start = hypergraph.eptr[edge];
        const int end = hypergraph.eptr[edge + 1];
        if (start >= end) {
            continue;
        }

        const int base_part = partition[hypergraph.eind[start]];
        bool cut = false;
        for (int idx = start + 1; idx < end; ++idx) {
            if (partition[hypergraph.eind[idx]] != base_part) {
                cut = true;
                break;
            }
        }
        if (cut) {
            result.cutsize += hypergraph.hwts[edge];
        }
    }

    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        const int part = partition[vertex];
        if (part >= 0 && part < num_parts) {
            result.balance[part] += hypergraph.vwts[vertex];
        }
    }
    return result;
}

}  // namespace kspecpart
