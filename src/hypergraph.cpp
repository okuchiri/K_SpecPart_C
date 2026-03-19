#include "kspecpart/hypergraph.hpp"

#include <vector>

namespace kspecpart {

Hypergraph build_hypergraph(int num_vertices,
                            int num_hyperedges,
                            const std::vector<int>& eptr,
                            const std::vector<int>& eind,
                            const std::vector<int>& fixed,
                            const std::vector<int>& vwts,
                            const std::vector<int>& hwts) {
    std::vector<std::vector<int>> incident(num_vertices);
    for (int edge = 0; edge < num_hyperedges; ++edge) {
        for (int idx = eptr[edge]; idx < eptr[edge + 1]; ++idx) {
            incident[eind[idx]].push_back(edge);
        }
    }

    std::vector<int> vptr(num_vertices + 1, 0);
    std::vector<int> vind;
    vind.reserve(eind.size());
    for (int vertex = 0; vertex < num_vertices; ++vertex) {
        vptr[vertex] = static_cast<int>(vind.size());
        vind.insert(vind.end(), incident[vertex].begin(), incident[vertex].end());
    }
    vptr[num_vertices] = static_cast<int>(vind.size());

    Hypergraph hypergraph;
    hypergraph.num_vertices = num_vertices;
    hypergraph.num_hyperedges = num_hyperedges;
    hypergraph.eptr = eptr;
    hypergraph.eind = eind;
    hypergraph.vptr = vptr;
    hypergraph.vind = vind;
    hypergraph.fixed = fixed;
    hypergraph.vwts = vwts;
    hypergraph.hwts = hwts;
    return hypergraph;
}

Hypergraph remove_single_hyperedges(const Hypergraph& hypergraph) {
    std::vector<int> eptr = {0};
    std::vector<int> eind;
    std::vector<int> hwts;
    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        const int start = hypergraph.eptr[edge];
        const int end = hypergraph.eptr[edge + 1];
        if (end - start <= 1) {
            continue;
        }
        eind.insert(eind.end(), hypergraph.eind.begin() + start, hypergraph.eind.begin() + end);
        eptr.push_back(static_cast<int>(eind.size()));
        hwts.push_back(hypergraph.hwts[edge]);
    }

    return build_hypergraph(hypergraph.num_vertices,
                            static_cast<int>(hwts.size()),
                            eptr,
                            eind,
                            hypergraph.fixed,
                            hypergraph.vwts,
                            hwts);
}

}  // namespace kspecpart
