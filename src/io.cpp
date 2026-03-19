#include "kspecpart/io.hpp"

#include "kspecpart/hypergraph.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kspecpart {

namespace {

std::vector<int> parse_ints(const std::string& line) {
    std::istringstream iss(line);
    std::vector<int> values;
    int value = 0;
    while (iss >> value) {
        values.push_back(value);
    }
    return values;
}

}  // namespace

Hypergraph read_hypergraph_file(const std::string& file_name, const std::string& fixed_file_name) {
    std::ifstream input(file_name);
    if (!input) {
        throw std::runtime_error("failed to open hypergraph file: " + file_name);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty hypergraph file: " + file_name);
    }
    const std::vector<int> header = parse_ints(line);
    if (header.size() < 2) {
        throw std::runtime_error("invalid hypergraph header in " + file_name);
    }

    const int num_hyperedges = header[0];
    const int num_vertices = header[1];
    const int wt_type = header.size() > 2 ? header[2] : 0;

    std::vector<int> eptr = {0};
    std::vector<int> eind;
    std::vector<int> hwts(num_hyperedges, 1);
    std::vector<int> vwts(num_vertices, 1);
    std::vector<int> fixed(num_vertices, -1);
    eind.reserve(num_hyperedges * 4);

    for (int edge = 0; edge < num_hyperedges; ++edge) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("unexpected EOF while reading hyperedges from " + file_name);
        }
        const std::vector<int> values = parse_ints(line);
        if (values.empty()) {
            eptr.push_back(static_cast<int>(eind.size()));
            continue;
        }

        int start_index = 0;
        if (wt_type != 0 && wt_type != 10) {
            hwts[edge] = values[0];
            start_index = 1;
        }
        for (int idx = start_index; idx < static_cast<int>(values.size()); ++idx) {
            eind.push_back(values[idx] - 1);
        }
        eptr.push_back(static_cast<int>(eind.size()));
    }

    int vertex = 0;
    while (vertex < num_vertices && std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        vwts[vertex++] = std::stoi(line);
    }

    if (!fixed_file_name.empty()) {
        std::ifstream fixed_input(fixed_file_name);
        if (!fixed_input) {
            throw std::runtime_error("failed to open fixed file: " + fixed_file_name);
        }
        vertex = 0;
        while (vertex < num_vertices && std::getline(fixed_input, line)) {
            if (line.empty()) {
                continue;
            }
            fixed[vertex++] = std::stoi(line);
        }
    }

    return build_hypergraph(num_vertices, num_hyperedges, eptr, eind, fixed, vwts, hwts);
}

std::vector<int> read_partition_file(const std::string& file_name) {
    std::ifstream input(file_name);
    if (!input) {
        throw std::runtime_error("failed to open partition file: " + file_name);
    }

    std::vector<int> partition;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        partition.push_back(std::stoi(line));
    }
    return partition;
}

void write_partition_file(const std::string& file_name, const std::vector<int>& partition) {
    std::ofstream output(file_name);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + file_name);
    }
    for (int part : partition) {
        output << part << '\n';
    }
}

}  // namespace kspecpart
