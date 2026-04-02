#pragma once

#include "kspecpart/definitions.hpp"

#include <string>
#include <vector>

namespace kspecpart {

Hypergraph read_hypergraph_file(const std::string& file_name, const std::string& fixed_file_name = "");
std::vector<int> read_partition_file(const std::string& file_name);
void write_hypergraph_file(const std::string& file_name, const Hypergraph& hypergraph);
void write_partition_file(const std::string& file_name, const std::vector<int>& partition);

}  // namespace kspecpart
