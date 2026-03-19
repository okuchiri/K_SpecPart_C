#pragma once

#include "kspecpart/definitions.hpp"

#include <string>

namespace kspecpart {

bool parse_arguments(int argc, char** argv, SpecPartOptions& options, std::string& error);
PartitionResult specpart_run(const SpecPartOptions& options);

}  // namespace kspecpart
