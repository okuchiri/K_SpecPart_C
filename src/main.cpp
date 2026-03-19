#include "kspecpart/kspecpart.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " --hypergraph <file> [options]\n"
        << "Options:\n"
        << "  --fixed-file <file>\n"
        << "  --hint-file <file>\n"
        << "  --output <file>           Default: partition.part\n"
        << "  --imb <int>               Default: 2\n"
        << "  --num-parts <int>         Default: 2\n"
        << "  --eigvecs <int>           Default: 2\n"
        << "  --refine-iters <int>      Default: 2\n"
        << "  --solver-iters <int>      Default: 40\n"
        << "  --best-solns <int>        Default: 3\n"
        << "  --ncycles <int>           Default: 1\n"
        << "  --seed <int>              Default: 0\n";
}

}  // namespace

int main(int argc, char** argv) {
    kspecpart::SpecPartOptions options;
    std::string error;
    if (!kspecpart::parse_arguments(argc, argv, options, error)) {
        if (!error.empty()) {
            std::cerr << error << '\n';
        }
        print_usage(argv[0]);
        return error.empty() ? 0 : 1;
    }

    try {
        const kspecpart::PartitionResult result = kspecpart::specpart_run(options);
        std::cout << "Output written to " << options.output_file << '\n';
        std::cout << "Final cutsize " << result.cutsize << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
