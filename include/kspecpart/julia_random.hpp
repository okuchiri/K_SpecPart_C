#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace kspecpart {

class JuliaXoshiro256PlusPlus {
public:
    using State = std::array<std::uint64_t, 5>;

    explicit JuliaXoshiro256PlusPlus(std::int64_t seed = 0);
    explicit JuliaXoshiro256PlusPlus(const State& state);

    void seed_rng(std::int64_t seed);

    std::uint64_t next_u64();
    std::uint64_t next_uint52_raw();
    double next_float64();
    void fill_float64_bulk(double* data, int count);
    int rand_less_than_masked_52(int sup_inclusive, std::uint64_t mask);
    std::vector<int> randperm_zero_based(int n);
    State state() const;
    void set_state(const State& state);
    JuliaXoshiro256PlusPlus fork_task_local();

private:
    std::uint64_t s0_ = 0;
    std::uint64_t s1_ = 0;
    std::uint64_t s2_ = 0;
    std::uint64_t s3_ = 0;
    std::uint64_t s4_ = 0;
};

using AlgorithmRng = JuliaXoshiro256PlusPlus;

}  // namespace kspecpart
