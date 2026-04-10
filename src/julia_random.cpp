#include "kspecpart/julia_random.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include <openssl/sha.h>

namespace kspecpart {

namespace {

constexpr std::uint64_t kTaskForkMultiplier = 0xd1342543de82ef95ULL;
constexpr int kJuliaXoshiroWidth = 8;
constexpr int kJuliaXoshiroBulkThresholdBytes = 64;
constexpr std::uint64_t kJuliaXoshiroForkMul0 = 0x02011ce34bce797fULL;
constexpr std::uint64_t kJuliaXoshiroForkMul1 = 0x5a94851fb48a6e05ULL;
constexpr std::uint64_t kJuliaXoshiroForkMul2 = 0x3688cf5d48899fa7ULL;
constexpr std::uint64_t kJuliaXoshiroForkMul3 = 0x867b4bb4c42e5661ULL;
constexpr std::array<std::uint64_t, 4> kTaskForkXorConstants = {
    0x214c146c88e47cb7ULL,
    0xa66d8cc21285aafaULL,
    0x68c7ef2d7b1a54d4ULL,
    0xb053a7d7aa238c61ULL,
};
constexpr std::array<std::uint64_t, 4> kTaskForkMultipliers = {
    0xaef17502108ef2d9ULL,
    0xf34026eeb86766afULL,
    0x38fd70ad58dd9fbbULL,
    0x6677f9b93ab0c04dULL,
};

std::uint64_t load_le64(const unsigned char* bytes) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

std::uint64_t rotl64(std::uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

double julia_bits_to_float64(std::uint64_t value) {
    return static_cast<double>(value >> 11U) * 0x1.0p-53;
}

std::array<unsigned char, SHA256_DIGEST_LENGTH> hash_integer_seed(std::int64_t seed) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    const bool negative = seed < 0;
    std::uint64_t remaining = negative
        ? static_cast<std::uint64_t>(~seed)
        : static_cast<std::uint64_t>(seed);

    while (true) {
        const std::uint32_t word = static_cast<std::uint32_t>(remaining & 0xffffffffULL);
        unsigned char word_bytes[4];
        word_bytes[0] = static_cast<unsigned char>(word & 0xffU);
        word_bytes[1] = static_cast<unsigned char>((word >> 8U) & 0xffU);
        word_bytes[2] = static_cast<unsigned char>((word >> 16U) & 0xffU);
        word_bytes[3] = static_cast<unsigned char>((word >> 24U) & 0xffU);
        SHA256_Update(&ctx, word_bytes, sizeof(word_bytes));
        remaining >>= 32U;
        if (remaining == 0) {
            break;
        }
    }

    if (negative) {
        const unsigned char marker = 0x01U;
        SHA256_Update(&ctx, &marker, 1);
    }

    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256_Final(digest.data(), &ctx);
    return digest;
}

}  // namespace

JuliaXoshiro256PlusPlus::JuliaXoshiro256PlusPlus(std::int64_t seed) {
    seed_rng(seed);
}

JuliaXoshiro256PlusPlus::JuliaXoshiro256PlusPlus(const State& state) {
    set_state(state);
}

void JuliaXoshiro256PlusPlus::seed_rng(std::int64_t seed) {
    const auto digest = hash_integer_seed(seed);
    s0_ = load_le64(digest.data());
    s1_ = load_le64(digest.data() + 8);
    s2_ = load_le64(digest.data() + 16);
    s3_ = load_le64(digest.data() + 24);
    s4_ = s0_ + 3ULL * s1_ + 5ULL * s2_ + 7ULL * s3_;
}

std::uint64_t JuliaXoshiro256PlusPlus::next_u64() {
    const std::uint64_t tmp = s0_ + s3_;
    const std::uint64_t result = rotl64(tmp, 23) + s0_;
    const std::uint64_t t = s1_ << 17;

    s2_ ^= s0_;
    s3_ ^= s1_;
    s1_ ^= s2_;
    s0_ ^= s3_;
    s2_ ^= t;
    s3_ = rotl64(s3_, 45);
    return result;
}

std::uint64_t JuliaXoshiro256PlusPlus::next_uint52_raw() {
    return next_u64() >> 12U;
}

double JuliaXoshiro256PlusPlus::next_float64() {
    return static_cast<double>(next_u64() >> 11U) * 0x1.0p-53;
}

void JuliaXoshiro256PlusPlus::fill_float64_bulk(double* data, int count) {
    if (data == nullptr || count <= 0) {
        return;
    }

    int offset = 0;
    if (count * static_cast<int>(sizeof(double)) >= kJuliaXoshiroBulkThresholdBytes) {
        std::array<std::uint64_t, kJuliaXoshiroWidth> s0{};
        std::array<std::uint64_t, kJuliaXoshiroWidth> s1{};
        std::array<std::uint64_t, kJuliaXoshiroWidth> s2{};
        std::array<std::uint64_t, kJuliaXoshiroWidth> s3{};

        for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
            s0[static_cast<std::size_t>(lane)] = kJuliaXoshiroForkMul0 * next_u64();
        }
        for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
            s1[static_cast<std::size_t>(lane)] = kJuliaXoshiroForkMul1 * next_u64();
        }
        for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
            s2[static_cast<std::size_t>(lane)] = kJuliaXoshiroForkMul2 * next_u64();
        }
        for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
            s3[static_cast<std::size_t>(lane)] = kJuliaXoshiroForkMul3 * next_u64();
        }

        while (offset + kJuliaXoshiroWidth <= count) {
            for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
                const std::size_t idx = static_cast<std::size_t>(lane);
                const std::uint64_t result = rotl64(s0[idx] + s3[idx], 23) + s0[idx];
                const std::uint64_t t = s1[idx] << 17U;
                s2[idx] ^= s0[idx];
                s3[idx] ^= s1[idx];
                s1[idx] ^= s2[idx];
                s0[idx] ^= s3[idx];
                s2[idx] ^= t;
                s3[idx] = rotl64(s3[idx], 45);
                data[offset + lane] = julia_bits_to_float64(result);
            }
            offset += kJuliaXoshiroWidth;
        }
    }

    while (offset < count) {
        data[offset] = next_float64();
        ++offset;
    }
}

int JuliaXoshiro256PlusPlus::rand_less_than_masked_52(int sup_inclusive, std::uint64_t mask) {
    if (sup_inclusive < 0) {
        throw std::invalid_argument("sup_inclusive must be non-negative");
    }
    const std::uint64_t upper = static_cast<std::uint64_t>(sup_inclusive);
    while (true) {
        const std::uint64_t value = next_uint52_raw() & mask;
        if (value <= upper) {
            return static_cast<int>(value);
        }
    }
}

std::vector<int> JuliaXoshiro256PlusPlus::randperm_zero_based(int n) {
    if (n < 0) {
        throw std::invalid_argument("randperm size must be non-negative");
    }
    std::vector<int> permutation(static_cast<std::size_t>(n), 0);
    if (n == 0) {
        return permutation;
    }

    permutation[0] = 0;
    std::uint64_t mask = 3;
    for (int i = 2; i <= n; ++i) {
        const int index = rand_less_than_masked_52(i - 1, mask);
        if (i - 1 != index) {
            permutation[static_cast<std::size_t>(i - 1)] = permutation[static_cast<std::size_t>(index)];
        }
        permutation[static_cast<std::size_t>(index)] = i - 1;
        if (static_cast<std::uint64_t>(i) == mask + 1ULL) {
            mask = 2ULL * mask + 1ULL;
        }
    }
    return permutation;
}

JuliaXoshiro256PlusPlus::State JuliaXoshiro256PlusPlus::state() const {
    return {s0_, s1_, s2_, s3_, s4_};
}

void JuliaXoshiro256PlusPlus::set_state(const State& state) {
    s0_ = state[0];
    s1_ = state[1];
    s2_ = state[2];
    s3_ = state[3];
    s4_ = state[4];
}

JuliaXoshiro256PlusPlus JuliaXoshiro256PlusPlus::fork_task_local() {
    State child{};
    const std::uint64_t splitmix = s4_;
    s4_ = splitmix * kTaskForkMultiplier + 1ULL;
    child[4] = s4_;

    const std::array<std::uint64_t, 4> parent_state = {s0_, s1_, s2_, s3_};
    for (int index = 0; index < 4; ++index) {
        std::uint64_t c = parent_state[static_cast<std::size_t>(index)];
        const std::uint64_t w =
            splitmix ^ kTaskForkXorConstants[static_cast<std::size_t>(index)];
        c += w * (2ULL * c + 1ULL);
        c ^= c >> ((c >> 59U) + 5U);
        c *= kTaskForkMultipliers[static_cast<std::size_t>(index)];
        c ^= c >> 43U;
        child[static_cast<std::size_t>(index)] = c;
    }

    return JuliaXoshiro256PlusPlus(child);
}

}  // namespace kspecpart
