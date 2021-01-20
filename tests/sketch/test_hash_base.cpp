#include "sketch/hash_base.hpp"

#include <gtest/gtest.h>

#include <random>
#include <unordered_map>
#include <unordered_set>

namespace {

using namespace ts;
using namespace ::testing;

constexpr uint8_t SKETCH_DIM = 3;
constexpr uint8_t SET_SIZE = 4 * 4;
constexpr uint32_t MAX_LEN = 3;
constexpr uint8_t HASH_SIZE = MAX_LEN * SET_SIZE;

class Hash : public HashBase<uint32_t>, public testing::TestWithParam<HashAlgorithm> {
  public:
    Hash()
            : HashBase<uint32_t>(SET_SIZE,
                                 SKETCH_DIM,
                                 HASH_SIZE,
                                 HashAlgorithm::uniform,
            /*seed=*/31415) {}
};

// test that the uniform hash function is bijective, i.e. it is in effect a permutation:
TEST_F(Hash, HashesDistinct) {
    for (uint32_t s = 0; s < SKETCH_DIM; ++s) {
        std::unordered_set<uint8_t> seen(SKETCH_DIM);
        for (uint32_t i = 0; i < hash_size; ++i) {
            uint8_t v = this->hash(s, i);
            ASSERT_FALSE(seen.find(v) != seen.end());
            seen.insert(v);
        }
        ASSERT_EQ(hash_size, seen.size());
    }
}

class Hash2 : public HashBase<uint32_t>, public testing::TestWithParam<HashAlgorithm> {
  public:
    Hash2()
        : HashBase<uint32_t>(SET_SIZE,
                             SKETCH_DIM,
                             HASH_SIZE,
                             GetParam(),
                             /*seed=*/31415) {}
};

// test that the hash values are consistent - i.e. asking for the same value returns the same result
TEST_P(Hash2, HashesConsistent) {
    std::vector<std::unordered_map<uint8_t, uint8_t>> hashes(SKETCH_DIM);
    for (uint32_t s = 0; s < SKETCH_DIM; ++s) {
        for (uint32_t i = 0; i < hash_size; ++i) {
            uint8_t v = this->hash(s, i);
            ASSERT_FALSE(hashes[s].find(i) != hashes[s].end());
            hashes[s][i] = v;
        }
        ASSERT_EQ(hash_size, hashes[s].size());
    }

    for (uint32_t s = 0; s < SKETCH_DIM; ++s) {
        for (uint32_t i = 0; i < hash_size; ++i) {
            uint8_t v = this->hash(s, i);
            ASSERT_EQ(v, hashes[s][i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Method,
                         Hash2,
                         ::testing::Values(HashAlgorithm::uniform,
                                           HashAlgorithm::crc32,
                                           HashAlgorithm::murmur));

} // namespace
