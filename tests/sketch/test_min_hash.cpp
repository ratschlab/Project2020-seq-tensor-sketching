#include "sketch/hash_min.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <random>

namespace {

using namespace ts;
using namespace ::testing;

TEST(MinHash, Empty) {
    MinHash<uint8_t> under_test(4 * 4 * 4, 3);
    Vec<uint8_t> sketch = under_test.compute(std::vector<uint8_t>());
    ASSERT_THAT(sketch, ElementsAre(0, 0, 0));
}

TEST(MinHash, Repeat) {
    MinHash<uint8_t> under_test(4 * 4 * 4, 3);
    std::vector<uint8_t> sequence = { 0, 1, 2, 3, 4, 5 };
    Vec<uint8_t> sketch1 = under_test.compute(sequence);
    Vec<uint8_t> sketch2 = under_test.compute(sequence);
    ASSERT_THAT(sketch1, ElementsAreArray(sketch2));
}

TEST(MinHash, Permute) {
    MinHash<uint8_t> under_test(4 * 4 * 4, 3);
    std::vector<uint8_t> sequence1 = { 0, 1, 2, 3, 4, 5 };
    std::vector<uint8_t> sequence2 = { 5, 4, 3, 2, 1, 0 };
    Vec<uint8_t> sketch1 = under_test.compute(sequence1);
    Vec<uint8_t> sketch2 = under_test.compute(sequence2);
    ASSERT_THAT(sketch1, ElementsAreArray(sketch2));
}

TEST(MinHash, PermuteAndRepeat) {
    MinHash<uint8_t> under_test(4 * 4 * 4, 3);
    std::vector<uint8_t> sequence1 = { 0, 1, 2, 3, 4, 5 };
    std::vector<uint8_t> sequence2 = { 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0 };
    Vec<uint8_t> sketch1 = under_test.compute(sequence1);
    Vec<uint8_t> sketch2 = under_test.compute(sequence2);
    ASSERT_THAT(sketch1, ElementsAreArray(sketch2));
}

Vec2D<uint8_t> hash_init(uint32_t set_size, uint32_t sketch_dim) {
    Vec2D<uint8_t> hashes(sketch_dim, Vec<uint8_t>(set_size, 0));
    for (size_t m = 0; m < sketch_dim; m++) {
        std::iota(hashes[m].begin(), hashes[m].end(), 0);
    }
    return hashes;
}

TEST(MinHash, PresetHash) {
    MinHash<uint8_t> under_test(4 * 4, 3);
    under_test.set_hashes_for_testing(hash_init(4 * 4, 3));
    for (uint32_t i = 0; i < 4 * 4; ++i) {
        std::vector<uint8_t> sequence(4*4-i);
        std::iota(sequence.begin(), sequence.end(), i);
        Vec<uint8_t> sketch = under_test.compute(sequence);
        ASSERT_THAT(sketch, ElementsAreArray({ i, i, i }));
    }
}

} // namespace
