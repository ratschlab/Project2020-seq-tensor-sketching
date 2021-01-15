
#include "sketch/hash_weighted.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

namespace {

using namespace ts;
using namespace ::testing;

TEST(WeightedMinHash, Empty) {
    WeightedMinHash<uint8_t> under_test(4 * 4 * 4, 3, 100, HashAlgorithm::uniform, /*seed=*/31415);
    std::vector<uint8_t> sketch = under_test.compute(std::vector<uint8_t>());
    ASSERT_THAT(sketch, ElementsAre(0, 0, 0));
}

TEST(WeightedMinHash, Repeat) {
    WeightedMinHash<uint8_t> under_test(4 * 4 * 4, 3, 100, HashAlgorithm::uniform, /*seed=*/31415);
    std::vector<uint8_t> sequence = { 0, 1, 2, 3, 4, 5 };
    std::vector<uint8_t> sketch1 = under_test.compute(sequence);
    std::vector<uint8_t> sketch2 = under_test.compute(sequence);
    ASSERT_THAT(sketch1, ElementsAreArray(sketch2));
}

TEST(WeightedMinHash, Permute) {
    WeightedMinHash<uint8_t> under_test(4 * 4 * 4, 3, 100, HashAlgorithm::uniform, /*seed=*/31415);
    std::vector<uint8_t> sequence1 = { 0, 1, 2, 3, 4, 5 };
    std::vector<uint8_t> sequence2 = { 5, 4, 3, 2, 1, 0 };
    std::vector<uint8_t> sketch1 = under_test.compute(sequence1);
    std::vector<uint8_t> sketch2 = under_test.compute(sequence2);
    ASSERT_THAT(sketch1, ElementsAreArray(sketch2));
}

std::vector<std::unordered_map<uint8_t, uint8_t>>
hash_init(uint32_t set_sz, uint32_t sketch_size, uint32_t max_seq_len) {
    std::vector<std::unordered_map<uint8_t, uint8_t>> hashes(sketch_size);
    for (size_t m = 0; m < sketch_size; m++) {
        for (uint32_t v = 0; v < set_sz * max_seq_len; ++v) {
            hashes[m][v] = v;
        }
    }
    return hashes;
}

TEST(WeightedMinHash, PresetHash) {
    WeightedMinHash<uint8_t> under_test(4 * 4, 3, 100, HashAlgorithm::uniform, /*seed=*/31415);
    under_test.set_hashes_for_testing(hash_init(4 * 4, 3, 100));
    for (uint32_t i = 0; i < 4 * 4; ++i) {
        std::vector<uint8_t> sequence(4 * 4 - i);
        std::iota(sequence.begin(), sequence.end(), i);
        std::vector<uint8_t> sketch = under_test.compute(sequence);
        ASSERT_THAT(sketch, ElementsAreArray({ i, i, i }));
    }
}

TEST(WeightedMinHash, PresetHashRepeat) {
    constexpr uint32_t set_size = 4 * 4; // corresponds to k-mers of length 2 over the DNA alphabet
    WeightedMinHash<uint8_t> under_test(set_size, 3, 100, HashAlgorithm::uniform, /*seed=*/31415);
    under_test.set_hashes_for_testing(hash_init(set_size, 3, 100));
    for (uint32_t i = 0; i < set_size; ++i) {
        std::vector<uint8_t> sequence(2 * (set_size - i));
        std::iota(sequence.begin(), sequence.begin() + sequence.size() / 2, i);
        std::iota(sequence.begin() + sequence.size() / 2, sequence.end(), i);
        std::vector<uint8_t> sketch = under_test.compute(sequence);
        ASSERT_THAT(sketch, ElementsAreArray({ i, i, i }));
    }
}

#ifndef NDEBUG
TEST(WeightedMinhash, SequenceTooLong) {
    constexpr uint32_t set_size = 4 * 4; // corresponds to k-mers of length 2 over the DNA alphabet
    WeightedMinHash<uint8_t> under_test(set_size, 3, 100, HashAlgorithm::uniform, /*seed=*/31415);
    std::vector<uint8_t> sequence(100 + 1);
    ASSERT_THROW(under_test.compute(sequence), std::invalid_argument);
}
#endif

} // namespace
