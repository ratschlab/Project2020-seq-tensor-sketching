#include "sketch/hash_ordered.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

namespace {

using namespace ts;
using namespace ::testing;

constexpr uint32_t alphabet_size = 4;
const uint32_t set_size = int_pow<uint32_t>(alphabet_size, 3); // k-mers of length 3
constexpr uint32_t sketch_dim = 2;
constexpr uint32_t tuple_length = 3;
constexpr uint32_t max_sequence_len = 200;

TEST(OrderedMinHash, Empty) {
    OrderedMinHash<uint8_t> under_test(set_size, sketch_dim, max_sequence_len, tuple_length,
                                       HashAlgorithm::uniform, /*seed=*/31415);
    ASSERT_THROW(under_test.compute(std::vector<uint8_t>()), std::invalid_argument);
}

TEST(OrderedMinHash, Repeat) {
    OrderedMinHash<uint8_t> under_test(set_size, sketch_dim, max_sequence_len, tuple_length,
                                       HashAlgorithm::uniform, /*seed=*/31415);
    std::vector<uint8_t> sequence = { 0, 1, 2, 3, 4, 5 };
    Vec2D<uint8_t> sketch1 = under_test.compute_2d(sequence);
    Vec2D<uint8_t> sketch2 = under_test.compute_2d(sequence);
    ASSERT_EQ(sketch_dim, sketch1.size());
    ASSERT_EQ(sketch_dim, sketch2.size());
    for (uint32_t i = 0; i < sketch_dim; ++i) {
        ASSERT_THAT(sketch1[i], ElementsAreArray(sketch2[i]));
    }
}

TEST(OrderedMinHash, ReverseOrder) {
    OrderedMinHash<uint8_t> under_test(set_size, sketch_dim, max_sequence_len, tuple_length,
                                       HashAlgorithm::uniform, /*seed=*/31415);
    std::vector<uint8_t> sequence1 = { 0, 1, 2, 3, 4, 5 };
    std::vector<uint8_t> sequence2 = { 5, 4, 3, 2, 1, 0 };
    Vec2D<uint8_t> sketch1 = under_test.compute_2d(sequence1);
    Vec2D<uint8_t> sketch2 = under_test.compute_2d(sequence2);
    ASSERT_EQ(sketch_dim, sketch1.size());
    ASSERT_EQ(sketch_dim, sketch2.size());
    for (uint32_t i = 0; i < sketch_dim; ++i) {
        std::reverse(sketch1[i].begin(), sketch1[i].end()); // reversed order of appearance
        ASSERT_THAT(sketch1[i], ElementsAreArray(sketch2[i]));
    }
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

TEST(OrderedMinHash, PresetHash) {
    OrderedMinHash<uint8_t> under_test(set_size, sketch_dim, max_sequence_len, tuple_length,
                                       HashAlgorithm::uniform, /*seed=*/31415);
    under_test.set_hashes_for_testing(hash_init(set_size, sketch_dim, max_sequence_len));
    for (uint32_t i = 0; i < set_size - tuple_length; ++i) {
        std::vector<uint8_t> sequence(set_size - i);
        std::iota(sequence.begin(), sequence.end(), i);
        Vec2D<uint8_t> sketch = under_test.compute_2d(sequence);
        for (uint32_t s = 0; s < sketch_dim; ++s) {
            ASSERT_THAT(sketch[s], ElementsAreArray({ i, i + 1, i + 2 }));
        }
    }
}

TEST(OrderedMinHash, PresetHashRepeat) {
    OrderedMinHash<uint8_t> under_test(set_size, sketch_dim, max_sequence_len, tuple_length,
                                       HashAlgorithm::uniform, /*seed=*/ 31415);
    under_test.set_hashes_for_testing(hash_init(set_size, sketch_dim, max_sequence_len));
    for (uint32_t i = 0; i < set_size - tuple_length; ++i) {
        std::vector<uint8_t> sequence(2 * (set_size - i));
        std::iota(sequence.begin(), sequence.begin() + sequence.size() / 2, i);
        std::iota(sequence.begin() + sequence.size() / 2, sequence.end(), i);
        Vec2D<uint8_t> sketch = under_test.compute_2d(sequence);
        for (uint32_t s = 0; s < sketch_dim; ++s) {
            ASSERT_THAT(sketch[s], ElementsAreArray({ i, i + 1, i + 2 }));
        }
    }
}

#ifndef NDEBUG
TEST(OrderedMinhash, SequenceTooLong) {
    OrderedMinHash<uint8_t> under_test(set_size, sketch_dim, max_sequence_len, tuple_length,
                                       HashAlgorithm::uniform, /*seed=*/ 31415);
    std::vector<uint8_t> sequence(max_sequence_len + 1);
    ASSERT_THROW(under_test.compute(sequence), std::invalid_argument);
}
#endif

} // namespace
