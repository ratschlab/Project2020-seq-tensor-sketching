#include "sketch/tensor2.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace {

using namespace ts;
using namespace ::testing;

constexpr uint8_t alphabet_size = 4;
const uint32_t set_size = int_pow<uint32_t>(alphabet_size, 3); // k-mers of length 3
constexpr uint32_t sketch_dim = 2;
constexpr uint32_t tuple_length = 3;

TEST(Tensor2, Empty) {
    Tensor2<uint8_t, double> under_test(alphabet_size, sketch_dim, tuple_length);
    Vec<double> sketch = under_test.compute(std::vector<uint8_t>());
    ASSERT_EQ(sketch.size(), sketch_dim);
    ASSERT_THAT(sketch, ElementsAre(0, 0));
}

/** The sequence has one char, which is shorter than the tuple length, so the sketch will be 0 */
TEST(Tensor2, OneChar) {
    Tensor2<uint8_t, double> under_test(alphabet_size, sketch_dim, tuple_length);
    for (uint8_t c = 0; c < alphabet_size; ++c) {
        Vec<double> sketch = under_test.compute({ c });
        ASSERT_THAT(sketch, ElementsAre(0, 0));
    }
}

/**
 * The size of the sequence equals the size of the tuple, so the sketch will be 1 in one position
 * (position H(x)), and 0 in all the other positions
 */
TEST(Tensor2, FullStringDistinctChars) {
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        for (uint32_t tuple_len = 2; tuple_len < 10; ++tuple_len) {
            Tensor2<uint8_t, double> under_test(tuple_len, sketch_dimension, tuple_len);
            std::vector<uint8_t> sequence(tuple_len);
            std::iota(sequence.begin(), sequence.end(), 0U);
            Vec<double> sketch = under_test.compute(sequence);
            ASSERT_EQ(sketch.size(), sketch_dimension);
            for (uint32_t i = 0; i < sketch_dimension; ++i) {
                ASSERT_TRUE(std::abs(sketch[i]) == 0 || std::abs(sketch[i]) == 1);
            }
            ASSERT_EQ(1, std::abs(std::accumulate(sketch.begin(), sketch.end(), 0)));
        }
    }
}

/**
 * The size of the sequence equals the size of the tuple, so the sketch will be 1 in one position
 * (position H(x)), and 0 in all the other positions
 */
TEST(Tensor2, FullStringRandomChars) {
    std::mt19937 gen(1234567);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        for (uint32_t tuple_len = 2; tuple_len < 10; ++tuple_len) {
            std::uniform_int_distribution<uint8_t> rand_char(0, tuple_len - 1);
            Tensor2<uint8_t, double> under_test(tuple_len, sketch_dimension, tuple_len);
            std::vector<uint8_t> sequence(tuple_len);
            for (uint8_t &c : sequence) {
                c = rand_char(gen);
            }
            Vec<double> sketch = under_test.compute(sequence);
            ASSERT_EQ(sketch.size(), sketch_dimension);
            for (uint32_t i = 0; i < sketch_dimension; ++i) {
                ASSERT_TRUE(std::abs(sketch[i]) == 0 || std::abs(sketch[i]) == 1);
            }
            ASSERT_EQ(1, std::abs(std::accumulate(sketch.begin(), sketch.end(), 0)))
                    << "Dim=" << sketch_dimension << " t=" << tuple_len;
        }
    }
}

/**
 * If a sequence contains identical characters, its sketch will be +/-1 in one position and 0 in all
 * others, because all subsequences of length t are identical.
 */
TEST(Tensor2, SameChars) {
    std::mt19937 gen(321567);
    std::uniform_int_distribution<uint8_t> rand_char(0, alphabet_size - 1);
    std::uniform_int_distribution<uint8_t> rand_seq_len(0, 100);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        for (uint32_t tuple_len = 2; tuple_len < 10; ++tuple_len) {
            Tensor2<uint8_t, double> under_test(tuple_len, sketch_dimension, tuple_len);
            uint8_t sequence_length = tuple_len + rand_seq_len(gen);
            std::vector<uint8_t> sequence(sequence_length, rand_char(gen));
            Vec<double> sketch = under_test.compute(sequence);
            ASSERT_EQ(sketch.size(), sketch_dimension);
            for (uint32_t i = 0; i < sketch_dimension; ++i) {
                ASSERT_TRUE(std::abs(sketch[i]) == 0 || std::abs(sketch[i]) == 1);
            }
            ASSERT_EQ(1, std::abs(std::accumulate(sketch.begin(), sketch.end(), 0)))
                    << "Dim=" << sketch_dimension << " t=" << tuple_len;
        }
    }
}

/**
 * If a sequence contains distinct characters, the the tensor sketch for t=1 will contain multiples
 * of (1/alphabet_size), because T(a)=1/alphabet_size for all characters a.
 */
TEST(Tensor2, DistinctCharsTuple1) {
    std::mt19937 gen(321567);
    constexpr uint8_t tuple_len = 1;
    std::vector<uint8_t> sequence(alphabet_size);
    std::iota(sequence.begin(), sequence.end(), 0);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        Tensor2<uint8_t, double> under_test(alphabet_size, sketch_dimension, tuple_len);

        Vec<double> sketch = under_test.compute(sequence);
        ASSERT_EQ(sketch.size(), sketch_dimension);
        for (uint32_t i = 0; i < sketch_dimension; ++i) {
            double factor = sketch[i] / (1. / alphabet_size);
            ASSERT_NEAR(0, std::round(factor) - factor, 1e-3);
        }
    }
}

/**
 * If a sequence of length seq_len contains distinct characters, the the tensor sketch for
 * t=seq_len-1 will contain multiples of (1/t), because T(a)=1/t for all the seq_len subsequences of
 * length seq_len-1
 */
TEST(Tensor2, DistinctCharsTupleTMinus1) {
    std::mt19937 gen(321567);
    for (uint32_t tuple_len = 1; tuple_len < 10; ++tuple_len) {
        const uint8_t alphabet_size = tuple_len + 1;
        std::vector<uint8_t> sequence(alphabet_size);
        std::iota(sequence.begin(), sequence.end(), 0);
        for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
            Tensor2<uint8_t, double> under_test(alphabet_size, sketch_dimension, tuple_len);

            Vec<double> sketch = under_test.compute(sequence);
            ASSERT_EQ(sketch.size(), sketch_dimension);
            for (uint32_t i = 0; i < sketch_dimension; ++i) {
                double factor = sketch[i] / (1. / alphabet_size);
                ASSERT_NEAR(0, std::round(factor) - factor, 1e-3);
            }
        }
    }
}

} // namespace
