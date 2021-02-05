#include "sketch/tensor.hpp"
#include "util/utils.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using namespace ts;
using namespace ::testing;

constexpr uint8_t alphabet_size = 4;
const uint32_t set_size = int_pow<uint32_t>(alphabet_size, 3); // k-mers of length 3
constexpr uint32_t sketch_dim = 2;
constexpr uint32_t tuple_length = 3;

template <typename set_type>
void rand_init(uint32_t sketch_size, Vec2D<set_type> *hashes, Vec2D<bool> *signs) {
    std::mt19937 gen(3412343);
    std::uniform_int_distribution<set_type> rand_hash2(0, sketch_size - 1);
    std::uniform_int_distribution<set_type> rand_bool(0, 1);

    for (size_t h = 0; h < hashes->size(); h++) {
        for (size_t c = 0; c < alphabet_size; c++) {
            (*hashes)[h][c] = rand_hash2(gen);
            (*signs)[h][c] = rand_bool(gen);
        }
    }
}

TEST(Tensor, Empty) {
    Tensor<uint8_t> under_test(alphabet_size, sketch_dim, tuple_length, /*seed=*/31415);
    std::vector<double> sketch = under_test.compute(std::vector<uint8_t>());
    ASSERT_EQ(sketch.size(), sketch_dim);
    ASSERT_THAT(sketch, ElementsAre(0, 0));
}

/** The sequence has one char, which is shorter than the tuple length, so the sketch will be 0 */
TEST(Tensor, OneChar) {
    Tensor<uint8_t> under_test(alphabet_size, sketch_dim, tuple_length, /*seed=*/31415);
    for (uint8_t c = 0; c < alphabet_size; ++c) {
        std::vector<double> sketch = under_test.compute({ c });
        ASSERT_THAT(sketch, ElementsAre(0, 0));
    }
}

/** The sequence has one char, the tuple length is 1, so we should have a value of +/-1 on position
 * h(seq[0]) */
TEST(Tensor, OneCharTuple1) {
    constexpr uint32_t tuple_len = 1;
    Tensor<uint8_t> under_test(alphabet_size, sketch_dim, tuple_len, /*seed=*/31415);

    Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_len, alphabet_size);
    Vec2D<bool> signs = new2D<bool>(tuple_len, alphabet_size);
    rand_init(sketch_dim, &hashes, &signs);
    under_test.set_hashes_for_testing(hashes, signs);

    for (uint8_t c = 0; c < alphabet_size; ++c) {
        std::vector<double> sketch = under_test.compute({ c });
        for (uint32_t i = 0; i < sketch_dim; ++i) {
            int8_t sign = signs[0][c] ? 1 : -1;
            ASSERT_EQ(sketch[i] * sign, hashes[0][c] % sketch_dim == i) << "Char: " << (int)c;
        }
    }
}

/**
 * The size of the sequence equals the size of the tuple, so the sketch will be 1 in one position
 * (position H(x)), and 0 in all the other positions
 */
TEST(Tensor, FullStringDistinctChars) {
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        for (uint32_t tuple_len = 2; tuple_len < 10; ++tuple_len) {
            Tensor<uint8_t> under_test(tuple_len, sketch_dimension, tuple_len, /*seed=*/31415);
            std::vector<uint8_t> sequence(tuple_len);
            std::iota(sequence.begin(), sequence.end(), 0U);
            std::vector<double> sketch = under_test.compute(sequence);
            ASSERT_EQ(sketch.size(), sketch_dimension);
            for (uint32_t i = 0; i < sketch_dimension; ++i) {
                ASSERT_TRUE(std::abs(sketch[i]) == 0 || std::abs(sketch[i]) == 1);
            }
            ASSERT_EQ(1, std::abs(std::accumulate(sketch.begin(), sketch.end(), 0)))
                    << "D=" << sketch_dimension << " t=" << tuple_len;
        }
    }
}

/**
 * The size of the sequence equals the size of the tuple, so the sketch will be 1 or -1 in one
 * position (position H(x)), and 0 in all the other positions.
 */
TEST(Tensor, FullStringRandomChars) {
    std::mt19937 gen(1234567);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        for (uint32_t tuple_len = 2; tuple_len < 10; ++tuple_len) {
            std::uniform_int_distribution<uint8_t> rand_char(0, alphabet_size - 1);
            Tensor<uint8_t> under_test(alphabet_size, sketch_dimension, tuple_len, /*seed=*/31415);

            Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_len, alphabet_size);
            Vec2D<bool> signs = new2D<bool>(tuple_len, alphabet_size);
            rand_init(sketch_dim, &hashes, &signs);
            under_test.set_hashes_for_testing(hashes, signs);

            std::vector<uint8_t> sequence(tuple_len);
            for (uint8_t &c : sequence) {
                c = rand_char(gen);
            }
            std::vector<double> sketch = under_test.compute(sequence);

            uint32_t pos = 0; // the position where the sketch must be one
            int8_t s = 1; // the sign of the sketch
            for (uint32_t i = 0; i < sequence.size(); ++i) {
                pos += hashes[i][sequence[i]];
                s *= signs[i][sequence[i]] ? 1 : -1;
            }
            pos %= sketch_dimension;

            ASSERT_EQ(sketch.size(), sketch_dimension);
            for (uint32_t i = 0; i < sketch_dimension; ++i) {
                ASSERT_EQ(i == pos ? s : 0, sketch[i]);
            }
        }
    }
}

/**
 * If a sequence contains identical characters, its sketch will be +/-1 in one position and 0 in all
 * others, because all subsequences of length t are identical.
 */
TEST(Tensor, SameChars) {
    std::mt19937 gen(342111);
    std::uniform_int_distribution<uint8_t> rand_char(0, alphabet_size - 1);
    std::uniform_int_distribution<uint8_t> rand_seq_len(0, 100);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        for (uint32_t tuple_len = 2; tuple_len < 10; ++tuple_len) {
            Tensor<uint8_t> under_test(alphabet_size, sketch_dimension, tuple_len, /*seed=*/31415);
            uint8_t sequence_length = tuple_len + rand_seq_len(gen);
            std::vector<uint8_t> sequence(sequence_length, rand_char(gen));
            std::vector<double> sketch = under_test.compute(sequence);
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
 * If a sequence contains distinct characters, then the tensor sketch for t=1 will contain multiples
 * of (1/alphabet_size), because T(a)=1/alphabet_size for all characters a.
 */
TEST(Tensor, DistinctCharsTuple1) {
    std::mt19937 gen(321567);
    constexpr uint8_t tuple_len = 1;
    std::vector<uint8_t> sequence(alphabet_size);
    std::iota(sequence.begin(), sequence.end(), 0);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        Tensor<uint8_t> under_test(alphabet_size, sketch_dimension, tuple_len, /*seed=*/31415);

        std::vector<double> sketch = under_test.compute(sequence);
        ASSERT_EQ(sketch.size(), sketch_dimension);
        for (uint32_t i = 0; i < sketch_dimension; ++i) {
            double factor = sketch[i] / (1. / alphabet_size);
            ASSERT_NEAR(factor, std::round(factor), 1e-3);
        }
    }
}

/**
 * If a sequence of length seq_len contains distinct characters, then the tensor sketch for
 * t=seq_len-1 will contain multiples of (1/seq_len), because T(a)=1/seq_len for all the seq_len
 * subsequences of length seq_len-1.
 */
TEST(Tensor, DistinctCharsTupleTMinus1) {
    std::mt19937 gen(321567);
    for (uint32_t tuple_len = 1; tuple_len < 10; ++tuple_len) {
        const uint8_t alphabet_size = tuple_len + 1;
        std::vector<uint8_t> sequence(alphabet_size);
        std::iota(sequence.begin(), sequence.end(), 0);
        for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
            Tensor<uint8_t> under_test(alphabet_size, sketch_dimension, tuple_len, /*seed=*/31415);

            std::vector<double> sketch = under_test.compute(sequence);
            ASSERT_EQ(sketch.size(), sketch_dimension);
            for (uint32_t i = 0; i < sketch_dimension; ++i) {
                double factor = sketch[i] / (1. / alphabet_size);
                ASSERT_NEAR(factor, std::round(factor), 1e-3);
            }
        }
    }
}

} // namespace
