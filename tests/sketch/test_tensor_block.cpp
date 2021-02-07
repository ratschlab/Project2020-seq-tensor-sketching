#include "sketch/tensor.hpp"
#include "sketch/tensor_block.hpp"

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

TEST(TensorBlock, Empty) {
    TensorBlock<uint8_t> under_test(alphabet_size, sketch_dim, tuple_length, 1, /*seed=*/31415);
    std::vector<double> sketch = under_test.compute(std::vector<uint8_t>());
    ASSERT_EQ(sketch.size(), sketch_dim);
    ASSERT_THAT(sketch, ElementsAre(0, 0));
}

/** The sequence has one char, which is shorter than the tuple length, so the sketch will be 0 */
TEST(TensorBlock, OneChar) {
    TensorBlock<uint8_t> under_test(alphabet_size, sketch_dim, tuple_length, 1, /*seed=*/31415);
    for (uint8_t c = 0; c < alphabet_size; ++c) {
        std::vector<double> sketch = under_test.compute({ c });
        ASSERT_THAT(sketch, ElementsAre(0, 0));
    }
}

/** The sequence has one char, the tuple length is 1, so we should have a value of s(seq[0]) on
 * position h(seq[0]) */
TEST(TensorBlock, OneCharTuple1) {
    constexpr uint32_t tuple_len = 1;
    TensorBlock<uint8_t> under_test(alphabet_size, sketch_dim, tuple_len, 1, /*seed=*/31415);

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
 * The size of the sequence equals the size of the tuple, so the sketch will be 1 or -1 in one
 * position (position H(x)), and 0 in all the other positions, no matter what the block size is.
 */
TEST(TensorBlock, FullStringRandomChars) {
    std::mt19937 gen(1234567);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        for (uint32_t tuple_len = 1; tuple_len < 10; ++tuple_len) {
            for (uint32_t block_size = 1; block_size < sqrt(tuple_len); ++block_size) {
                if (tuple_len % block_size != 0) {
                    continue;
                }
                std::uniform_int_distribution<uint8_t> rand_char(0, alphabet_size - 1);
                TensorBlock<uint8_t> under_test(alphabet_size, sketch_dimension, tuple_len,
                                                block_size,
                                                /*seed=*/31415);

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
                    ASSERT_EQ(i == pos ? s : 0, sketch[i])
                            << "t=" << tuple_len << " k=" << block_size << " i=" << i;
                }
            }
        }
    }
}

/**
 * If a sequence contains identical characters, its sketch will be +/-1 in one position and 0 in all
 * others, no matter what the block size, because all subsequences of length t are identical.
 */
TEST(TensorBlock, SameChars) {
    std::mt19937 gen(342111);
    std::uniform_int_distribution<uint8_t> rand_char(0, alphabet_size - 1);
    std::uniform_int_distribution<uint8_t> rand_seq_len(0, 100);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        for (uint32_t tuple_len = 2; tuple_len < 10; ++tuple_len) {
            for (uint32_t block_size = 1; block_size < sqrt(tuple_len); ++block_size) {
                if (tuple_len % block_size != 0) {
                    continue;
                }
                TensorBlock<uint8_t> under_test(tuple_len, sketch_dimension, tuple_len, block_size,
                                                /*seed=*/127383);
                uint8_t sequence_length = tuple_len + rand_seq_len(gen);
                std::vector<uint8_t> sequence(sequence_length, rand_char(gen));
                std::vector<double> sketch = under_test.compute(sequence);
                ASSERT_EQ(sketch.size(), sketch_dimension);
                for (uint32_t i = 0; i < sketch_dimension; ++i) {
                    ASSERT_TRUE(std::abs(sketch[i]) == 0 || std::abs(sketch[i]) == 1)
                            << "Dim=" << sketch_dimension << " t=" << tuple_len
                            << " block=" << block_size;
                }
                ASSERT_EQ(1, std::abs(std::accumulate(sketch.begin(), sketch.end(), 0)));
            }
        }
    }
}

/**
 * If a sequence contains distinct characters, then the tensor sketch for t=1 will contain multiples
 * of (1/alphabet_size), because T(a)=1/alphabet_size for all characters a.
 */
TEST(TensorBlock, DistinctCharsTuple1) {
    std::mt19937 gen(321567);
    constexpr uint8_t tuple_len = 1;
    std::vector<uint8_t> sequence(alphabet_size);
    std::iota(sequence.begin(), sequence.end(), 0);
    for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
        TensorBlock<uint8_t> under_test(alphabet_size, sketch_dimension, tuple_len, 1,
                                        /*seed=*/31415);

        std::vector<double> sketch = under_test.compute(sequence);
        ASSERT_EQ(sketch.size(), sketch_dimension);
        for (uint32_t i = 0; i < sketch_dimension; ++i) {
            double factor = sketch[i] / (1. / alphabet_size);
            ASSERT_NEAR(0, std::round(factor) - factor, 1e-3);
        }
    }
}

/**
 * If a sequence of length seq_len contains distinct characters, then the tensor sketch for
 * t=seq_len-1 and block_size k will contain multiples of k/(t+k), because T(a)=k/(t+k) for
 * all the t/k + 1 subsequences of length t formed out of blocks of size (k denotes the block size).
 */
TEST(TensorBlock, DistinctCharsTupleTMinus1) {
    std::mt19937 gen(321567);
    for (uint32_t tuple_len = 1; tuple_len < 10; ++tuple_len) {
        const uint8_t alphabet_sz = tuple_len + 1;
        std::vector<uint8_t> sequence(alphabet_sz);
        std::iota(sequence.begin(), sequence.end(), 0);
        for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
            for (uint32_t block_size = 1; block_size < sqrt(tuple_len); ++block_size) {
                if (tuple_len % block_size != 0) {
                    continue;
                }
                TensorBlock<uint8_t> under_test(alphabet_sz, sketch_dimension, tuple_len,
                                                block_size, /*seed=*/31415);
                std::vector<double> sketch = under_test.compute(sequence);

                ASSERT_EQ(sketch.size(), sketch_dimension);
                for (uint32_t i = 0; i < sketch_dimension; ++i) {
                    double factor = sketch[i]
                            / (static_cast<double>(block_size) / (block_size + tuple_len));
                    ASSERT_NEAR(factor, std::round(factor), 1e-3)
                            << tuple_len << " " << sketch_dimension << " " << i << " "
                            << block_size;
                }
            }
        }
    }
}

/**
 * For a block size of 1, TensorBlock and Tensor should return identical results.
 */
TEST(TensorBlock, SameAsTensorSketchForBlockSizeOne) {
    std::mt19937 gen(321567);
    std::uniform_int_distribution<uint8_t> rand_sigma(3, 20);
    for (uint32_t tuple_len = 1; tuple_len < 10; ++tuple_len) {
        const uint8_t alphabet_sz = rand_sigma(gen);
        std::uniform_int_distribution<uint8_t> rand_char(0, alphabet_sz - 1);

        std::vector<uint8_t> sequence(std::uniform_int_distribution<uint8_t>(0, 100)(gen));
        std::generate(sequence.begin(), sequence.end(), [&]() { return rand_char(gen); });
        for (uint32_t sketch_dimension = 3; sketch_dimension < 10; ++sketch_dimension) {
            TensorBlock<uint8_t> block(alphabet_sz, sketch_dimension, tuple_len, 1, /*seed=*/31415);
            Tensor<uint8_t> sketch(alphabet_sz, sketch_dimension, tuple_len, /*seed=*/31415);

            std::vector<double> sketch1 = block.compute(sequence);
            std::vector<double> sketch2 = sketch.compute(sequence);

            ASSERT_THAT(sketch1, ElementsAreArray(sketch2));
        }
    }
}

} // namespace
