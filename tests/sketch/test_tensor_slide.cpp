#include "sketch/tensor_slide.hpp"

#include "util/utils.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <random>

namespace {

using namespace ts;
using namespace ::testing;

constexpr uint32_t alphabet_size = 4;
const uint32_t set_size = int_pow<uint32_t>(alphabet_size, 3); // k-mers of length 3
constexpr uint32_t sketch_dim = 2;
constexpr uint32_t tuple_length = 3;
constexpr uint32_t window_length = 32;
constexpr uint32_t stride = 8;


template <typename set_type>
void rand_init(uint32_t sketch_size, Vec2D<set_type> *hashes, Vec2D<bool> *signs) {
    std::mt19937 gen(3412343);
    std::uniform_int_distribution<set_type> rand_hash(0, sketch_size - 1);
    std::uniform_int_distribution<uint8_t> rand_bool(0, 1);

    for (size_t h = 0; h < hashes->size(); h++) {
        for (size_t c = 0; c < alphabet_size; c++) {
            (*hashes)[h][c] = rand_hash(gen);
            (*signs)[h][c] = rand_bool(gen);
        }
    }
}

TEST(TensorSlide, Empty) {
    TensorSlide<uint8_t> under_test(set_size, sketch_dim, tuple_length, window_length,
                                             stride, /*seed=*/ 31415);
    Vec2D<double> sketch = under_test.compute(std::vector<uint8_t>());
    ASSERT_EQ(0, sketch.size());
}

TEST(TensorSlide, TupleOne) {
    constexpr uint32_t tuple_len = 1;
    Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_len, alphabet_size);
    Vec2D<bool> signs = new2D<bool>(tuple_len, alphabet_size);
    rand_init(sketch_dim, &hashes, &signs);

    TensorSlide<uint8_t> under_test(set_size, sketch_dim, 1, 1, 1, /*seed=*/ 31415);
    under_test.set_hashes_for_testing(hashes, signs);

    std::vector<uint8_t> sequence(4);
    std::iota(sequence.begin(), sequence.end(), 0);
    Vec2D<double> sketch = under_test.compute(sequence);
    ASSERT_EQ(sequence.size(), sketch.size());

    for (uint32_t s = 0; s < sequence.size(); ++s) {
        for (uint32_t i = 0; i < sketch_dim; ++i) {
            int8_t sign = signs[0][sequence[s]] ? 1 : -1;
            ASSERT_EQ(sign * sketch[s][i], hashes[0][sequence[s]] % sketch_dim == i)
                    << "Char: " << (int)sequence[s] << " Index: " << i;
        }
    }
}

TEST(TensorSlide, OneCharStrideOne) {
    constexpr uint32_t tuple_len = 1;
    Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_len, alphabet_size);
    Vec2D<bool> signs = new2D<bool>(tuple_len, alphabet_size);
    rand_init(sketch_dim, &hashes, &signs);
    TensorSlide<uint8_t> under_test(set_size, sketch_dim, tuple_len, 1, 1, /*seed=*/ 31415);
    under_test.set_hashes_for_testing(hashes, signs);

    for (uint8_t c = 0; c < alphabet_size; ++c) {
        Vec2D<double> sketch = under_test.compute({ c });
        for (uint32_t i = 0; i < sketch_dim; ++i) {
            int8_t sign = signs[0][c] ? 1 : -1;
            ASSERT_EQ(sign * sketch[0][i], hashes[0][c] % sketch_dim == i) << "Char: " << (int)c;
        }
    }
}

TEST(TensorSlide, TwoCharsStrideOne) {
    constexpr uint32_t tuple_len = 1;
    Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_len, alphabet_size);
    Vec2D<bool> signs = new2D<bool>(tuple_len, alphabet_size);
    rand_init(sketch_dim, &hashes, &signs);
    TensorSlide<uint8_t> tensor_slide(set_size, sketch_dim, tuple_len, 1, 1, /*seed=*/ 31415);
    Tensor<uint8_t> tensor_sketch(set_size, sketch_dim, tuple_len, /*seed=*/ 31415);
    tensor_sketch.set_hashes_for_testing(hashes, signs);
    tensor_slide.set_hashes_for_testing(hashes, signs);

    for (uint8_t c = 0; c < alphabet_size - 1; ++c) {
        Vec2D<double> sketch_slide = tensor_slide.compute({ c, (uint8_t)(c + 1) });
        ASSERT_EQ(2, sketch_slide.size());
        for (uint32_t i : { 0, 1 }) {
            std::vector<double> sketch = tensor_sketch.compute({ (uint8_t)(c + i) });
            ASSERT_THAT(sketch_slide[i], ElementsAreArray(sketch));
        }
    }
}

TEST(TensorSlide, ThreeCharsStrideOne) {
    constexpr uint32_t tuple_len = 1;
    Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_len, alphabet_size);
    Vec2D<bool> signs = new2D<bool>(tuple_len, alphabet_size);
    rand_init(sketch_dim, &hashes, &signs);
    TensorSlide<uint8_t> tensor_slide(set_size, sketch_dim, tuple_len, 1, 1, /*seed=*/ 31415);
    Tensor<uint8_t> tensor_sketch(set_size, sketch_dim, tuple_len, /*seed=*/ 31415);
    tensor_sketch.set_hashes_for_testing(hashes, signs);
    tensor_slide.set_hashes_for_testing(hashes, signs);

    for (uint8_t c = 0; c < alphabet_size - 2; ++c) {
        Vec2D<double> sketch_slide
                = tensor_slide.compute({ c, (uint8_t)(c + 1), (uint8_t)(c + 2) });
        ASSERT_EQ(3, sketch_slide.size());
        for (uint32_t i : { 0, 1, 2 }) {
            std::vector<double> sketch = tensor_sketch.compute({ (uint8_t)(c + i) });
            ASSERT_THAT(sketch_slide[i], ElementsAreArray(sketch));
        }
    }
}

TEST(TensorSlide, TwoCharsStrideTwo) {
    constexpr uint32_t tuple_len = 2;
    Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_len, alphabet_size);
    Vec2D<bool> signs = new2D<bool>(tuple_len, alphabet_size);
    rand_init(sketch_dim, &hashes, &signs);
    TensorSlide<uint8_t> tensor_slide(set_size, sketch_dim, tuple_len, 2, 2, /*seed=*/ 31415);
    Tensor<uint8_t> tensor_sketch(set_size, sketch_dim, tuple_len, /*seed=*/ 31415);
    tensor_sketch.set_hashes_for_testing(hashes, signs);
    tensor_slide.set_hashes_for_testing(hashes, signs);

    for (uint8_t c = 0; c < alphabet_size - 1; ++c) {
        std::vector<double> sketch = tensor_sketch.compute({ c, (uint8_t)(c + 1) });
        Vec2D<double> sketch_slide = tensor_slide.compute({ c, (uint8_t)(c + 1) });
        ASSERT_EQ(1, sketch_slide.size());
        ASSERT_THAT(sketch_slide[0], ElementsAreArray(sketch));
    }
}

TEST(TensorSlide, ThreeChars) {
    constexpr uint32_t tuple_len = 3;
    Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_len, alphabet_size);
    Vec2D<bool> signs = new2D<bool>(tuple_len, alphabet_size);
    rand_init(sketch_dim, &hashes, &signs);
    TensorSlide<uint8_t> tensor_slide(set_size, sketch_dim, tuple_len, 3, 3, /*seed=*/ 31415);
    Tensor<uint8_t> tensor_sketch(set_size, sketch_dim, tuple_len, /*seed=*/ 31415);
    tensor_sketch.set_hashes_for_testing(hashes, signs);
    tensor_slide.set_hashes_for_testing(hashes, signs);

    for (uint8_t c = 0; c < alphabet_size - 2; ++c) {
        std::vector<double> sketch
                = tensor_sketch.compute({ c, (uint8_t)(c + 1), (uint8_t)(c + 2) });
        Vec2D<double> sketch_slide
                = tensor_slide.compute({ c, (uint8_t)(c + 1), (uint8_t)(c + 2) });
        ASSERT_EQ(1, sketch_slide.size());
        ASSERT_THAT(sketch_slide[0], ElementsAreArray(sketch));
    }
}

TEST(TensorSlide, SameAsTensorSketchOneWindow) {
    std::mt19937 gen(3412343);
    std::uniform_int_distribution<uint8_t> rand_alphabet(0, alphabet_size - 1);
    for (uint32_t trial = 0; trial < 10; ++trial) {
        const uint32_t sketch_dimension = std::uniform_int_distribution<uint8_t>(3, 10)(gen);
        const uint32_t tuple_size = std::uniform_int_distribution<uint8_t>(3, 10)(gen);
        const uint32_t sequence_size = std::uniform_int_distribution<uint8_t>(tuple_size, 50)(gen);

        Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_size, alphabet_size);
        Vec2D<bool> signs = new2D<bool>(tuple_size, alphabet_size);
        rand_init(sketch_dim, &hashes, &signs);

        Tensor<uint8_t> tensor_sketch(alphabet_size, sketch_dimension, tuple_size, /*seed=*/ 31415);
        tensor_sketch.set_hashes_for_testing(hashes, signs);

        TensorSlide<uint8_t> tensor_slide(alphabet_size, sketch_dimension, tuple_size,
                                                   sequence_size, sequence_size, /*seed=*/ 31415);
        tensor_slide.set_hashes_for_testing(hashes, signs);

        std::vector<uint8_t> sequence(sequence_size);
        std::transform(sequence.begin(), sequence.end(), sequence.begin(),
                       [&](uint8_t) { return rand_alphabet(gen); });
        std::vector<double> sketch = tensor_sketch.compute(sequence);
        Vec2D<double> slide_sketch = tensor_slide.compute(sequence);

        ASSERT_EQ(1, slide_sketch.size());
        ASSERT_EQ(sketch_dimension, sketch.size());
        ASSERT_THAT(sketch, ElementsAreArray(slide_sketch[0]));
    }
}

TEST(TensorSlide, SameAsTensorSketchMultipleWindows) {
    std::mt19937 gen(3412343);
    std::uniform_int_distribution<uint8_t> rand_alphabet(0, alphabet_size - 1);
    for (uint32_t trial = 0; trial < 10; ++trial) {
        const uint32_t sketch_dimension = std::uniform_int_distribution<uint8_t>(3, 10)(gen);
        const uint32_t tuple_size = std::uniform_int_distribution<uint8_t>(3, 10)(gen);
        const uint32_t window_size = std::uniform_int_distribution<uint8_t>(tuple_size, 12)(gen);
        const uint32_t sequence_size = 3 * window_size;

        Vec2D<uint8_t> hashes = new2D<uint8_t>(tuple_size, alphabet_size);
        Vec2D<bool> signs = new2D<bool>(tuple_size, alphabet_size);
        rand_init(sketch_dim, &hashes, &signs);

        Tensor<uint8_t> tensor_sketch(alphabet_size, sketch_dimension, tuple_size, /*seed=*/ 31415);
        tensor_sketch.set_hashes_for_testing(hashes, signs);

        TensorSlide<uint8_t> tensor_slide(alphabet_size, sketch_dimension, tuple_size,
                                                   window_size, window_size, /*seed=*/ 31415);
        tensor_slide.set_hashes_for_testing(hashes, signs);

        std::vector<uint8_t> sequence(sequence_size);
        std::transform(sequence.begin(), sequence.end(), sequence.begin(),
                       [&](uint8_t) { return rand_alphabet(gen); });
        Vec2D<double> sketch(3);
        sketch[0] = tensor_sketch.compute(
                std::vector<uint8_t>(sequence.begin(), sequence.begin() + window_size));
        sketch[1] = tensor_sketch.compute(std::vector<uint8_t>(sequence.begin() + window_size,
                                                               sequence.begin() + 2 * window_size));
        sketch[2] = tensor_sketch.compute(
                std::vector<uint8_t>(sequence.begin() + 2 * window_size, sequence.end()));
        Vec2D<double> slide_sketch = tensor_slide.compute(sequence);

        ASSERT_EQ(3, slide_sketch.size());
        for (uint32_t i = 0; i < 3; ++i) {
            for (uint32_t j = 0; j < sketch_dim; ++j) {
                ASSERT_NEAR(sketch[i][j], slide_sketch[i][j], 1e-5);
            }
        }
    }
}

} // namespace
