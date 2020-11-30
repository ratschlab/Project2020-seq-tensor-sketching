#include "sketch/tensor.hpp"

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
constexpr uint32_t num_phases = 2;
constexpr uint32_t num_bins = 255;

TEST(Tensor, Empty) {
    Tensor<uint8_t> under_test(set_size, sketch_dim, num_phases, num_bins, tuple_length);
    Vec<uint8_t> sketch = under_test.compute(std::vector<uint8_t>());
    ASSERT_THAT(sketch, ElementsAre(0, 0, 0));
}

} // namespace

