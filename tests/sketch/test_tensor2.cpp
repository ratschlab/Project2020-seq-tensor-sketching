#include "sketch/tensor2.hpp"

#include <gtest/gtest.h>

namespace {

using namespace ts;
using namespace ::testing;

constexpr uint32_t alphabet_size = 4;
const uint32_t set_size = int_pow<uint32_t>(alphabet_size, 3); // k-mers of length 3
constexpr uint32_t sketch_dim = 2;
constexpr uint32_t tuple_length = 3;
constexpr uint32_t num_phases = 2;

TEST(Tensor2, Empty) {
Tensor2<uint8_t, double> under_test(alphabet_size, sketch_dim, num_phases, tuple_length);
// Vec<uint8_t> sketch = under_test.compute(std::vector<uint8_t>());
//ASSERT_THAT(sketch, ElementsAre(0, 0, 0));
}

} // namespace

