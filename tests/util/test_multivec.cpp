#include "util/utils.hpp"

#include <gtest/gtest.h>

#include <random>

namespace {
template <typename T>
class Pow : public ::testing::Test {};

typedef ::testing::Types<uint64_t, uint32_t> PowTypes;

TYPED_TEST_SUITE(Pow, PowTypes);

TYPED_TEST(Pow, Zero) {
    std::mt19937 rng(123457);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 10000);

    for (uint32_t i = 0; i < 10; ++i) {
        EXPECT_EQ(1, ts::int_pow<TypeParam>(dist(rng), 0));
    }
}

TYPED_TEST(Pow, Random) {
    std::mt19937 rng(123457);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 10);
    std::uniform_int_distribution<std::mt19937::result_type> pow_dist(0, 5);

    for (uint32_t i = 0; i < 10; ++i) {
        TypeParam base = pow_dist(rng);
        TypeParam exp = dist(rng);
        EXPECT_EQ(std::pow(base, exp), ts::int_pow<TypeParam>(base, exp));
    }
}

} // namespace
