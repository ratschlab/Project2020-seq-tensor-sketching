#include "util/seqgen.hpp"

#include <gtest/gtest.h>

#include <random>

namespace {
using namespace ts;

template <typename T>
class Seq2Kmer : public ::testing::Test {};

typedef ::testing::Types<uint64_t, uint32_t> PowTypes;

TYPED_TEST_SUITE(Seq2Kmer, PowTypes);

TYPED_TEST(Seq2Kmer, Empty) {
    Vec<int> kmers = seq2kmer<int, int>(Seq<int>(), 31, 4);
    ASSERT_EQ(0, kmers.size());
}

} // namespace
