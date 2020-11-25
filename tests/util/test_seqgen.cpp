#include "util/seqgen.hpp"

#include <gtest/gtest.h>

#include <random>

namespace {
using namespace ts;

template <typename T>
class Seq2Kmer : public ::testing::Test {};

typedef ::testing::Types<uint64_t, uint32_t> KmerTypes;

TYPED_TEST_SUITE(Seq2Kmer, KmerTypes);

TYPED_TEST(Seq2Kmer, Empty) {
    Vec<int> kmers = seq2kmer<int, int>(Seq<int>(), 31, 4);
    ASSERT_EQ(0, kmers.size());
}

TYPED_TEST(Seq2Kmer, Sequence) {
    Seq<uint8_t> sequence = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
    Vec<TypeParam> expected_kmers = { 0, 0, 16, 20, 21, 21, 37, 41, 42, 42, 58, 62, 63, 63 };
    Vec<TypeParam> kmers = seq2kmer<uint8_t, TypeParam>(sequence, 3, 4);

    ASSERT_EQ(expected_kmers.size(), kmers.size());
    for (uint32_t i = 0; i < kmers.size(); ++i) {
        ASSERT_EQ(expected_kmers[i], kmers[i]);
    }
}

} // namespace
