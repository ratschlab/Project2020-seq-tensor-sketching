#include "util/spearman.hpp"

#include <gtest/gtest.h>

#include <random>

namespace {

TEST(Spearman, Identical) {
    std::mt19937 rng(123457);
    std::uniform_int_distribution<uint32_t> dist(0, 10000);
    std::uniform_int_distribution<uint32_t> sz_dist(5, 10);
    for (uint32_t trial = 0; trial < 10; ++trial) {
        size_t size = sz_dist(rng);
        std::vector<uint8_t> a;
        std::vector<uint8_t> b;
        for (uint32_t i = 0; i < size; ++i) {
            size_t v = dist(rng);
            a.push_back(v);
            b.push_back(v);
        }
        ASSERT_EQ(1, spearman(a, b));
    }
}

TEST(Spearman, Linear) {
    std::mt19937 rng(123457);
    std::uniform_real_distribution<> dist(1, 10000);
    std::uniform_int_distribution<> sz_dist(5, 10);
    double coef = dist(rng);
    for (uint32_t trial = 0; trial < 10; ++trial) {
        size_t size = sz_dist(rng);
        std::vector<double> a;
        std::vector<double> b;
        for (uint32_t i = 0; i < size; ++i) {
            size_t v = dist(rng);
            a.push_back(v);
            b.push_back(v * coef);
        }
        ASSERT_EQ(1, spearman(a, b)) << "Trial " << trial << " Coef: " << coef;
    }
}

TEST(Spearman, LinearInverse) {
    size_t size = 10;
    std::vector<double> a(size);
    std::vector<double> b(size);
    for (uint32_t i = 0; i < size; ++i) {
        a[i] = 2 * i + 5;
        b[size - i - 1] = 2 * i + 5;
    }
    ASSERT_EQ(-1, spearman(a, b));
}

TEST(Spearman, Quadratic) {
    std::mt19937 rng(123457);
    std::uniform_real_distribution<> dist(1, 10000);
    std::uniform_int_distribution<> sz_dist(5, 10);
    double coef = dist(rng);
    for (uint32_t trial = 0; trial < 10; ++trial) {
        size_t size = sz_dist(rng);
        std::vector<double> a;
        std::vector<double> b;
        for (uint32_t i = 0; i < size; ++i) {
            size_t v = dist(rng);
            a.push_back(v);
            b.push_back(v * v);
        }
        ASSERT_EQ(1, spearman(a, b)) << "Trial " << trial << " Coef: " << coef;
    }
}

TEST(Spearman, QuadraticInverse) {
    size_t size = 10;
    std::vector<double> a(size);
    std::vector<double> b(size);
    for (uint32_t i = 0; i < size; ++i) {
        a[i] = 2 * i * i + 5;
        b[size - i - 1] = 2 * i * i + 5;
    }
    ASSERT_EQ(-1, spearman(a, b));
}

TEST(Spearman, AllIdentical) {
    size_t size = 10;
    std::vector<double> a(size);
    std::vector<double> b(size);
    for (uint32_t i = 0; i < size; ++i) {
        a[i] = 2 * i * i + 5;
        b[i] = 2 * i * i + 5;
    }
    ASSERT_EQ(1, spearman(a, b));
}

TEST(Spearman, SomeValues) {
    std::vector<double> a = { 35, 23, 47, 17, 10, 43, 9, 6, 28 };
    std::vector<double> b = { 30, 33, 45, 23, 8, 49, 12, 4, 31 };
    ASSERT_EQ(0.9, spearman(a, b));
}

TEST(Spearman, LinearRepeats) {
    std::mt19937 rng(123457);
    std::uniform_real_distribution<> dist(1, 10000);
    std::uniform_int_distribution<> sz_dist(5, 10);
    double coef = dist(rng);
    for (uint32_t trial = 0; trial < 10; ++trial) {
        size_t size = sz_dist(rng);
        std::vector<double> a;
        std::vector<double> b;
        for (uint32_t i = 0; i < size; ++i) {
            size_t v = dist(rng);
            a.push_back(v);
            a.push_back(v);
            b.push_back(v * coef);
            b.push_back(v * coef);
        }
        ASSERT_EQ(1, spearman(a, b)) << "Trial " << trial << " Coef: " << coef;
    }
}

TEST(Spearman, Rankify) {
    std::vector<double> a = { 1, 1, 2, 2, 3, 3, 4, 5, 5 };
    std::vector<double> expected_ranks { 1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7, 8.5, 8.5 };
    ASSERT_EQ(expected_ranks, rankify(a));
}

TEST(Spearman, SomeValuesRepeats) {
    std::vector<double> a = { 1, 1, 2, 2, 3, 3, 4, 5, 5 };
    std::vector<double> b = { 7, 8, 8, 19, 19, 3, 3, 5, 9 };

    ASSERT_NEAR(-0.19314, spearman(a, b), 1e-5);
}


} // namespace
