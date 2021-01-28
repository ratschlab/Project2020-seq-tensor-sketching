#include "phylogeny/upgma.hpp"

#include <gtest/gtest.h>

namespace {

using namespace ts;

TEST(upgma, empty) {
    ASSERT_TRUE(upgma({}).empty());
}

TEST(upgma, one) {
    Tree graph = upgma({ { 11.5 } });
    ASSERT_TRUE(graph.size() == 1);
    ASSERT_TRUE(graph[0].age == 0);
    ASSERT_TRUE(graph[0].left == NO_CHILD);
    ASSERT_TRUE(graph[0].right == NO_CHILD);
}

TEST(upgma, some_values) {
    std::vector<std::vector<double>> dist_mat = { { 0, 17, 21, 31, 23 },
                                                  { 17, 0, 30, 34, 21 },
                                                  { 21, 30, 0, 28, 39 },
                                                  { 31, 34, 28, 0, 43 },
                                                  { 23, 21, 39, 43, 0 } };
    Tree graph = upgma(dist_mat);
    ASSERT_EQ(9, graph.size());
    ASSERT_EQ(graph[8].age, 16.5);
    ASSERT_EQ(graph[7].age, 14);
    ASSERT_EQ(graph[6].age, 11);
    ASSERT_EQ(graph[5].age, 8.5);
    for (uint32_t i = 0; i < 5; ++i) {
        ASSERT_EQ(graph[i].age, 0);
    }
}

} // namespace
