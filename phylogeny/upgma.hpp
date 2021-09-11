#pragma once

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ts {

/**
 * Node in the phylogeny node.
 */
struct Node {
    double age;
    uint32_t left, right;
};

constexpr uint32_t NO_CHILD = std::numeric_limits<uint32_t>::max();

/**
 * To avoid dynamic memory allocation, the tree is represented as a map from node id to the actual
 * Node. The root is at index size()-1
 */
using Tree = std::vector<Node>;

/**
 * Runs UPGMA (Unweighted Pair Group Method with Arithmetic Mean Algorithm) on the given distance
 * matrix and returns the reconstructed phylogeny graph as a (parent->children) map.
 */
Tree upgma(const std::vector<std::vector<double>> &dist_mat);

} // namespace ts
