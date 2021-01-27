#pragma once

#include <unordered_map>
#include <vector>

struct Node {
    double age;
    uint32_t left, right;
};

constexpr uint32_t NO_CHILD = std::numeric_limits<uint32_t>::max();

// map from node id to the actual Node. The root is at index size()-1
using Tree = std::unordered_map<uint32_t, Node>;

/**
 * Runs UPGMA (Unweighted Pair Group Method with Arithmetic Mean Algorithm) on the given distance
 * matrix and returns the reconstructed phylogeny graph as a (parent->children) map.
 */
Tree upgma(const std::vector<std::vector<double>> &dist_mat);
