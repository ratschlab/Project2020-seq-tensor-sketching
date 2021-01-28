#include "upgma.hpp"

#include <unordered_set>
#include <vector>

namespace ts {

Tree upgma(const std::vector<std::vector<double>> &dist_mat) {
    if (dist_mat.empty()) {
        return {};
    }

    Tree result(2 * dist_mat.size() - 1);
    // {nodeId, nodeCount} pairs of all the cluster roots
    std::unordered_map<uint32_t, uint32_t> roots;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, double>> D;
    for (uint32_t i = 0; i < dist_mat.size(); ++i) {
        roots.insert({ i, 1 });
        result[i] = { 0, NO_CHILD, NO_CHILD };
        for (uint32_t j = 0; j < dist_mat.size(); ++j) {
            D[i][j] = dist_mat[i][j];
        }
    }
    for (uint32_t step = 0; step < dist_mat.size() - 1; ++step) {
        double minDist = std::numeric_limits<double>::max();
        uint32_t min_i, min_j;
        for (const auto &root1 : roots) {
            for (const auto &root2 : roots) {
                if (root1.first == root2.first) {
                    continue;
                }
                double currentDist = D[root1.first][root2.first];
                if (currentDist < minDist) {
                    minDist = currentDist;
                    min_i = root1.first;
                    min_j = root2.first;
                }
            }
        }
        uint32_t new_node = dist_mat.size() + step;

        result[new_node] = { minDist / 2., min_i, min_j };
        // update D
        for (const auto &root : roots) {
            D[new_node][root.first]
                    = (D[min_i][root.first] * roots[min_i] + D[min_j][root.first] * roots[min_j])
                    / (roots[min_i] + roots[min_j]);
            D[root.first][new_node] = D[new_node][root.first];
        }
        D.erase(min_i);
        D.erase(min_j);
        for (auto &row : D) {
            row.second.erase(min_i);
            row.second.erase(min_j);
        }
        roots[new_node] = roots[min_i] + roots[min_j];
        roots.erase(min_i);
        roots.erase(min_j);
    }
    return result;
}

} // namespace ts
