#pragma once

#include "util/timer.hpp"
#include "util/utils.hpp"

#include <random>
#include <unordered_map>
#include <unordered_set>

namespace ts {
template <typename T>
class HashBase {
  public:
    HashBase() {}

    HashBase(T set_size, size_t sketch_dim, size_t hash_size)
        : set_size(set_size),
          sketch_dim(sketch_dim),
          hash_size(hash_size),
          hashes(std::vector<std::unordered_map<T, T>>(sketch_dim)),
          values(std::vector<std::unordered_set<T>>(sketch_dim)) {
        std::random_device rd;
        rng = std::mt19937(rd());
        rand = std::uniform_int_distribution<T>(0, hash_size - 1);
    }

    void set_hashes_for_testing(const std::vector<std::unordered_map<T, T>> &h) { hashes = h; }

  protected:
    T set_size;
    size_t sketch_dim;
    size_t hash_size;

    /**
     * Returns the hash value for the index-th hash function.
     * Since the Hashes are generated on demand.
     */
    T hash(size_t index, size_t value) {
        if (hashes[index].contains(value)) {
            return hashes[index][value];
        }
        while (true) {
            T next_value = rand(rng);
            if (!values[index].contains(next_value)) {
                hashes[index][value] = next_value;
                values[index].insert(next_value);
                return next_value;
            }
        }
    }

  private:
    /** Contains the sketch_dim permutations (hashes) that are used to compute the min-hash */
    std::vector<std::unordered_map<T, T>> hashes;
    /** Contains the values used so far for each on-demand permutation */
    std::vector<std::unordered_set<T>> values;
    std::uniform_int_distribution<T> rand;
    std::mt19937 rng;
};

} // namespace ts
