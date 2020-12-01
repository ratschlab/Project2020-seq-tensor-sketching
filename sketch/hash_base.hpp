#pragma once

#include "util/timer.hpp"
#include "util/utils.hpp"

#include <random>

namespace ts {
template <typename T>
class HashBase {
  public:
    HashBase() {}

    HashBase(T set_size, size_t sketch_dim, size_t hash_size)
        : set_size(set_size), sketch_dim(sketch_dim) {
        rand_init(hash_size);
    }

    void set_hashes_for_testing(const Vec2D<T> &hashes) { this->hashes = hashes; }

  protected:
    void rand_init(size_t hash_size) {
        std::random_device rd;
        auto eng = std::mt19937(rd());
        hashes = Vec2D<T>(sketch_dim, Vec<T>(hash_size, T(0)));
        for (size_t m = 0; m < sketch_dim; m++) {
            std::iota(hashes[m].begin(), hashes[m].end(), T(0));
            std::shuffle(hashes[m].begin(), hashes[m].end(), eng);
        }
    }

    T set_size;
    size_t sketch_dim;
    /** Contains the sketch_dim permutations (hashes) that are used to compute the min-hash */
    Vec2D<T> hashes;
};

} // namespace ts
