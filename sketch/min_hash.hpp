#pragma once

#include "util/timer.hpp"
#include "util/utils.hpp"

#include <cstdint>
#include <random>

namespace ts { // ts = Tensor Sketch

/**
 * Implements min-hash-based sketching, as described in https://en.wikipedia.org/wiki/MinHash.
 * Given a set S, and a sequence s=s1...sn with elements from S, this class computes a vector
 * {hmin_1(s), hmin_2(s), ..., hmin_sketch_size(s)}, where hmin_k(s)=s_i, such that h_k(s_i) is the
 * smallest of h_k(s_1), h_k(s_2), ..., h_k(s_n) and h_k:S->{1..#set_size} is a random permuation of
 * the elements in S.
 * @tparam T the type of S's elements
 */
template <class T>
class MinHash {
  public:
    /**
     * Constructs a min-hasher for the given alphabet size which constructs sketches of the set size
     * and sketch dimension.
     * @param set_size the number of elements in S,
     * @param sketch_dim the number of components (elements) in the sketch vector.
     */
    MinHash(T set_size, size_t sketch_dim) : set_size(set_size), sketch_dim(sketch_dim) {
        rand_init();
    }

    /**
     * Computes the min-hash sketch for the given sequence.
     * @param seq the sequence to compute the min-hash for
     * @return the min-hash sketch of seq
     */
    template <class embed_type>
    Vec<embed_type> compute(const std::vector<T> &seq) {
        Vec<embed_type> sketch;
        Timer::start("minhash");
        sketch = Vec<embed_type>(sketch_dim);
        for (size_t si = 0; si < sketch_dim; si++) {
            T min_char;
            size_t min_rank = set_size + 1;
            Vec<size_t> h = hashes[si];
            for (auto s : seq) {
                if (h[s] < min_rank) {
                    min_rank = h[s];
                    min_char = s;
                }
            }
            sketch[si] = min_char;
        }
        Timer::stop();
        return sketch;
    }

  private:
    void rand_init() {
        std::random_device rd;
        auto eng = std::mt19937(rd());
        hashes = Vec2D<size_t>(sketch_dim, Vec<size_t>(set_size, 0));
        for (size_t m = 0; m < sketch_dim; m++) {
            std::iota(hashes[m].begin(), hashes[m].end(), 0);
            std::shuffle(hashes[m].begin(), hashes[m].end(), eng);
        }
    }

  private:
    T set_size;
    size_t sketch_dim;
    /** Contains the sketch_dim permutations (hashes) that are used to compute the min-hash */
    Vec2D<size_t> hashes;
};
} // namespace ts
