#pragma once

#include "util/timer.hpp"
#include "util/utils.hpp"

#include <iostream>
#include <random>

namespace ts { // ts = Tensor Sketch

/**
 * Naive implementation of weighted min-hash sketching. For more efficient implementations, see
 * https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36928.pdf and
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2010/06/ConsistentWeightedSampling2.pdf
 *
 * Given a set S, and a sequence s=s1...sn with elements from S, this class computes a vector
 * {hmin_1(s), hmin_2(s), ..., hmin_sketch_size(s)}, where hmin_k(s)=s_i, such that h_k(s_i, #s_i)
 * is the smallest of h_k(s_1, 1..#s_1), h_k(s_2, 1..#s_2), ..., h_k(s_n, 1..#s_n) and
 * h_k:Sx{1..n} -> {1..#set_size} is a random permuation of the elements in S and #s_i denotes the
 * number of occurences of s_i in the sequence s.
 * @tparam T the type of S's elements
 */
template <class T>
class WeightedMinHash {
  public:
    /**
     * Constructs a weighted min-hasher for the given alphabet size which constructs sketches of the
     * given set size, dimension and maximum length.
     * @param set_size the number of elements in S,
     * @param sketch_dim the number of components (elements) in the sketch vector.
     * @param max_len maximum sequence length to be hashed.
     */
    WeightedMinHash(T set_size, size_t sketch_dim, size_t max_len)
        : set_size(set_size), sketch_dim(sketch_dim), max_len(max_len) {
        rand_init();
    }

    template <class embed_type>
    Vec<embed_type> compute(const std::vector<T> &seq) {
        if (seq.size() > max_len) {
            std::cerr << "Sequence too long. Maximum sequence length is " << max_len
                      << ". Set --max_length to a higher value." << std::endl;
        }
        Timer::start("weighted_minhash");
        Vec<embed_type> sketch = Vec<embed_type>(sketch_dim);
        for (size_t si = 0; si < sketch_dim; si++) {
            T min_char;
            size_t min_rank = hashes[0].size() + 1; // set_size * max_len + 1
            Vec<size_t> cnts(set_size, 0);
            for (const auto s : seq) {
                auto r = hashes[si][s + cnts[s] * set_size];
                cnts[s]++;
                if (r < min_rank) {
                    min_rank = r;
                    min_char = s;
                }
            }
            sketch[si] = min_char;
        }
        Timer::stop();

        return  sketch;
    }

  private:
    void rand_init() {
        std::random_device rd;
        auto eng = std::mt19937(rd());
        hashes = Vec2D<size_t>(sketch_dim, Vec<size_t>(set_size * max_len, 0));
        for (size_t m = 0; m < sketch_dim; m++) {
            std::iota(hashes[m].begin(), hashes[m].end(), 0);
            std::shuffle(hashes[m].begin(), hashes[m].end(), eng);
        }
    }

  private:
    T set_size;
    size_t sketch_dim;
    size_t max_len;
    Vec2D<size_t> hashes;
};

} // namespace ts
