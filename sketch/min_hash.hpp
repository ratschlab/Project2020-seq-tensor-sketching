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
 * This class assumes that S= {0,1,2....,#set_size}.
 * @tparam T the type of S's elements.
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
     * Computes the min-hash sketch for the given kmers.
     * @param kmers kmers extracted from a sequence
     * @return the min-hash sketch of #kmers
     */
    Vec<T> compute(const std::vector<T> &kmers) {
        Vec<T> sketch;
        Timer::start("minhash");
        sketch = Vec<T>(sketch_dim);
        for (size_t si = 0; si < sketch_dim; si++) {
            T min_char;
            size_t min_rank = set_size + 1;
            Vec<T> h = hashes[si];
            for (auto s : kmers) {
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

    /**
     * Computes the min-hash sketch for the given sequence.
     * @param sequence the sequence to compute the min-hash for
     * @param k-mer length; the sequence will be transformed into k-mers and the k-mers will be
     * hashed
     * @param number of characters in the alphabet over which sequence is defined
     * @return the min-hash sketch of sequence
     * @tparam C the type of characters in the sequence
     */
    template <typename C>
    Vec<T> compute(const std::vector<C> &sequence, uint32_t k, uint32_t alphabet_size) {
        Timer::start("compute_sequence");
        Vec<T> kmers = seq2kmer<C, T>(sequence, k, alphabet_size);
        Vec<T> sketch;
        sketch = Vec<T>(sketch_dim);
        for (size_t si = 0; si < sketch_dim; si++) {
            T min_char;
            size_t min_rank = set_size + 1;
            Vec<T> h = hashes[si];
            for (auto kmer : kmers) {
                if (h[kmer] < min_rank) {
                    min_rank = h[kmer];
                    min_char = kmer;
                }
            }
            sketch[si] = min_char;
        }
        Timer::stop();
        return sketch;
    }

    void set_hashes_for_testing(const Vec2D<T> &hashes) { this->hashes = hashes; }

  private:
    void rand_init() {
        std::random_device rd;
        auto eng = std::mt19937(rd());
        hashes = Vec2D<T>(sketch_dim, Vec<T>(set_size, T(0)));
        for (size_t m = 0; m < sketch_dim; m++) {
            std::iota(hashes[m].begin(), hashes[m].end(), T(0));
            std::shuffle(hashes[m].begin(), hashes[m].end(), eng);
        }
    }

  private:
    T set_size;
    size_t sketch_dim;
    /** Contains the sketch_dim permutations (hashes) that are used to compute the min-hash */
    Vec2D<T> hashes;
};
} // namespace ts
