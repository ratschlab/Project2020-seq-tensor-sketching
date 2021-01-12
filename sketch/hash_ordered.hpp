#pragma once

#include "hash_base.hpp"

#include "util/utils.hpp"

#include <iostream>
#include <random>

namespace ts { // ts = Tensor Sketch

/**
 * Naive implementation of the Ordered MinHash sketching method described in:
 * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6612865/
 *
 * @tparam T the type of element in the sequences to be sketched
 */
template <class T>
class OrderedMinHash : public HashBase<T> {
  public:
    OrderedMinHash() {}
    /**
     * @param set_size the number of elements in S
     * @param sketch_dim the number of components (elements) in the sketch vector.
     * @param max_len maximum sequence length to be hashed.
     * @param tup_len the sketching will select the tup_len lowest values for each hash function
     */
    OrderedMinHash(T set_size, size_t sketch_dim, size_t max_len, size_t tup_len)
        : HashBase<T>(set_size, sketch_dim, set_size * max_len),
          max_len(max_len),
          tup_len(tup_len) {}

    Vec2D<T> compute(const std::vector<T> &kmers) {
        Vec2D<T> sketch(this->sketch_dim);
        if (kmers.size() < tup_len) {
            throw std::invalid_argument("Sequence of kmers must be longer than tuple length");
        }
        for (size_t pi = 0; pi < this->sketch_dim; pi++) {
            std::unordered_map<size_t, uint32_t> counts;
            std::vector<std::pair<T, T>> ranks;
            for (auto s : kmers) {
                ranks.push_back({ this->hash(pi, s + this->set_size * counts[s]), s });
                counts[s]++;
#ifndef NDEBUG
                assert(counts[s] != 0); // no overflow
                if (counts[s] > max_len) {
                    throw std::invalid_argument("Kmer  " + std::to_string(s) + " repeats more than "
                                                + std::to_string(max_len)
                                                + " times. Set --max_len to a higher value.");
                }
#endif
            }
            std::sort(ranks.begin(), ranks.end());
            std::vector<T> tup;
            for (auto pair = ranks.begin(); pair != ranks.end() && pair != ranks.begin() + tup_len;
                 pair++) {
                tup.push_back(pair->second);
            }
            sketch[pi] = tup;
        }
        return sketch;
    }

    std::vector<T> compute_flat(const std::vector<T> &kmers) {
        Timer timer("ordered_minhash");
        std::vector<T> sketch;

        Vec2D<T> sketch2D = compute(kmers);
        for (const auto &tuple : sketch2D) {
            T sum = 0;
            for (const auto &item : tuple) {
                sum = sum * this->set_size + item; // TODO: deal with overflows
            }
            sketch.push_back(sum);
        }

        return sketch;
    }

    /**
     * Computes the ordered min-hash sketch for the given sequence.
     * @param sequence the sequence to compute the ordered min-hash for
     * @param k-mer length; the sequence will be transformed into k-mers and the k-mers will be
     * hashed
     * @param number of characters in the alphabet over which sequence is defined
     * @return the ordered min-hash sketch of sequence
     * @tparam C the type of characters in the sequence
     */
    template <typename C>
    void compute(Vec2D<T> &sketch, const std::vector<C> &sequence, uint32_t k, uint32_t alphabet_size) {
        std::vector<T> kmers = seq2kmer<C, T>(sequence, k, alphabet_size);
        sketch = compute(kmers);
    }

    static T dist(const std::vector<T> &a, const std::vector<T> &b) {
        Timer timer("ordered_minhash_dist");
        return hamming_dist(a, b);
    }

  private:
    size_t max_len;
    size_t tup_len;
};

} // namespace ts
