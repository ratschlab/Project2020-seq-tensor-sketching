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

    Vec2D<T> compute(const Seq<T> &kmers) {
        Vec2D<T> sketch;
        if (kmers.size() < tup_len) {
            throw std::invalid_argument("Sequence must be longer than tuple length");
        }
        if (kmers.size() > max_len) {
            throw std::invalid_argument("Sequence too long. Maximum sequence length is "
                                        + std::to_string(max_len)
                                        + ". Set --max_length to a higher value.");
        }
        for (size_t pi = 0; pi < this->sketch_dim; pi++) {
            Vec<size_t> counts(this->set_size, 0);
            Vec<std::pair<T, T>> ranks;
            for (auto s : kmers) {
                ranks.push_back({ this->hashes[pi][s + this->set_size * counts[s]], s });
                counts[s]++;
            }
            std::sort(ranks.begin(), ranks.end());
            Vec<T> tup;
            for (auto pair = ranks.begin(); pair != ranks.end() && pair != ranks.begin() + tup_len;
                 pair++) {
                tup.push_back(pair->second);
            }
            sketch.push_back(tup);
        }
        return sketch;
    }

    Vec<T> compute_flat(const Seq<T> &kmers) {
        Vec<T> sketch;
        Timer::start("ordered_minhash_flat");
        Vec2D<T> sketch2D = compute<T>(kmers);
        for (const auto &tuple : sketch2D) {
            T sum = 0;
            for (const auto &item : tuple) {
                sum = sum * this->set_size + item; // TODO: deal with overflows
            }
            sketch.push_back(sum);
        }
        Timer::stop();

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
    Vec2D<T> compute(const std::vector<C> &sequence, uint32_t k, uint32_t alphabet_size) {
        Timer::start("compute_sequence");
        Vec<T> kmers = seq2kmer<C, T>(sequence, k, alphabet_size);
        Vec2D<T> sketch = compute(kmers);
        Timer::stop();
        return sketch;
    }

  private:
    size_t max_len;
    size_t tup_len;
};

} // namespace ts
