#pragma once

#include "util/utils.hpp"

namespace ts { // ts = Tensor Sketch

/**
 * Naive implementation of the Ordered MinHash sketching method described in:
 * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6612865/
 *
 * @tparam T the type of element in the sequences to be sketched
 */
template <class T>
class OrderedMinHash {
  public:
    /**
     * @param set_size the number of elements in S
     * @param sketch_dim the number of components (elements) in the sketch vector.
     * @param max_len maximum sequence length to be hashed.
     * @param tup_len the sketching will select the tup_len lowest values for each hash function
     */
    OrderedMinHash(size_t set_size, size_t sketch_dim, size_t max_len, size_t tup_len)
        : set_size(set_size), sketch_dim(sketch_dim), max_len(max_len), tup_len(tup_len) {
        rand_init();
    }

    template <class embed_type>
    Vec2D<embed_type> compute(const Seq<T> &seq) {
        Vec2D<embed_type> sketch;
        if (seq.size() > max_len) {
            std::cerr << "Sequence too long. Maximum sequence length is " << max_len
                      << ". Set --max_length to a higher value." << std::endl;
        }
        for (size_t pi = 0; pi < sketch_dim; pi++) {
            Vec<size_t> counts(set_size, 0);
            Vec<std::pair<embed_type, T>> ranks;
            for (auto s : seq) {
                ranks.push_back({ perms[pi][s + set_size * counts[s]], s });
                counts[s]++;
            }
            std::sort(ranks.begin(), ranks.end());
            Vec<embed_type> tup;
            for (auto pair = ranks.begin(); pair != ranks.begin() + tup_len; pair++) {
                tup.push_back(pair->second);
            }
            sketch.push_back(tup);
        }
        return sketch;
    }


    template <class embed_type>
    Vec<embed_type> compute_flat(const Seq<T> &seq) {
        Vec<embed_type> sketch;
        Timer::start("ordered_minhash_flat");
        Vec2D<embed_type> embed2D = compute<embed_type>(seq);
        for (const auto &tuple : embed2D) {
            int sum = 0;
            for (const auto &item : tuple) {
                sum = sum * set_size + item; // TODO: deal with overflows
            }
            sketch.push_back(sum);
        }
        Timer::stop();

        return sketch;
    }

  private:
    size_t set_size;
    size_t sketch_dim;
    size_t max_len;
    size_t tup_len;

    Vec2D<int> perms;

    void rand_init() {
        std::random_device rd;
        auto gen = std::mt19937(rd());

        int total_len = set_size * max_len;
        perms = Vec2D<int>(sketch_dim, Vec<int>(total_len, 0));
        for (size_t pi = 0; pi < sketch_dim; pi++) {
            std::iota(perms[pi].begin(), perms[pi].end(), 0);
            std::shuffle(perms[pi].begin(), perms[pi].end(), gen);
        }
    }
};

} // namespace ts
