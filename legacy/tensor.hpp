#pragma once

#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

#include "util/multivec.hpp"

namespace ts { // ts = Tensor Sketch


/**
 * Computes tensor sketches for a given sequence as described in
 * https://www.biorxiv.org/content/10.1101/2020.11.13.381814v1
 * @tparam set_type the type of the characters in sketched sequences
 * @tparam sketch_type the type of elements in the sketch
 */
template <class set_type, class sketch_type>
class Tensor {
  public:
    /**
     * @param alphabet_size the size of the alphabet over which the sequences to be sketched are
     * defined
     * @param sketch_count number of different sketches to compute
     * @param embedded_dim the dimension of the embedded (sketched) space, denoted by D in the paper
     * @param num_bins number of bins for discretization of the sketches.
     * @param tup_len the length of the subsequences considered for sketching, denoted by t in the
     * paper
     */
    Tensor(set_type alphabet_size,
           size_t sketch_count,
           size_t embedded_dim,
           size_t num_bins,
           size_t tup_len)
        : sketch_count(sketch_count),
          alphabet_size(alphabet_size),
          embedded_dim(embedded_dim),
          num_bins(num_bins),
          tup_len(tup_len) {
        rand_init();
    }

    std::vector<sketch_type> compute(const std::vector<set_type> &sequence) {
        Timer::start("tensor_sketch");

        std::vector<sketch_type> sketch(sketch_count, 0);
        for (size_t m = 0; m < sketch_count; m++) {
            auto cnt = new2D<double>(tup_len + 1, embedded_dim, sketch_type(0));
            cnt[0][0] = 1; // base case
            for (size_t i = 0; i < sequence.size(); i++) {
                for (int32_t t = (int32_t)tup_len - 1; t >= 0; t--) {
                    auto pi = hashes[m][t][sequence[i]];
                    for (size_t p = 0; p < embedded_dim; p++) {
                        auto shift = (p + pi) % embedded_dim;
                        cnt[t + 1][shift] += cnt[t][p];
                    }
                }
            }
            const auto &top_cnt = cnt[tup_len]; // this is T^p
            auto prod = std::inner_product(s[m].begin(), s[m].end(), top_cnt.begin(), 0.0);
            prod /= l1(top_cnt); // this is the total no of sequences
            if (num_bins == 0) {
                sketch[m] = prod;
            } else {
                sketch_type bin = std::upper_bound(bins.begin(), bins.begin() + num_bins, prod)
                        - bins.begin();
                sketch[m] = bin;
            }
        }
        Timer::stop();

        return sketch;
    }

  protected:
    void rand_init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> unif_hash(0, embedded_dim - 1);

        hashes = new3D<set_type>(sketch_count, tup_len, alphabet_size);
        s = new2D<int8_t>(sketch_count, embedded_dim);
        for (size_t m = 0; m < sketch_count; m++) {
            for (size_t t = 0; t < tup_len; t++) {
                for (size_t c = 0; c < alphabet_size; c++) {
                    hashes[m][t][c] = unif_hash(gen);
                }
            }
            for (size_t p = 0; p < embedded_dim; p++) {
                s[m][p] = (p % 2 == 0) ? 1 : -1; // use oddity of p to assign (-1) or (1)
            }
        }
        bins = std::vector<double>(num_bins);
        for (size_t b = 0; b < num_bins; b++) {
            bins[b] = std::tan(M_PI * ((b + .5) / num_bins - .5));
        }
        bins.push_back(std::numeric_limits<double>::max());
        bins.insert(bins.begin(), std::numeric_limits<double>::lowest());
    }

  protected:
    size_t sketch_count;
    set_type alphabet_size;
    size_t embedded_dim;
    size_t num_bins;
    size_t tup_len;

    /**
     * Denotes the hash functions h1,....hD:A->{1....D}.
     */
    Vec3D<set_type> hashes;
    /**
     * Sign function, corresponds to s1,s2,...st:A->{-1,1} in the paper. The first index denotes the
     * sketch count, the second index the embedded dimension
     */
     //TODO: figure out why second index is not the tuple and why there is no 3rd index.
    Vec2D<int8_t> s;

    /** Bins the possible values of a sketch into #num_bins integer values */
    std::vector<double> bins;
};

} // namespace ts
