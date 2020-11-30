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
 * @tparam T the type of elements in the sketch and in the sequences to be sketched.
 */
template <class T>
class Tensor {
  public:
    /**
     * @param set_size the number of elements in S
     * @param sketch_dim the number of components (elements) in the sketch vector
     * @param num_phases number of phases to be used for modular arithmetic
     * @param num_bins number of bins for discretization
     * @param tup_len the length of the subsequences considered for sketching (denoted by t in the
     * paper).
     */
    Tensor(T set_size, size_t sketch_dim, size_t num_phases, size_t num_bins, size_t tup_len)
        : sketch_dim(sketch_dim),
          set_size(set_size),
          num_phases(num_phases),
          num_bins(num_bins),
          tup_len(tup_len) {
        rand_init();
    }

    Vec<T> compute(const std::vector<T> &kmers) {
        Vec<T> sketch;
        Timer::start("tensor_sketch");
        sketch = Vec<T>(sketch_dim, 0);
        for (size_t m = 0; m < sketch_dim; m++) {
            auto cnt = new2D<double>(tup_len + 1, num_phases, 0);
            cnt[0][0] = 1; // base case
            for (size_t i = 0; i < kmers.size(); i++) {
                for (int32_t t = (int32_t)tup_len - 1; t >= 0; t--) {
                    auto pi = iphase[m][t][kmers[i]];
                    for (size_t p = 0; p < num_phases; p++) {
                        auto shift = (p + pi) % num_phases;
                        cnt[t + 1][shift] += cnt[t][p];
                    }
                }
            }
            const auto &top_cnt = cnt[tup_len];
            auto prod = std::inner_product(icdf[m].begin(), icdf[m].end(), top_cnt.begin(),
                                           (double)0);
            double norm = l1(top_cnt);
            prod = prod / norm;
            T bin = std::upper_bound(bins.begin(), bins.begin() + num_bins, prod) - bins.begin();
            if (num_bins == 0) {
                sketch[m] = prod;
            } else {
                sketch[m] = bin;
            }
        }
        Timer::stop();

        return sketch;
    }

  protected:
    virtual void rand_init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> unif_iphase(0, num_phases - 1);
        std::uniform_real_distribution<double> unif(0, 1);

        iphase = new3D<int>(sketch_dim, tup_len, set_size);
        icdf = new2D<double>(sketch_dim, num_phases);
        for (size_t m = 0; m < sketch_dim; m++) {
            double bias = 0;
            for (size_t t = 0; t < tup_len; t++) {
                bias += unif(gen);
                for (size_t c = 0; c < set_size; c++) {
                    iphase[m][t][c] = unif_iphase(gen);
                }
            }
            for (size_t p = 0; p < num_phases; p++) {
                icdf[m][p] = 1 - 2 * unif(gen); // use random sign (-1) or (1)
                icdf[m][p] = (p % 2 == 0) ? 1 : -1; // use oddity of p to assign (-1) or (1)
            }
        }
        bins = Vec<double>(num_bins);
        for (size_t b = 0; b < num_bins; b++) {
            bins[b] = std::tan(M_PI * (((double)b + .5) / num_bins - .5));
        }
        bins.push_back(std::numeric_limits<double>::max());
        bins.insert(bins.begin(), -std::numeric_limits<double>::lowest());
    }

  protected:
    size_t sketch_dim;
    size_t set_size;
    size_t num_phases;
    size_t num_bins;
    size_t tup_len;

    Vec3D<int> iphase;
    Vec2D<double> icdf;
    Vec<double> bins;
};

} // namespace ts
