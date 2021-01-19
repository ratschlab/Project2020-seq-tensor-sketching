#pragma once

#include "immintrin.h" // for AVX
#include "nmmintrin.h" // for SSE4.2
#include "sketch//sketch_base.hpp"
#include "util/multivec.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <set>

namespace ts { // ts = Tensor Sketch

/**
 * Computes tensor sketches for a given sequence as described in
 * https://www.biorxiv.org/content/10.1101/2020.11.13.381814v1
 * @tparam seq_type the type of elements in the sequences to be sketched.
 */
template <class seq_type>
class Tensor : public SketchBase<std::vector<double>, false> {
  public:
    // Tensor sketch output should be transformed if the command line flag is set.
    constexpr static bool transform_sketches = false;

    /**
     * @param alphabet_size the number of elements in the alphabet S over which sequences are
     * defined (e.g. 4 for DNA)
     * @param sketch_dim the dimension of the embedded (sketched) space, denoted by D in the paper
     * @param subsequence_len the length of the subsequences considered for sketching, denoted by t
     * in the paper
     * @param seed the seed to initialize the random number generator used for the random hash
     * functions.
     */
    Tensor(seq_type alphabet_size,
           size_t sketch_dim,
           size_t subsequence_len,
           uint32_t seed,
           const std::string &name = "TS",
           double gap_penalty = 0,
           bool use_permutation = false,
           bool injective_hash = false)
        : SketchBase<std::vector<double>, false>(name),
          alphabet_size(alphabet_size),
          sketch_dim(sketch_dim),
          subsequence_len(subsequence_len),
          gap_penalty(gap_penalty),
          rng(seed),
          injective_hash(injective_hash),
          use_permutation(use_permutation) {
        init();
    }

    void init() {
        hashes = new2D<seq_type>(subsequence_len, alphabet_size);
        signs = new2D<bool>(subsequence_len, alphabet_size);
        permutation_hashes = new3D<seq_type>(subsequence_len, alphabet_size, sketch_dim);

        std::uniform_int_distribution<seq_type> rand_hash2(0, sketch_dim - 1);
        std::uniform_int_distribution<seq_type> rand_bool(0, 1);

        if (use_permutation) {
            for (size_t h = 0; h < subsequence_len; h++) {
                for (size_t c = 0; c < alphabet_size; c++) {
                    auto &p = permutation_hashes[h][c];
                    std::iota(begin(p), end(p), 0);
                    std::shuffle(begin(p), end(p), rng);
                    signs[h][c] = rand_bool(rng);
                }
            }
        } else if (injective_hash) {
            // Make sure that there are plenty of dimensions to find an injective hash function.
            assert(alphabet_size <= sketch_dim / 2);

            for (size_t h = 0; h < subsequence_len; h++) {
                std::set<int> done;
                for (size_t c = 0; c < alphabet_size; c++) {
                    int r;
                    do {
                        r = rand_hash2(rng);
                    } while (not done.insert(r).second);
                    hashes[h][c] = r;
                    signs[h][c] = rand_bool(rng);
                }
            }

        } else {
            for (size_t h = 0; h < subsequence_len; h++) {
                for (size_t c = 0; c < alphabet_size; c++) {
                    hashes[h][c] = rand_hash2(rng);
                    signs[h][c] = rand_bool(rng);
                }
            }
        }
    }

    /**
     * Computes the sketch of the given sequence.
     * @param seq the sequence to be sketched
     * @return an array of size #sketch_dim containing the sequence's sketch
     */
    std::vector<double> compute(const std::vector<seq_type> &seq) {
        Timer timer("tensor_sketch");
        // Tp corresponds to T+, Tm to T- in the paper; Tp[0], Tm[0] are sentinels and contain the
        // initial condition for empty strings; Tp[p], Tm[p] represent the partial sketch when
        // considering hashes h1...hp, over the prefix x1...xi. The final result is then
        // Tp[t]-Tm[t], where t is #sequence_len
        auto Tp = new2D<double>(subsequence_len + 1, sketch_dim, 0);
        auto Tm = new2D<double>(subsequence_len + 1, sketch_dim, 0);

        // the initial condition states that the sketch for the empty string is (1,0,..)
        Tp[0][0] = 1;
        for (uint32_t i = 0; i < seq.size(); i++) {
            // must traverse in reverse order, to avoid overwriting the values of Tp and Tm before
            // they are used in the recurrence
            for (uint32_t p = std::min(i + 1, (uint32_t)subsequence_len); p >= 1; --p) {
                double z = p / (i + 1.0); // probability that the last index is i
                seq_type r = hashes[p - 1][seq[i]];
                // std::cerr << "Index: " << p-1 << " " << int(seq[i]) << std::endl;
                const auto &perm = permutation_hashes[p - 1][seq[i]];
                bool s = signs[p - 1][seq[i]];
                double gp = p == subsequence_len ? 0 : gap_penalty;
                if (use_permutation) {
                    if (s) {
                        Tp[p] = perm_sum(Tp[p], Tp[p - 1], perm, z, gp);
                        Tm[p] = perm_sum(Tm[p], Tm[p - 1], perm, z, gp);
                    } else {
                        Tp[p] = perm_sum(Tp[p], Tm[p - 1], perm, z, gp);
                        Tm[p] = perm_sum(Tm[p], Tp[p - 1], perm, z, gp);
                    }
                } else {
                    if (s) {
                        Tp[p] = shift_sum(Tp[p], Tp[p - 1], r, z, gp);
                        Tm[p] = shift_sum(Tm[p], Tm[p - 1], r, z, gp);
                    } else {
                        Tp[p] = shift_sum(Tp[p], Tm[p - 1], r, z, gp);
                        Tm[p] = shift_sum(Tm[p], Tp[p - 1], r, z, gp);
                    }
                }
            }
        }
        std::vector<double> sketch(sketch_dim, 0);
        for (uint32_t m = 0; m < sketch_dim; m++) {
            sketch[m] = Tp[subsequence_len][m] - Tm[subsequence_len][m];
        }

        return sketch;
    }

    /** Sets the hash and sign functions to predetermined values for testing */
    void set_hashes_for_testing(const Vec2D<seq_type> &h, const Vec2D<bool> &s) {
        hashes = h;
        signs = s;
    }

    double dist(const std::vector<double> &a, const std::vector<double> &b) {
        Timer timer("tensor_sketch_dist");
        return l2_dist(a, b);
    }

  protected:
    /** Computes (1-z)*a + z*b_shift */
    inline std::vector<double> shift_sum(const std::vector<double> &a,
                                         const std::vector<double> &b,
                                         seq_type shift,
                                         double z,
                                         double gap_penalty = 0) {
        assert(a.size() == b.size());
        size_t len = a.size();
        std::vector<double> result(a.size());
        for (uint32_t i = 0; i < a.size(); i++) {
            result[i] = (1 - gap_penalty) * (1 - z) * a[i] + z * b[(len + i - shift) % len];
            //assert(result[i] <= 1 + 1e-5 && result[i] >= -1e-5);
        }
        return result;
    }

    /** Computes (1-z)*a + z*b_shift */
    void shift_sum_inplace(std::vector<double> &a,
                           const std::vector<double> &b,
                           seq_type shift,
                           double z,
                           double gap_penalty = 0) {
        assert(a.size() == b.size());
        size_t len = a.size();
        for (uint32_t i = 0; i < len; i++) {
            a[i] = (1 - gap_penalty) * (1 - z) * a[i] + z * b[(len + i - shift) % len];
            //assert(a[i] <= 1 + 1e-5 && a[i] >= -1e-5);
        }
    }

    std::vector<double> perm_sum(const std::vector<double> &a,
                                 const std::vector<double> &b,
                                 const std::vector<seq_type> &perm,
                                 double z,
                                 double gap_penalty = 0) {
        // std::cerr << "Start perm sum" << std::endl;
        assert(a.size() == b.size());
        if (perm.size() != sketch_dim) {
            // std::cerr << "SIZE: " << perm.size() << std::endl;
            // std::cerr << "SKETCH SIZE: " << int(sketch_size) << std::endl;
            assert(false);
        }

        std::vector<double> result(a.size());
        for (uint32_t i = 0; i < a.size(); i++) {
            result[i] = (1 - gap_penalty) * (1 - z) * a[i] + z * b[perm[i]];
            //assert(result[i] <= 1 + 1e-5 && result[i] >= -1e-5);
        }
        // std::cerr << "done" << std::endl;
        return result;
    }

    /** Size of the alphabet over which sequences to be sketched are defined, e.g. 4 for DNA */
    const seq_type alphabet_size;
    /** Number of elements in the sketch, denoted by D in the paper */
    const uint32_t sketch_dim;
    /** The length of the subsequences considered for sketching, denoted by t in the paper */
    const uint8_t subsequence_len;
    /** Gaps penalize the weight by (1-gap_penalty)^gaps */
    const double gap_penalty;

    /**
     * Denotes the hash functions h1,....ht:A->{1....D}, where t is #subsequence_len and D is
     * #sketch_dim
     */
    Vec2D<seq_type> hashes;

    /** The sign functions s1...st:A->{-1,1} */
    Vec2D<bool> signs;

    std::mt19937 rng;

    bool injective_hash;
    bool use_permutation;
    Vec3D<seq_type> permutation_hashes;
};

} // namespace ts
