#pragma once

#include "sequence/alphabets.hpp"
#include "sketch/sketch_base.hpp"
#include "util/multivec.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include <bits/stdint-uintn.h>
#include <cassert>
#include <cmath>
#include <iostream> //todo: remove
#include <random>

namespace ts { // ts = Tensor Sketch

/**
 * Computes tensor sketches for a given sequence as described in
 * https://www.biorxiv.org/content/10.1101/2020.11.13.381814v1
 * @tparam seq_type the type of elements in the sequences to be sketched.
 */
template <class seq_type>
class TensorBinom : public SketchBase<std::vector<double>, false> {
  public:
    /**
     * @param alphabet_size the number of elements in the alphabet S over which sequences are
     * defined (e.g. 4 for DNA)
     * @param sketch_size the dimension of the embedded (sketched) space, denoted by D in the paper
     * @param subsequence_len the length of the subsequences considered for sketching, denoted by t
     * in the paper
     */
    TensorBinom(seq_type alphabet_size_arg,
                size_t sketch_size,
                size_t subsequence_len,
                uint32_t seed,
                const std::string &name = "BTS",
                double gap_penalty = 0)
        : SketchBase<std::vector<double>, false>(name),
          rng(seed),
          alphabet_size(alphabet_size_arg),
          sketch_size(sketch_size),
          subsequence_len(subsequence_len),
          gap_penalty(gap_penalty),
          signs(alphabet_size) {
        init();
    }

    void init() {
        std::uniform_int_distribution<seq_type> rand_hash2(0, sketch_size - 1);
        std::uniform_int_distribution<seq_type> rand_bool(0, 1);

        signs.resize(alphabet_size);

        for (size_t c = 0; c < alphabet_size; c++) {
            signs[c] = rand_bool(rng);
        }

        permutations.resize(alphabet_size);
        for (size_t i = 0; i < alphabet_size; ++i) {
            permutations[i].resize(sketch_size);
            // Assign a random permutation to permutations[i].
            std::iota(std::begin(permutations[i]), std::end(permutations[i]), 0);
            std::shuffle(std::begin(permutations[i]), std::end(permutations[i]), rng);
        }
    }

    /**
     * Computes the sketch of the given sequence.
     * @param seq the sequence to be sketched
     * @return an array of size #sketch_size containing the sequence's sketch
     */
    std::vector<double> compute(const std::vector<seq_type> &seq) {
        // 0: subsequence didn't start yet
        // 1: subsequence in progress
        // 2: subsequence ended
        //
        // layer 1 gets a (1-gap_penalty) factor applied when the character is not chosen.
        std::array<std::vector<double>, 3> Tp;
        std::array<std::vector<double>, 3> Tm;
        for (auto &t : Tp)
            t.resize(sketch_size, 0);
        for (auto &t : Tm)
            t.resize(sketch_size, 0);

        // the initial condition states that the sketch for the empty string is (1,0,..)
        Tp[0][0] = 1;
        // Probability that we include index i is constant.
        const double z = subsequence_len * 1.0 / seq.size();
        for (uint32_t i = 0; i < seq.size(); i++) {
            // TODO: We _could_ choose the sign (and optionally hash function as well) on the layer,
            // but that would still only specialize them for the first and last character of the
            // sequence, not all k characters.
            const bool s = signs[seq[i]];

            // Temporary: T01 = T[0] + T[1]
            std::vector<double> Tp01(sketch_size, 0), Tm01(sketch_size, 0);
            for (int i = 0; i < sketch_size; ++i) {
                Tp01[i] = Tp[0][i] + Tp[1][i];
                Tm01[i] = Tm[0][i] + Tm[1][i];
            }

            // Update layer 2 and 1:
            for (int p : { 2, 1 }) {
                const double gp = p == 2 ? 0 : gap_penalty;
                if (s) {
                    Tp[p] = shift_sum(Tp[p], Tp01, z, seq[i], gp);
                    Tm[p] = shift_sum(Tm[p], Tm01, z, seq[i], gp);
                } else {
                    Tp[p] = shift_sum(Tp[p], Tm01, z, seq[i], gp);
                    Tm[p] = shift_sum(Tm[p], Tp01, z, seq[i], gp);
                }
            }

            // Update layer 0:
            for (double &x : Tp[0])
                x *= (1 - z);
            for (double &x : Tm[0])
                x *= (1 - z);
        }
        std::vector<double> sketch(sketch_size, 0);
        for (uint32_t m = 0; m < sketch_size; m++) {
            sketch[m] = Tp[2][m] - Tm[2][m];
        }
        return sketch;
    }

    /** Sets the hash and sign functions to predetermined values for testing */
    void set_hashes_for_testing(const Vec2D<seq_type> &h, const std::vector<bool> &s) {
        signs = s;
        permutations = h;
    }

    double dist(const std::vector<double> &a, const std::vector<double> &b) {
        Timer timer("tensor_sketch_dist");
        return l2_dist(a, b);
    }

  protected:
    /** Computes (1-z)*a + z*b_shift */
    std::vector<double> shift_sum(const std::vector<double> &a,
                                  const std::vector<double> &b,
                                  double z,
                                  seq_type ch,
                                  double gap_penalty) {
        assert(a.size() == b.size());
        std::vector<double> result(a.size());
        assert(ch < alphabet_size);
        const auto &perm = permutations[ch];
        assert(perm.size() == a.size());
        for (uint32_t i = 0; i < a.size(); i++) {
            result[i] = (1 - z) * (1 - gap_penalty) * a[i] + z * b[perm[i]];

            // assert(result[i] <= 1 + 1e-5 && result[i] >= -1e-5);
        }
        return result;
    }


    std::mt19937 rng;

    /** Size of the alphabet over which sequences to be sketched are defined, e.g. 4 for DNA */
    const seq_type alphabet_size;
    /** Number of elements in the sketch, denoted by D in the paper */
    const uint8_t sketch_size;
    /** The length of the subsequences considered for sketching, denoted by t in the paper */
    const uint8_t subsequence_len;
    /** Gaps penalize the weight by (1-gap_penalty)^gaps */
    const double gap_penalty;

    /** The sign functions s1...st:A->{-1,1} */
    std::vector<bool> signs;

    /** The permuations to apply for each element in A. */
    Vec2D<int> permutations;
};

} // namespace ts
