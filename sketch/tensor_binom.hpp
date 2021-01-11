#pragma once

#include "util/multivec.hpp"

#include <cassert>
#include <iostream> //todo: remove
#include <random>

namespace ts { // ts = Tensor Sketch

/**
 * Computes tensor sketches for a given sequence as described in
 * https://www.biorxiv.org/content/10.1101/2020.11.13.381814v1
 * @tparam seq_type the type of elements in the sequences to be sketched.
 */
template <class seq_type>
class TensorBinom {
  public:
    /**
     * @param alphabet_size the number of elements in the alphabet S over which sequences are
     * defined (e.g. 4 for DNA)
     * @param sketch_size the dimension of the embedded (sketched) space, denoted by D in the paper
     * @param subsequence_len the length of the subsequences considered for sketching, denoted by t
     * in the paper
     */
    TensorBinom(seq_type alphabet_size,
                size_t sketch_size,
                size_t subsequence_len,
                bool use_permutation)
        : alphabet_size(alphabet_size),
          sketch_size(sketch_size),
          subsequence_len(subsequence_len),
          hashes(alphabet_size),
          signs(alphabet_size),
          use_permutation(use_permutation) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<seq_type> rand_hash2(0, sketch_size - 1);
        std::uniform_int_distribution<seq_type> rand_bool(0, 1);

        for (size_t c = 0; c < alphabet_size; c++) {
            hashes[c] = rand_hash2(gen);
            signs[c] = 1; // rand_bool(gen);
        }

        permutations = new2D<int>(alphabet_size, sketch_size);
        for (size_t i = 0; i < alphabet_size; ++i) {
            // Assign a random permutation to permutations[i].
            std::iota(std::begin(permutations[i]), std::end(permutations[i]), 0);
            std::shuffle(std::begin(permutations[i]), std::end(permutations[i]), gen);
        }
    }

    /**
     * Computes the sketch of the given sequence.
     * @param seq the sequence to be sketched
     * @return an array of size #sketch_size containing the sequence's sketch
     */
    std::vector<double> compute(const std::vector<seq_type> &seq) {
        // Tp corresponds to T+, Tm to T- in the paper; Tp[0], Tm[0] are sentinels and contain the
        // initial condition for empty strings; Tp[p], Tm[p] represent the partial sketch when
        // considering hashes h1...hp, over the prefix x1...xi. The final result is then
        // Tp[t]-Tm[t], where t is #sequence_len
        std::vector<double> Tp(sketch_size, 0);
        std::vector<double> Tm(sketch_size, 0);

        // the initial condition states that the sketch for the empty string is (1,0,..)
        Tp[0] = 1;
        for (uint32_t i = 0; i < seq.size(); i++) {
            // Probability that we include index i.
            double z = subsequence_len * 1.0 / seq.size();
            seq_type r = hashes[seq[i]];
            bool s = signs[seq[i]];
            if (s) {
                Tp = shift_sum(Tp, Tp, r, z, seq[i]);
                Tm = shift_sum(Tm, Tm, r, z, seq[i]);
            } else {
                Tp = shift_sum(Tp, Tm, r, z, seq[i]);
                Tm = shift_sum(Tm, Tp, r, z, seq[i]);
            }
        }
        std::vector<double> sketch(sketch_size, 0);
        for (uint32_t m = 0; m < sketch_size; m++) {
            sketch[m] = Tp[m] - Tm[m];
        }
        return sketch;
    }

    /** Sets the hash and sign functions to predetermined values for testing */
    void set_hashes_for_testing(const Vec2D<seq_type> &h, const Vec2D<bool> &s) {
        hashes = h;
        signs = s;
    }

  protected:
    /** Computes (1-z)*a + z*b_shift */
    std::vector<double> shift_sum(const std::vector<double> &a,
                                  const std::vector<double> &b,
                                  seq_type shift,
                                  double z,
                                  seq_type ch) {
        assert(a.size() == b.size());
        size_t len = a.size();
        std::vector<double> result(a.size());
        assert(ch < alphabet_size);
        const auto &perm = permutations[ch];
        assert(perm.size() == a.size());
        for (uint32_t i = 0; i < a.size(); i++) {
            if (use_permutation) {
                result[i] = (1 - z) * a[i] + z * b[perm[i]];
            } else {
                result[i] = (1 - z) * a[i] + z * b[(len + i - shift) % len];
            }

            assert(result[i] <= 1 + 1e-5 && result[i] >= -1e-5);
        }
        return result;
    }

    /** Size of the alphabet over which sequences to be sketched are defined, e.g. 4 for DNA */
    seq_type alphabet_size;
    /** Number of elements in the sketch, denoted by D in the paper */
    uint8_t sketch_size;
    /** The length of the subsequences considered for sketching, denoted by t in the paper */
    uint8_t subsequence_len;

    /**
     * Denotes the hash functions h1,....ht:A->{1....D}, where t is #subsequence_len and D is
     * #sketch_size
     */
    std::vector<seq_type> hashes;

    /** The sign functions s1...st:A->{-1,1} */
    std::vector<bool> signs;

    /** The permuations to apply for each element in A. */
    Vec2D<int> permutations;

    bool use_permutation;
};

} // namespace ts
