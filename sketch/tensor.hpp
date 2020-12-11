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
class Tensor {
  public:
    /**
     * @param alphabet_size the number of elements in the alphabet S over which sequences are
     * defined (e.g. 4 for DNA)
     * @param sketch_size the dimension of the embedded (sketched) space, denoted by D in the paper
     * @param subsequence_len the length of the subsequences considered for sketching, denoted by t
     * in the paper
     */
    Tensor(seq_type alphabet_size, size_t sketch_size, size_t subsequence_len)
        : alphabet_size(alphabet_size),
          sketch_size(sketch_size),
          subsequence_len(subsequence_len),
          hashes(new2D<seq_type>(subsequence_len, alphabet_size)),
          signs(new2D<bool>(subsequence_len, alphabet_size)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<seq_type> rand_hash2(0, sketch_size - 1);
        std::uniform_int_distribution<seq_type> rand_bool(0, 1);

        for (size_t h = 0; h < subsequence_len; h++) {
            for (size_t c = 0; c < alphabet_size; c++) {
                hashes[h][c] = rand_hash2(gen);
                signs[h][c] = rand_bool(gen);
            }
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
        auto Tp = new2D<double>(subsequence_len + 1, sketch_size, 0);
        auto Tm = new2D<double>(subsequence_len + 1, sketch_size, 0);

        // the initial condition states that the sketch for the empty string is (1,0,..)
        Tp[0][0] = 1;
        for (uint32_t i = 0; i < seq.size(); i++) {
            // must traverse in reverse order, to avoid overwriting the values of Tp and Tm before
            // they are used in the recurrence
            for (uint32_t p = std::min(i + 1, (uint32_t)subsequence_len); p >= 1; --p) {
                double z = p / (i + 1.0); // probability that the last index is i
                seq_type r = hashes[p - 1][seq[i]];
                bool s = signs[p - 1][seq[i]];
                if (s) {
                    Tp[p] = shift_sum(Tp[p], Tp[p - 1], r, z);
                    Tm[p] = shift_sum(Tm[p], Tm[p - 1], r, z);
                } else {
                    Tp[p] = shift_sum(Tp[p], Tm[p - 1], r, z);
                    Tm[p] = shift_sum(Tm[p], Tp[p - 1], r, z);
                }
            }
        }
        std::vector<double> sketch(sketch_size, 0);
        for (uint32_t m = 0; m < sketch_size; m++) {
            sketch[m] = Tp[subsequence_len][m] - Tm[subsequence_len][m];
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
                                  double z) {
        assert(a.size() == b.size());
        size_t len = a.size();
        std::vector<double> result(a.size());
        for (uint32_t i = 0; i < a.size(); i++) {
            result[i] = (1 - z) * a[i] + z * b[(len + i - shift) % len];
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
    Vec2D<seq_type> hashes;

    /** The sign functions s1...st:A->{-1,1} */
    Vec2D<bool> signs;
};

} // namespace ts
