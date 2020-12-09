#pragma once

#include "util/multivec.hpp"

#include <cassert>
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
        : alphabet_size(alphabet_size), sketch_size(sketch_size), subsequence_len(subsequence_len) {
        rand_init();
    }

    /**
     * Computes the sketch of the given sequence.
     * @param seq the sequence to be sketched
     * @return an array of size #sketch_size containing the sequence's sketch
     */
    std::vector<double> compute(const std::vector<seq_type> &seq) {
        if (seq.empty()) {
            return std::vector<double>(sketch_size);
        }

        // T1 is T(x,p,+1), T2 is T(x,p,-1) in the paper
        auto T1 = new2D<double>(subsequence_len + 1, sketch_size, 0);
        auto T2 = new2D<double>(subsequence_len + 1, sketch_size, 0);
        auto T1n = new2D<double>(subsequence_len + 1, sketch_size, 0);
        auto T2n = new2D<double>(subsequence_len + 1, sketch_size, 0);

        // the initial condition states that the sketch for the empty string is (1,0,..)
        T1n[0][0] = T1[0][0] = 1;
        for (uint32_t i = 0; i < seq.size(); i++) {
            for (uint32_t t = 1; t <= std::min(i + 1, (uint32_t)subsequence_len); ++t) {
                double z = t / (i + 1.0); // probability that the last index is i
                seq_type r = hashes[t - 1][seq[i]];
                bool s = signs[t - 1][seq[i]];
                if (s) {
                    T1n[t] = shift_sum(T1[t], T1[t - 1], r, z);
                    T2n[t] = shift_sum(T2[t], T2[t - 1], r, z);
                } else {
                    T1n[t] = shift_sum(T1[t], T2[t - 1], r, z);
                    T2n[t] = shift_sum(T2[t], T1[t - 1], r, z);
                }
            }
            std::swap(T1, T1n);
            std::swap(T2, T2n);
        }
        std::vector<double> sketch(sketch_size, 0);
        for (uint32_t m = 0; m < sketch_size; m++) {
            sketch[m] = T1[subsequence_len][m] - T2[subsequence_len][m];
        }
        return sketch;
    }

    std::vector<double> compute_old(const std::vector<seq_type> &seq) {
        std::vector<double> sketch;
        auto M = new2D<double>(subsequence_len + 1, sketch_size, 0);
        M[0][0] = 1;
        for (int i = 0; i < (int)seq.size(); i++) {
            for (int t = subsequence_len - 1; t >= 0; t--) {
                double z = (t + 1.0) / (i + 1);
                auto r = hashes[t][seq[i]];
                M[t + 1] = shift_sum(M[t + 1], M[t], r, z);
            }
        }
        sketch = std::vector<double>(sketch_size, 0);
        for (int m = 0; m < sketch_size; m++) {
            sketch[m] = M[subsequence_len][m];
        }
        return sketch;
    }

    void set_hashes_for_testing(const Vec2D<seq_type> &hashes, const Vec2D<bool> &signs) {
        this->hashes = hashes;
        this->signs = signs;
    }


  protected:
    std::vector<double> shift_sum(const std::vector<double> &a, const std::vector<double> &b, seq_type shift, double z) {
        assert(a.size() == b.size());
        size_t len = a.size();
        std::vector<double> result(a.size());
        for (uint32_t i = 0; i < a.size(); i++) {
            result[i] = (1 - z) * a[i] + z * b[(len + i - shift) % len];
            assert(result[i] <= 1 + 1e-5 && result[i] >= -1 - 1e-5);
        }
        return result;
    }

    virtual void rand_init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<seq_type> rand_hash2(0, sketch_size - 1);
        std::uniform_int_distribution<seq_type> rand_bool(0, 1);

        hashes = new2D<seq_type>(subsequence_len, alphabet_size);
        signs = new2D<bool>(subsequence_len, alphabet_size);
        for (size_t h = 0; h < subsequence_len; h++) {
            for (size_t c = 0; c < alphabet_size; c++) {
                hashes[h][c] = rand_hash2(gen);
                signs[h][c] = rand_bool(gen);
            }
        }
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
