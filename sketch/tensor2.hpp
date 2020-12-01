#pragma once

#include "util/multivec.hpp"

#include <random>

namespace ts { // ts = Tensor Sketch

/**
 * Alternative to #Tensor.
 * Computes tensor sketches for a given sequence as described in
 * https://www.biorxiv.org/content/10.1101/2020.11.13.381814v1
 * @tparam T the type of elements in the sketch and in the sequences to be sketched.
 */
template <class T>
class Tensor2 {
  public:
    /**
     * @param alphabet_size the number of elements in S
     * @param embedded_dim the dimension of the embedded (sketched) space, denoted by D in the paper
     * @param tup_len the length of the subsequences considered for sketching, denoted by t in the
     * paper
     */
    Tensor2(T alphabet_size, size_t embed_dim, size_t num_phases, size_t tup_len)
        : alphabet_size(alphabet_size),
          embed_dim(embed_dim),
          num_phases(num_phases),
          tup_len(tup_len),
          hash_len(num_phases * embed_dim) {
        rand_init();
    }

    template <class seq_type, class embed_type>
    void tensor2_sketch(const Seq<seq_type> &seq, Vec<embed_type> &embedding) {
        auto M = new2D<double>(tup_len + 1, hash_len, 0);
        M[0][0] = 1;
        for (int i = 0; i < seq.size(); i++) {
            for (int t = tup_len - 1; t >= 0; t--) {
                double z = (t + 1.0) / (i + 1);
                auto r = hash[t][seq[i]];
                shift_sum(M[t + 1], M[t], r, z);
            }
        }
        embedding = Vec<embed_type>(embed_dim, 0);
        for (int m = 0; m < embed_dim; m++) {
            embed_type prod = 0;
            for (int r = 0; r < num_phases; r++) {
                prod += ((r % 2 == 0) ? 1 : -1) * M[tup_len][m * num_phases + r];
            }
            int exp;
            frexp(prod, &exp);
            embedding[m] = exp * sgn(prod);
            embedding[m] = prod;
        }
    }

  private:
    void conv_sum(Vec<T> &a, const Vec<T> &b, double z) {
        assert(a.size() == b.size());
        int len = a.size;
        Vec<T> result(len, 0);
        for (int i = 0; i < len; i++) {
            T val = 0;
            for (int j = 0; j < len; j++) {
                val += a[(i + j) % len] * b[(i - j + len) % len];
            }
            a[i] = (1 - z) * a[i] + z * val;
        }
        std::swap(a, result);
    }

    void shift_sum(Vec<T> &a, const Vec<T> &b, int sh, double z) {
        assert(a.size() == b.size());
        int len = a.size();
        for (int i = 0; i < a.size(); i++) {
            a[i] = (1 - z) * a[i] + z * b[(i - sh + len) % len];
        }
    }

  protected:
    size_t alphabet_size;
    size_t embed_dim;
    size_t num_phases;
    size_t tup_len;
    size_t hash_len;

    Vec2D<int> hash;

    virtual void rand_init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> rand_hash2(0, hash_len - 1);

        hash = new2D<int>(tup_len, alphabet_size);
        for (size_t h = 0; h < tup_len; h++) {
            for (size_t c = 0; c < alphabet_size; c++) {
                hash[h][c] = rand_hash2(gen);
            }
        }
    }
};

template <class T>
class TensorSlide2 : public Tensor2<T> {
  public:
    TensorSlide2(T alphabet_size,
                 size_t embed_dim,
                 size_t num_phases,
                 size_t tup_len,
                 size_t win_len,
                 size_t stride)
        : Tensor2<T>(alphabet_size, embed_dim, num_phases, tup_len),
          win_len(win_len),
          stride(stride) {}

    template <class seq_type, class embed_type>
    void tensor2_slide_sketch(const Vec<seq_type> &seq, Vec2D<embed_type> &embedding) {
        auto M = new3D<double>(this->tup_len + 1, this->tup_len + 1, this->hash_len, 0);
        for (size_t p = 0; p < this->tup_len; p++) {
            M[p + 1][p][0] = 1;
        }

        for (size_t i = 0; i < seq.size(); i++) {
            for (size_t p = 0; p < this->tup_len; p++) {
                for (size_t q = this->tup_len - 1; q >= p; q--) {
                    double z = (double)(q - p + 1) / std::min(i + 1, (size_t)win_len);
                    auto r = this->hash[q][seq[i]];
                    shift_sum(M[p + 1][q + 1], M[p + 1][q], r, z);
                }
            }

            if ((i + 1) % stride == 0) {
                Vec<embed_type> em(this->embed_dim);
                for (size_t m = 0; m < this->embed_dim; m++) {
                    embed_type prod = 0;
                    for (size_t r = 0; r < this->num_phases; r++) {
                        prod += ((r % 2 == 0) ? 1 : -1)
                                * M[1][this->tup_len][m * this->num_phases + r];
                    }
                    em[m] = prod;
                }
                embedding.push_back(em);
            }

            if (i < win_len) {
                continue;
            }

            for (size_t p = 0; p < this->tup_len; p++) {
                for (size_t q = p; q < this->tup_len; q++) {
                    double z = (double)(q - p + 1) / (win_len - q + p);
                    auto r = this->hash[q][seq[i]];
                    shift_sum(M[p + 1][q + 1], M[p + 1][q], r, -z);
                }
            }
        }
    }

    template <class seq_type, class embed_type>
    void conv_slide_sketch(const Vec2D<seq_type> &seq, Vec2D<embed_type> &embedding) {
        auto M = new3D<double>(this->tup_len + 1, this->tup_len + 1, this->hash_len, 0);
        for (size_t p = 0; p < this->tup_len; p++) {
            M[p + 1][p][0] = 1;
        }

        for (size_t i = 0; i < seq.size(); i++) {
            assert(seq[i].size() == this->alphabet_size);
            for (size_t p = 0; p < this->tup_len; p++) {
                for (int32_t q = (int32_t)this->tup_len - 1; q >= (int32_t)p; q--) {
                    double z = (double)(q - p + 1) / std::min((size_t)i + 1, (size_t)win_len);
                    auto r = this->hash[q][seq[i]];
                    shift_sum(M[p + 1][q + 1], M[p + 1][q], r, z);
                }
            }

            if ((i + 1) % stride == 0) {
                Vec<embed_type> em(this->embed_dim);
                for (size_t m = 0; m < this->embed_dim; m++) {
                    embed_type prod = 0;
                    for (size_t r = 0; r < this->num_phases; r++) {
                        prod += ((r % 2 == 0) ? 1 : -1)
                                * M[1][this->tup_len][m * this->num_phases + r];
                    }
                    em[m] = prod;
                }
                embedding.push_back(em);
            }

            if (i < win_len) {
                continue;
            }

            for (size_t p = 0; p < this->tup_len; p++) {
                for (size_t q = p; q < this->tup_len; q++) {
                    double z = (double)(q - p + 1) / (win_len - q + p);
                    auto r = this->hash[q][seq[i]];
                    shift_sum(M[p + 1][q + 1], M[p + 1][q], r, -z);
                }
            }
        }
    }

  private:
    int win_len;
    int stride;
};
} // namespace ts
