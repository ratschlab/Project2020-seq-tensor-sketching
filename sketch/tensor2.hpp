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
template <class set_type, class sketch_type>
class Tensor2 {
  public:
    /**
     * @param alphabet_size the number of elements in S
     * @param embedded_dim the dimension of the embedded (sketched) space, denoted by D in the paper
     * @param tup_len the length of the subsequences considered for sketching, denoted by t in the
     * paper
     */
    Tensor2(set_type alphabet_size, size_t embed_dim, size_t num_phases, size_t tup_len)
        : alphabet_size(alphabet_size),
          embed_dim(embed_dim),
          num_phases(num_phases),
          tup_len(tup_len),
          hash_len(num_phases * embed_dim) {
        rand_init();
    }

    Vec<sketch_type> compute(const Seq<set_type> &seq) {
        auto T = new2D<sketch_type>(tup_len + 1, hash_len, 0);
        T[0][0] = 1;
        for (uint32_t i = 0; i < seq.size(); i++) {
            for (int32_t t = (int32_t)tup_len - 1; t >= 0; t--) {
                double z = (t + 1.0) / (i + 1);
                set_type r = hashes[t][seq[i]];
                shift_sum(T[t + 1], T[t], r, z);
            }
        }
        Vec<sketch_type> sketch(embed_dim, 0);
        for (int32_t m = 0; m < embed_dim; m++) {
            sketch_type prod = 0;
            for (int32_t r = 0; r < num_phases; r++) {
                prod += ((r % 2 == 0) ? 1 : -1) * T[tup_len][m * num_phases + r];
            }
            sketch[m] = prod;
        }
        return sketch;
    }

  private:
    void shift_sum(Vec<sketch_type> &a, const Vec<sketch_type> &b, set_type shift, double z) {
        assert(a.size() == b.size());
        size_t len = a.size();
        for (uint32_t i = 0; i < a.size(); i++) {
            a[i] = (1 - z) * a[i] + z * b[(i - shift + len) % len];
        }
    }

  protected:
    set_type alphabet_size;
    uint8_t embed_dim;
    size_t num_phases;
    /** The length of the subsequences considered for sketching, denoted by t in the paper */
    uint8_t tup_len;
    /** The size of the embedded space, denoted by D in the paper */
    size_t hash_len;

    /**
     * Denotes the hash functions h1,....ht:A->{1....D}, where t is #tup_len and D is #hash_len
     */
    Vec2D<set_type> hashes;

    virtual void rand_init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<set_type> rand_hash2(0, hash_len - 1);

        hashes = new2D<set_type>(tup_len, alphabet_size);
        for (size_t h = 0; h < tup_len; h++) {
            for (size_t c = 0; c < alphabet_size; c++) {
                hashes[h][c] = rand_hash2(gen);
            }
        }
    }
};

template <class set_type, class sketch_type>
class TensorSlide2 : public Tensor2<set_type, sketch_type> {
  public:
    TensorSlide2(set_type alphabet_size,
                 size_t embed_dim,
                 size_t num_phases,
                 size_t tup_len,
                 size_t win_len,
                 size_t stride)
        : Tensor2<set_type, sketch_type>(alphabet_size, embed_dim, num_phases, tup_len),
          win_len(win_len),
          stride(stride) {}

    void compute(const Vec<set_type> &seq, Vec2D<sketch_type> &embedding) {
        auto T = new3D<sketch_type>(this->tup_len + 1, this->tup_len + 1, this->hash_len, 0);
        for (size_t p = 0; p < this->tup_len; p++) {
            T[p + 1][p][0] = 1;
        }

        for (size_t i = 0; i < seq.size(); i++) {
            for (size_t p = 0; p < this->tup_len; p++) {
                for (size_t q = this->tup_len - 1; q >= p; q--) {
                    double z = (double)(q - p + 1) / std::min(i + 1, (size_t)win_len);
                    auto r = this->hash[q][seq[i]];
                    shift_sum(T[p + 1][q + 1], T[p + 1][q], r, z);
                }
            }

            if ((i + 1) % stride == 0) {
                Vec<sketch_type> em(this->embed_dim);
                for (size_t m = 0; m < this->embed_dim; m++) {
                    sketch_type prod = 0;
                    for (size_t r = 0; r < this->num_phases; r++) {
                        prod += ((r % 2 == 0) ? 1 : -1)
                                * T[1][this->tup_len][m * this->num_phases + r];
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
                    shift_sum(T[p + 1][q + 1], T[p + 1][q], r, -z);
                }
            }
        }
    }

    void conv_slide_sketch(const Vec2D<set_type> &seq, Vec2D<sketch_type> &embedding) {
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
                Vec<sketch_type> em(this->embed_dim);
                for (size_t m = 0; m < this->embed_dim; m++) {
                    sketch_type prod = 0;
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
