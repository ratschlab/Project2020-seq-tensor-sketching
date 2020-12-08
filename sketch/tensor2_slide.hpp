#pragma once

#include "tensor2.hpp"

namespace ts {
template <class set_type>
class TensorSlide2 : public Tensor2<set_type> {
  public:
    TensorSlide2(set_type alphabet_size,
                 size_t sketch_size,
                 size_t tup_len,
                 size_t win_len,
                 size_t stride)
        : Tensor2<set_type>(alphabet_size, sketch_size, tup_len),
          win_len(win_len),
          stride(stride) {
        assert(stride <= win_len && "Stride cannot be larger than the window length");
        assert(tup_len <= stride && "Tuple length (t) cannot be larger than the stride");
    }

    Vec2D<double> compute_old(const Vec<set_type> &seq) {
        Vec2D<double> sketch;
        auto T = new3D<double>(this->tup_len + 1, this->tup_len + 1, this->sketch_size,
                                    double(0));
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
                Vec<double> em(this->sketch_size);
                for (size_t m = 0; m < this->sketch_size; m++) {
                    double prod = 0;
                    for (size_t r = 0; r < this->num_phases; r++) {
                        prod += ((r % 2 == 0) ? 1 : -1)
                                * T[1][this->tup_len][m * this->num_phases + r];
                    }
                    em[m] = prod;
                }
                sketch.push_back(em);
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

    std::vector<double> diff(const std::vector<double> &a,
                                  const std::vector<double> &b) {
        assert(a.size() == b.size());
        std::vector<double> result(a.size());
        for (uint32_t i = 0; i < result.size(); ++i) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    void set_zero(std::initializer_list<std::vector<double> *> list) {
        for (auto elem : list) {
            std::fill(elem->begin(), elem->end(), double(0));
        }
    }

    /**
     * Computes the sketch for the given sequence.
     * A sketch is computed every #stride characters on substrings of length #window.
     * @return seq.size()/stride sketches of size #sketch_size
     */
    Vec2D<double> compute(const Vec<set_type> &seq) {
        Vec2D<double> sketches;
        if (seq.size() < this->tup_len) {
            return new2D<double>(seq.size() / this->stride, this->sketch_size, double(0));
        }
        auto &hashes = this->hashes;
        auto &signs = this->signs;
        auto tup_len = this->tup_len;
        // first index: p; second index: q; third index: r
        // p,q go from 1 to tup_len; p==0 and p==tup_len+1 are sentinels for termination condition
        auto T1 = new3D<double>(tup_len + 2, tup_len + 1, this->sketch_size, 0);
        auto T2 = new3D<double>(tup_len + 2, tup_len + 1, this->sketch_size, 0);

        for (size_t p = 0; p <= tup_len; p++) {
            T1[p + 1][p][0] = 1;
        }

        // T[p][q] at step i represents the sketch for seq[i-w+1]...seq[i] when only using hash
        // functions 1<=p,p+1,...q<=t, where t is the sketch size
        for (size_t i = 0; i < seq.size(); i++) {
            for (size_t p = 1; p <= tup_len; p++) {
                // q-p must be smaller than i, hence the min in the condition
                for (uint32_t q = std::min(p + i, (size_t)tup_len); q >= p; q--) {
                    double z = (double)(q - p + 1) / std::min(i + 1, win_len + 1);
                    auto r = hashes[q - 1][seq[i]];
                    bool s = signs[q - 1][seq[i]];
                    if (s) {
                        T1[p][q] = this->shift_sum(T1[p][q], T1[p][q - 1], r, z);
                        T2[p][q] = this->shift_sum(T2[p][q], T2[p][q - 1], r, z);
                    } else {
                        T1[p][q] = this->shift_sum(T1[p][q], T2[p][q - 1], r, z);
                        T2[p][q] = this->shift_sum(T2[p][q], T1[p][q - 1], r, z);
                    }
                }
            }

            if (i >= win_len) { // only start deleting from front after reaching #win_len
                uint32_t ws = i - win_len; // the element to be removed from the sketch
                for (size_t diff = 0; diff < tup_len; ++diff) {
                    for (size_t p = 1; p <= tup_len - diff; p++) {
                        auto r = hashes[p - 1][seq[ws]];
                        bool s = signs[p - 1][seq[ws]];
                        size_t q = p + diff;
                        // this computes t/(w-t); in our case t (the tuple length) is diff+1
                        double z = (double)(diff + 1) / (win_len - diff);
                        if (s) {
                            T1[p][q] = this->shift_sum(T1[p][q], T1[p + 1][q], r, -z);
                            T2[p][q] = this->shift_sum(T2[p][q], T2[p + 1][q], r, -z);
                        } else {
                            T1[p][q] = this->shift_sum(T1[p][q], T2[p + 1][q], r, -z);
                            T2[p][q] = this->shift_sum(T2[p][q], T1[p + 1][q], r, -z);
                        }
                    }
                }
            }

            if ((i + 1) % stride == 0) { // save a sketch every stride times
                sketches.push_back(diff(T1[1].back(), T2[1].back()));
            }
        }
        return sketches;
    }

    void conv_slide_sketch(const Vec2D<set_type> &seq, Vec2D<double> &sketch) {
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
                Vec<double> em(this->sketch_size);
                for (size_t m = 0; m < this->sketch_size; m++) {
                    double prod = 0;
                    for (size_t r = 0; r < this->num_phases; r++) {
                        prod += ((r % 2 == 0) ? 1 : -1)
                                * M[1][this->tup_len][m * this->num_phases + r];
                    }
                    em[m] = prod;
                }
                sketch.push_back(em);
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
    size_t win_len;
    size_t stride;
};

} // namespace ts
