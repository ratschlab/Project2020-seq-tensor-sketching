#pragma once

#include "tensor.hpp"

namespace ts { // ts = Tensor Sketch

/**
 * Computes tensor slide sketches for a given sequence.
 * @tparam sketch_type the type of elements in the sequences to be sketched.
 */
template <class set_type, class sketch_type>
class TensorSlideOld : public Tensor<set_type, sketch_type> {
  public:
    /**
     * @param set_size the number of elements in S,
     * @param sketch_dim the number of components (elements) in the sketch vector.
     */
    TensorSlideOld(sketch_type set_size,
                size_t sketch_dim,
                size_t num_phases,
                size_t num_bins,
                size_t tup_len,
                size_t win_len,
                size_t stride,
                size_t offset)
        : Tensor<set_type, sketch_type>(set_size, sketch_dim, num_phases, num_bins, tup_len),
          win_len(win_len),
          stride(stride),
          offset(offset) {
        this->rand_init();
    }

    void compute(const Seq<set_type> &seq, Vec2D<sketch_type> &sketch) {
        Timer::start("tensor_slide_sketch");
        sketch = Vec2D<sketch_type>(this->sketch_count, std::vector<sketch_type>());
        for (size_t m = 0; m < this->sketch_count; m++) {
            auto cnt = new3D<double>(this->tup_len, this->tup_len, this->embedded_dim, 0);
            for (size_t i = 0; i < seq.size(); i++) {
                if (i >= win_len) {
                    size_t j = i - win_len;
                    for (size_t t = 0; t < this->tup_len; t++) {
                        auto pj = this->hashes[m][t][seq[j]];
                        cnt[t][t][pj]--;
                        for (int t2 = t - 1; t2 >= 0; t2--) {
                            auto pj = this->hashes[m][t2][seq[j]];
                            for (size_t p = 0; p < this->embedded_dim; p++) {
                                auto shift = (p + pj) % this->embedded_dim;
                                cnt[t2][t][shift] -= cnt[t2 + 1][t][p];
                            }
                        }
                    }
                }

                for (size_t t = 0; t < this->tup_len; t++) {
                    for (size_t t2 = this->tup_len - 1; t2 > t; t2--) {
                        auto pi = this->hashes[m][t2][seq[i]];
                        for (size_t p = 0; p < this->embedded_dim; p++) {
                            auto shift = (p + pi) % this->embedded_dim;
                            cnt[t][t2][shift] += cnt[t][t2 - 1][p];
                        }
                    }
                    auto pi = this->hashes[m][t][seq[i]];
                    cnt[t][t][pi]++;
                }
                if (sketch_now(i, seq.size(), stride, offset)) {
                    const auto &top_cnt = cnt[0][this->tup_len - 1];
                    auto prod = std::inner_product(this->s[m].begin(), this->s[m].end(),
                                                   top_cnt.begin(), (double)0);
                    prod = prod / l1(top_cnt);
                    //                    int exp;
                    //                    frexp(prod, &exp);
                    //                    embedding[m].push_back(exp * sgn(prod));
                    sketch_type bin = std::upper_bound(this->bins.begin(),
                                             this->bins.begin() + this->num_bins, prod)
                            - this->bins.begin();
                    sketch[m].push_back(bin);
                }
            }
        }
        Timer::stop();
    }

  private:
    size_t win_len;
    size_t stride;
    size_t offset;
};

} // namespace ts
