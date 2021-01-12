#pragma once

#include "tensor.hpp"

#include "util/utils.hpp"

#include <cstddef>
#include <vector>
#include <bit>

namespace ts {
/**
 * Computes sliding tensor sketches for a given sequence as described in
 * https://www.biorxiv.org/content/10.1101/2020.11.13.381814v1
 * @tparam seq_type the type of elements in the sequences to be sketched.
 */
template <class seq_type>
class TensorSlide : public Tensor<seq_type> {
  public:
    TensorSlide() {}

    /**
     * @param alphabet_size the number of elements in the alphabet S over which sequences are
     * defined (e.g. 4 for DNA)
     * @param sketch_dim the dimension of the embedded (sketched) space, denoted by D in the paper
     * @param subsequence_len the length of the subsequences considered for sketching, denoted by t
     * in the paper
     * @param win_len sliding sketches are computed for substrings of size win_len
     * @param stride sliding sketches are computed every stride characters
     */
    TensorSlide(seq_type alphabet_size,
                size_t sketch_dim,
                size_t tup_len,
                size_t win_len,
                size_t stride,
                size_t max_len = 0,
                size_t inner_dim = 0)
        : Tensor<seq_type>(alphabet_size, inner_dim>0? inner_dim : sketch_dim, tup_len),
                win_len(win_len), stride(stride), flat_dim(sketch_dim*16), max_len(max_len) {
        assert(stride <= win_len && "Stride cannot be larger than the window length");
        assert(tup_len <= stride && "Tuple length (t) cannot be larger than the stride");
        assert((this->flat_dim) % this->sketch_dim == 0 && "sketch_dim must be divisible by inner_dim");
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> distribution(0,1.0);
        rand_proj = new2D<double>(this->flat_dim, max_len * sketch_dim *2 / stride);
        for (auto & v : rand_proj) {
            for (double & e : v) {
                e = distribution(gen);
            }
        }
    }

    /**
     * Computes sliding sketches for the given sequence.
     * A sketch is computed every #stride characters on substrings of length #window.
     * @return seq.size()/stride sketches of size #sketch_dim
     */
    Vec2D<double> compute(const std::vector<seq_type> &seq) {
        Timer timer("tensor_slide_sketch");
        Vec2D<double> sketches;
        if (seq.size() < this->subsequence_len) {
            return new2D<double>(seq.size() / this->stride, this->sketch_dim, double(0));
        }
        auto &hashes = this->hashes;
        auto &signs = this->signs;
        auto tup_len = this->subsequence_len;
        // first index: p; second index: q; third index: r
        // p,q go from 1 to tup_len; p==0 and p==tup_len+1 are sentinels for termination condition
        auto T1 = new3D<double>(tup_len + 2, tup_len + 1, this->sketch_dim, 0);
        auto T2 = new3D<double>(tup_len + 2, tup_len + 1, this->sketch_dim, 0);

        for (uint32_t p = 0; p <= tup_len; p++) {
            T1[p + 1][p][0] = 1;
        }

        // T[p][q] at step i represents the sketch for seq[i-w+1]...seq[i] when only using hash
        // functions 1<=p,p+1,...q<=t, where t is the sketch size
        for (uint32_t i = 0; i < seq.size(); i++) {
            for (uint32_t p = 1; p <= tup_len; p++) {
                // q-p must be smaller than i, hence the min in the condition
                for (uint32_t q = std::min(p + i, (uint32_t)tup_len); q >= p; q--) {
                    double z = (double)(q - p + 1) / std::min(i + 1, win_len + 1);
                    auto r = hashes[q - 1][seq[i]];
                    bool s = signs[q - 1][seq[i]];
                    if (s) {
                        this->shift_sum_inplace(T1[p][q], T1[p][q - 1], r, z);
                        this->shift_sum_inplace(T2[p][q], T2[p][q - 1], r, z);
                    } else {
                        this->shift_sum_inplace(T1[p][q], T2[p][q - 1], r, z);
                        this->shift_sum_inplace(T2[p][q], T1[p][q - 1], r, z);
                    }
                }
            }

            if (i >= win_len) { // only start deleting from front after reaching #win_len
                uint64_t ws = i - win_len; // the element to be removed from the sketch
                for (uint32_t diff = 0; diff < tup_len; ++diff) {
                    for (uint32_t p = 1; p <= tup_len - diff; p++) {
                        auto r = hashes[p - 1][seq[ws]];
                        bool s = signs[p - 1][seq[ws]];
                        uint32_t q = p + diff;
                        // this computes t/(w-t); in our case t (the tuple length) is diff+1
                        double z = (double)(diff + 1) / (win_len - diff);
                        if (s) {
                            this->shift_sum_inplace(T1[p][q], T1[p + 1][q], r, -z);
                            this->shift_sum_inplace(T2[p][q], T2[p + 1][q], r, -z);
                        } else {
                            this->shift_sum_inplace(T1[p][q], T2[p + 1][q], r, -z);
                            this->shift_sum_inplace(T2[p][q], T1[p + 1][q], r, -z);
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

    double dist(const Vec2D<double> &a, const Vec2D<double> &b) {
        Timer timer("tensor_slide_sketch_dist");
        return l1_dist2D_minlen(a, b);
    }

    std::vector<double> flatten(const Vec2D<double> &sketch) {
        Timer timer("tensor_slide_flat");
        assert(rand_proj.size()==flat_dim && !rand_proj[0].empty() && " random matrix must be initialized");
        std::vector<double> v(this->flat_dim,0);
        for (size_t s = 0; s < this->flat_dim; s++) {
            size_t j = s % this->sketch_dim;
            for (size_t i=0; i< sketch.size();i++) {
//                for (size_t j=0; j< sketch[i].size();j++) {
                    v[s] += rand_proj[s][i*this->sketch_dim + j] * sketch[i][j];
//                }
            }
            v[s] /=  (double)(sketch.size() * sketch[0].size() ); // divide by number of elements to compute the mean
        }
        return  v;
    }

    double dist_flat(const std::vector<double> &v1, const std::vector<double> &v2) {
        Timer timer("tensor_slide_flat_dist");
        assert(v1.size()==v2.size());
        std::vector<double> d(v1.size());
        for (size_t i = 0; i < d.size(); i++) {
            d[i] = abs(v1[i] - v2[i]);
        }
        std::sort(d.begin(), d.end());
        return d[d.size()/2]; // return the median
    }

    std::vector<uint32_t> flatten_sign(const Vec2D<double> &sketch) {
        Timer timer("tensor_slide_sign");
        assert(rand_proj.size()==flat_dim && !rand_proj[0].empty() && " random matrix must be initialized");
        size_t pitch = (max_len + stride+1)/stride;
        std::vector<uint32_t> v(this->flat_dim/32,0);
        for (size_t s = 0; s < this->flat_dim; s++) {
            size_t j = s % this->sketch_dim;
            double val = 0;
            for (size_t i=0; i< sketch.size();i++) {
                val += rand_proj[s][i+ j * pitch] * sketch[i][j];
            }
            v[s >> 5] |= (val>0 ? 1 : 0) << (s & 31); // check the sign
        }
        return v;
    }

    double dist_sign(const std::vector<uint32_t> &v1, const std::vector<uint32_t> &v2) {
        Timer timer("tensor_slide_sign_dist");
        assert(v1.size()==v2.size());
        std::vector<double> d(v1.size());
        double val = 0;
        for (size_t i = 0; i < d.size(); i++) {
            val += std::__popcount(v1[i]^v2[i] );
        }
        return val;
    }

  private:
    std::vector<double> diff(const std::vector<double> &a, const std::vector<double> &b) {
        assert(a.size() == b.size());
        std::vector<double> result(a.size());
        for (uint32_t i = 0; i < result.size(); ++i) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    uint32_t win_len;
    uint32_t stride;
    uint32_t flat_dim;
    uint32_t max_len;
    Vec2D<double> rand_proj;
};

} // namespace ts
