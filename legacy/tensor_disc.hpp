//
// Created by Amir Joudaki on 6/18/20.
//

#ifndef SEQUENCE_SKETCHING_TENSOR_DISC_HPP
#define SEQUENCE_SKETCHING_TENSOR_DISC_HPP

#include "sketch/tensor_slide.hpp"

namespace ts { // ts = Tensor Sketch

template <typename T>
bool in_range(const std::vector<T> &vec, T min = 0, T max = std::numeric_limits<T>::max()) {
    for (auto &v : vec)
        if (v < min or v > max)
            return false;
    return true;
}
template <typename T>
bool in_range2(const Vec2D<T> &vec, T min = 0, T max = std::numeric_limits<T>::max()) {
    for (auto &v : vec)
        if (not in_range<T>(v))
            return false;
    return true;
}
template <typename T>
bool in_range3(const Vec3D<T> &vec, T min = 0, T max = std::numeric_limits<T>::max()) {
    for (auto &v : vec) {
        if (not in_range2<T>(v))
            return false;
    }
    return true;
}
template <typename T>
std::vector<T> circ_conv(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.sie());
    const int N = a.size();
    std::vector<T> res(N, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res[(i + j) % N] += a[i] * b[j];
        }
    }
    return res;
}
template <typename T>
std::vector<T> circ_shift(const std::vector<T> &a, int sh) {
    std::vector<T> b(a.size());
    for (int i = 0; i < a.size(); i++) {
        b[i] = a[(i + sh) % a.size()];
    }
    return b;
}


template <typename T>
std::vector<T> trans_conv(const std::vector<T> &a, const std::vector<T> &b, const std::vector<int> &inv_ind) {
    assert(a.size() == b.size());
    assert(inv_ind.size() == b.size());
    assert(in_range<int>(inv_ind, 0, a.size() - 1));
    const int N = a.size();
    std::vector<T> res(N, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res[(i + j) % N] += a[i] * b[j];
        }
    }
    return res;
}

template <class seq_type, class embed_type>
void tensor_disc_sketch(const Seq<seq_type> &seq,
                        Vec2D<embed_type> &embedding,
                        const TensorParams &params) {
    embedding = Vec2D<embed_type>(params.embed_dim);
    for (int m = 0; m < params.embed_dim; m++) {
        auto cnt = new2D<double>(params.tup_len, params.num_phases, 0);
        for (int i = 0; i < seq.size(); i++) {
            for (int t = params.tup_len - 1; t >= 1; t--) {
                auto pi = params.iphase[m][t][seq[i]];
                for (int p = 0; p < params.num_phases; p++) {
                    auto shift = (p + pi) % params.num_phases;
                    cnt[t][shift] += cnt[t - 1][p];
                }
            }
            auto pi = params.iphase[m][0][seq[i]];
            cnt[0][pi]++;
        }
        const auto &top_cnt = cnt[params.tup_len - 1];
        double norm = l1(top_cnt);
        for (const auto c : top_cnt) {
            embedding[m].push_back(c / norm);
        }
    }
}

template <class seq_type, class embed_type>
Vec3D<embed_type> tensor_disc_slide(const Seq<seq_type> &seq, const TensorSlideParams &params) {
    Vec3D<embed_type> embedding = Vec3D<embed_type>(params.embed_dim, Vec2D<embed_type>());
    for (int m = 0; m < params.embed_dim; m++) {
        Vec3D<embed_type> cnt
                = new3D<embed_type>(params.tup_len, params.tup_len, params.num_phases, 0);
        for (int i = 0; i < seq.size(); i++) {
            int j = i - params.win_len;
            if (j >= 0) {
                for (int t = 0; t < params.tup_len; t++) {
                    auto pj = params.iphase[m][t][seq[j]];
                    cnt[t][t][pj]--;
                    for (int t2 = t - 1; t2 >= 0; t2--) {
                        auto pj = params.iphase[m][t2][seq[j]];
                        //                            auto csh = circ_shift(cnt[t2+1][t], pj);
                        //                            cnt[t2][t] -= csh;
                        for (int p = 0; p < params.num_phases; p++) {
                            auto shift = (p + pj) % params.num_phases;
                            cnt[t2][t][shift] -= cnt[t2 + 1][t][p];
                        }
                    }
                }
            }

            for (int t = 0; t < params.tup_len; t++) {
                for (int t2 = params.tup_len - 1; t2 > t; t2--) {
                    auto pi = params.iphase[m][t2][seq[i]];
                    //                        auto csh = circ_shift(cnt[t][t2-1], pi);
                    //                        cnt[t][t2] += csh;
                    for (int p = 0; p < params.num_phases; p++) {
                        auto shift = (p + pi) % params.num_phases;
                        cnt[t][t2][shift] += cnt[t][t2 - 1][p];
                    }
                }
                auto pi = params.iphase[m][t][seq[i]];
                cnt[t][t][pi]++;
            }
            if ((i + 1) % params.stride == 0 or i == (seq.size() - 1)) {
                auto top_cnt = cnt[0][params.tup_len - 1];
                embed_type norm = l1(top_cnt);
                for (auto &c : top_cnt) {
                    c = norm == 0 ? 0 : (c / norm);
                }
                embedding[m].push_back(top_cnt);
            }
        }
    }
    return embedding;
}


template <class seq_type, class embed_type>
Vec3D<embed_type> tensor_disc_slide2(const Vec2D<seq_type> &seq, const TensorSlideParams &params) {
    assert(params.num_phases == params.alphabet_size);
    Vec3D<embed_type> embedding = Vec3D<embed_type>(params.embed_dim, Vec2D<embed_type>());
    for (int m = 0; m < params.embed_dim; m++) {
        Vec3D<embed_type> cnt
                = new3D<embed_type>(params.tup_len, params.tup_len, params.num_phases, 0);
        for (int i = 0; i < seq.size(); i++) {
            for (int t = 0; t < params.tup_len; t++) {
                for (int t2 = params.tup_len - 1; t2 > t; t2--) {
                    const auto &iph = params.iphase[m][t2];
                    auto conv = trans_conv(cnt[t][t2 - 1], seq[i], iph);
                    cnt[t][t2] += conv;
                    assert(in_range3<embed_type>(cnt));
                    //                        for (int p = 0; p < params.num_phases; p++) {
                    //                            auto shift = (p + iph[p]) % params.num_phases;
                    //                            cnt[t][t2][shift] += seq[i][p] * cnt[t][t2 -
                    //                            1][p]; assert(cnt[t][t2][shift] >= 0);
                    //                            // assert(in_range3(cnt));
                    //                        }
                }
                for (int p = 0; p < params.num_phases; p++) {
                    auto pi = params.iphase[m][t][p];
                    cnt[t][t][pi] += seq[i][p];
                    assert(cnt[t][t][pi] >= 0);
                    assert(in_range3<embed_type>(cnt));
                }
            }

            int j = i - params.win_len;
            if (j >= 0) {
                for (int t = 0; t < params.tup_len; t++) {
                    for (int p = 0; p < params.num_phases; p++) {
                        auto pj = params.iphase[m][t][p];
                        cnt[t][t][pj] -= seq[j][p];
                        assert(in_range3<embed_type>(cnt));
                    }
                    for (int t2 = t - 1; t2 >= 0; t2--) {
                        const auto &iph = params.iphase[m][t2];
                        auto conv = trans_conv(cnt[t2 + 1][t], seq[j], iph);
                        cnt[t2][t] -= conv;
                        assert(in_range3<embed_type>(cnt));
                        //                            for (int p = 0; p < params.num_phases; p++) {
                        //                                for (int p2 = 0; p2 < params.num_phases;
                        //                                p2++) {
                        //                                    auto shift = (p + iph[p2]) %
                        //                                    params.num_phases; cnt[t2][t][shift]
                        //                                    -= cnt[t2 + 1][t][p] * seq[j][p2];
                        //                                    assert(cnt[t2][t][shift] >= 0);
                        //                                    // assert(in_range3(embedding));
                        //                                }
                        //                            }
                    }
                }
            }
            if ((i + 1) % params.stride == 0 or i == (seq.size() - 1)) {
                std::vector<embed_type> top_cnt = cnt[0][params.tup_len - 1];
                embed_type norm = l1(top_cnt);
                for (auto &c : top_cnt) {
                    c = norm == 0 ? 0 : (c / norm);
                }
                assert(in_range3<embed_type>(cnt));
                embedding[m].push_back(top_cnt);
            }
        }
    }
    return embedding;
}
template <class seq_type, class embed_type>
Vec3D<embed_type> tensor_disc_slide3(const Vec3D<seq_type> &seq,
                                     const std::vector<TensorSlideParams> &params) {
    assert(seq.size() == params.size());
    Vec3D<embed_type> embedding3 = Vec3D<embed_type>();
    for (int m = 0; m < seq.size(); m++) {
        Vec3D<embed_type> embedding2 = tensor_disc_slide2<seq_type, embed_type>(seq[m], params[m]);
        for (auto &e : embedding2)
            embedding3.push_back(e);
    }
    return embedding3;
}

template <typename T>
std::vector<T> squeeze_tensor(const Vec3D<T> &a3) {
    std::vector<T> out;
    for (const auto &a2 : a3)
        for (const auto &a1 : a2) {
            for (auto v : a1) {
                out.push_back(v);
            }
        }
    return out;
}

} // namespace ts

#endif // SEQUENCE_SKETCHING_TENSOR_DISC_HPP
