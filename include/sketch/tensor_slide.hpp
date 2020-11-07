//
// Created by Amir Joudaki on 6/19/20.
//

#ifndef SEQUENCE_SKETCHING_TENSOR_SLIDE_H
#define SEQUENCE_SKETCHING_TENSOR_SLIDE_H

#include "../../cpp/legacy/vectool.hpp"
#include "tensor.hpp"

namespace SeqSketch {

    struct TensorSlideParams : public TensorParams {
        int win_len;
        int stride;
        int offset;
    };

    template<class seq_type, class embed_type>
    void tensor_slide_sketch(const Seq<seq_type> &seq, Vec2D<embed_type> &embedding, const TensorSlideParams &params) {
        start_timer("tensor_slide_sketch");
        embedding = Vec2D<embed_type>(params.embed_dim, Vec<embed_type>());
        for (int m = 0; m < params.embed_dim; m++) {
            auto cnt = new3D<float>(params.tup_len, params.tup_len, params.num_phases, 0);
            for (int i = 0; i < seq.size(); i++) {
                int j = i - params.win_len;
                if (j >= 0) {
                    for (int t = 0; t < params.tup_len; t++) {
                        auto pj = params.iphase[m][t][seq[j]];
                        cnt[t][t][pj]--;
                        for (int t2 = t - 1; t2 >= 0; t2--) {
                            auto pj = params.iphase[m][t2][seq[j]];
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
                        for (int p = 0; p < params.num_phases; p++) {
                            auto shift = (p + pi) % params.num_phases;
                            cnt[t][t2][shift] += cnt[t][t2 - 1][p];
                        }
                    }
                    auto pi = params.iphase[m][t][seq[i]];
                    cnt[t][t][pi]++;
                }
                if (sketch_now(i, seq.size(), params.stride, params.offset)) {
                    const auto &top_cnt = cnt[0][params.tup_len - 1];
                    auto prod = std::inner_product(params.icdf[m].begin(), params.icdf[m].end(), top_cnt.begin(), (double) 0);
                    prod = prod / l1(top_cnt);
                    //                    int exp;
                    //                    frexp(prod, &exp);
                    //                    embedding[m].push_back(exp * sgn(prod));
                    embed_type bin = std::upper_bound(params.bins.begin(), params.bins.begin() + params.num_bins, prod) - params.bins.begin();
                    embedding[m].push_back(bin);
                }
            }
        }
        stop_timer();
    }

}// namespace SeqSketch

#endif//SEQUENCE_SKETCHING_TENSOR_SLIDE_H
