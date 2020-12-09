////
//// Created by Amir Joudaki on 6/19/20.
////
//
//#ifndef SEQUENCE_SKETCHING_TENSOR_SLIDE2_H
//#define SEQUENCE_SKETCHING_TENSOR_SLIDE2_H
//
//#include "sketch/tensor_slide.hpp"
//
// namespace ts { // ts = Tensor Sketch
//
//
//    template<class seq_type, class embed_type>
//    void tensor_sketch_slide2(const std::vector<Seq<seq_type>> &seq2D, Vec2D<embed_type> &embedding, const
//    TensorSlideParams &params) {
//        assert(seq2D.size() == params.embed_dim);
//        embedding = Vec2D<embed_type>(params.embed_dim, std::vector<embed_type>());
//        for (int m = 0; m < params.embed_dim; m++) {
//            const auto &seq = seq2D[m];
//            auto cnt = new3D<float>(params.tup_len, params.tup_len, params.num_phases, 0);
//            for (int i = 0; i < seq.size(); i++) {
//                int j = i - params.win_len;
//                if (j >= 0) {
//                    for (int t = 0; t < params.tup_len; t++) {
//                        auto pj = params.iphases[m][t][seq[j]];
//                        cnt[t][t][pj]--;
//                        for (int t2 = t - 1; t2 >= 0; t2--) {
//                            auto pj = params.iphases[m][t2][seq[j]];
//                            for (int p = 0; p < params.num_phases; p++) {
//                                auto shift = (p + pj) % params.num_phases;
//                                cnt[t2][t][shift] -= cnt[t2 + 1][t][p];
//                            }
//                        }
//                    }
//                }
//
//                for (int t = 0; t < params.tup_len; t++) {
//                    for (int t2 = params.tup_len - 1; t2 > t; t2--) {
//                        auto pi = params.iphases[m][t2][seq[i]];
//                        for (int p = 0; p < params.num_phases; p++) {
//                            auto shift = (p + pi) % params.num_phases;
//                            cnt[t][t2][shift] += cnt[t][t2 - 1][p];
//                        }
//                    }
//                    auto pi = params.iphases[m][t][seq[i]];
//                    cnt[t][t][pi]++;
//                }
//                const auto &top_cnt = cnt[0][params.tup_len - 1];
//                auto prod = std::inner_product(params.icdf[m].begin(), params.icdf[m].end(),
//                top_cnt.begin(), (double) 0); auto norm = l1(top_cnt); prod = prod / norm;
//                embed_type bin = std::upper_bound(params.bins.begin(), params.bins.begin() +
//                params.num_bins, prod) - params.bins.begin(); if ((i + 1) % params.stride == 0 or
//                i == (seq.size() - 1)) {
//                    if (norm != 0)
//                        embedding[m].push_back(bin);
//                    else
//                        embedding[m].push_back(params.num_bins / 2);
//                }
//            }
//        }
//    }
//
//}
//
//#endif//SEQUENCE_SKETCHING_TENSOR_SLIDE2_H
