//
// Created by Amir Joudaki on 6/18/20.
//

#ifndef SEQUENCE_SKETCHING_TENSOR_HPP
#define SEQUENCE_SKETCHING_TENSOR_HPP

#include <cmath>
#include <cstdio>

#include "subindex.h"

namespace SeqSketch {


    struct TensorParams {
        int sig_len;
        int embed_dim;
        int num_phases;
        int num_bins;
        int tup_len;

        Vec3D<int> iphase;
        Vec2D<double> icdf;
        Vec<double> bins;

        virtual void rand_init() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> unif_iphase(0, num_phases - 1);
            std::uniform_real_distribution<double> unif(0, 1);
            double pie = std::atan(1) * 2;

            iphase = new3D<int>(embed_dim, tup_len, sig_len);
            icdf = new2D<double>(embed_dim, num_phases);
            for (int m = 0; m < embed_dim; m++) {
                double bias = 0;
                for (int t = 0; t < tup_len; t++) {
                    bias += unif(gen);
                    for (int c = 0; c < sig_len; c++) {
                        iphase[m][t][c] = unif_iphase(gen);
                    }
                }
                for (int p = 0; p < num_phases; p++) {
                    icdf[m][p] = 1 - 2 * unif(gen);    // use random sign (-1) or (1)
                    icdf[m][p] = (p % 2 == 0) ? 1 : -1;// use oddity of p to assign (-1) or (1)
                }
            }
            bins = Vec<double>(num_bins);
            for (int b = 0; b < num_bins; b++) {
                bins[b] = std::tan(pie * (((double) b + .5) / num_bins - .5));
            }
            bins.push_back(std::numeric_limits<double>::max());
            bins.insert(bins.begin(), -std::numeric_limits<double>::min());
        }
    };


    template<class seq_type, class embed_type>
    void tensor_sketch(const Seq<seq_type> &seq, Vec<embed_type> &embedding, const TensorParams &params) {
        start_timer("tensor_sketch");
        embedding = Vec<embed_type>(params.embed_dim, 0);
        for (int m = 0; m < params.embed_dim; m++) {
            auto cnt = new2D<double>(params.tup_len + 1, params.num_phases, 0);
            cnt[0][0] = 1;// base case
            for (int i = 0; i < seq.size(); i++) {
                for (int t = params.tup_len - 1; t >= 0; t--) {
                    auto pi = params.iphase[m][t][seq[i]];
                    for (int p = 0; p < params.num_phases; p++) {
                        auto shift = (p + pi) % params.num_phases;
                        cnt[t + 1][shift] += cnt[t][p];
                    }
                }
            }
            const auto &top_cnt = cnt[params.tup_len];
            auto prod = std::inner_product(params.icdf[m].begin(), params.icdf[m].end(), top_cnt.begin(), (double) 0);
            double norm = l1(top_cnt);
            prod = prod / norm;
            //            int exp;
            //            frexp(prod, &exp);
            //            embedding[m]= exp * sgn(prod);
            embed_type bin = std::upper_bound(params.bins.begin(), params.bins.begin() + params.num_bins, prod) - params.bins.begin();
            if (params.num_bins == 0) {
                embedding[m] = prod;
            } else {
                embedding[m] = bin;
            }
        }
        stop_timer();
    }


}// namespace SeqSketch

#endif//SEQUENCE_SKETCHING_TENSOR_HPP
