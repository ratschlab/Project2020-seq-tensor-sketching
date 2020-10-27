//
// Created by Amir Joudaki on 6/18/20.
//

#ifndef SEQUENCE_SKETCHING_WMINHASH_HPP
#define SEQUENCE_SKETCHING_WMINHASH_HPP
#include "args.hpp"
#include <iostream>
#include <random>

namespace SeqSketch {
    using namespace BasicTypes;

    struct WMHParams {
        size_t embed_dim, sig_len, max_len;
        Vec2D<size_t> perms;

        void rand_init() {
            std::random_device rd;
            auto eng = std::mt19937(rd());
            perms = Vec2D<size_t>(embed_dim, Vec<size_t>(sig_len * max_len, 0));
            for (int m = 0; m < embed_dim; m++) {
                std::iota(perms[m].begin(), perms[m].end(), 0);
                std::shuffle(perms[m].begin(), perms[m].end(), eng);
            }
        }
    };
    template<class seq_type>
    void weighted_minhash(const Seq<seq_type> &seq, Vec<seq_type> &embed, const WMHParams &params) {
        embed = Vec<seq_type>(params.embed_dim);
        for (int m = 0; m < params.embed_dim; m++) {
            seq_type min_char;
            size_t min_rank = params.sig_len + 1;
            Vec<int> cnts(params.sig_len, 0);
            for (auto s : seq) {
                auto r = params.perms[m][s + cnts[s] * params.sig_len];
                cnts[s]++;
                if (r < min_rank) {
                    min_rank = r;
                    min_char = s;
                }
            }
            embed[m] = min_char;
        }
    }
}// namespace SeqSketch

#endif//SEQUENCE_SKETCHING_WMINHASH_HPP
