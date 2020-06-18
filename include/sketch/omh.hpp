//
// Created by Amir Joudaki on 6/18/20.
//

#ifndef SEQUENCE_SKETCHING_OMH_HPP
#define SEQUENCE_SKETCHING_OMH_HPP
#include "vectools.hpp"

namespace Sketching {

    struct OMHParams {
        int sig_len;
        int max_seq_len;
        int embed_dim;
        int tup_len;

        MultiVec<int, int> perms;

        void init_rand() {
            std::random_device rd;
            std::mt19937 gen = std::mt19937(rd());
            // Dimensions: #perms X #sig X max-len
            std::vector<int> dims{max_seq_len, sig_len, embed_dim};
            perms.init(dims, 0);
            for (int pi = 0; pi < embed_dim; pi++) {
                std::iota(perms[pi].begin(), perms[pi].end(), 0);
                std::shuffle(perms[pi].begin(), perms[pi].end(), gen);
            }
        }
    };

    template<class seq_type, class embed_type, class size_type = std::size_t>
    void ordered_minhash(const Seq<seq_type> &seq, Vec2D<embed_type> &embed,
                         const OMHParams &params) {
        for (int pi = 0; pi < params.embed_dim; pi++) {
            Vec<size_type> counts(params.sig_len, 0);
            Vec<std::pair<embed_type, size_type>> ranks;
            for (auto s : seq) {
                ranks.push_back({params.perms[pi][s][counts[s]], s});
                counts[s]++;
            }
            std::sort(ranks.begin(), ranks.end());
            Vec<embed_type> tup;
            for (auto pair = ranks.begin(); pair != ranks.begin() + params.tup_len; pair++) {
                tup.push_back(pair->second);
            }
            embed.push_back(tup);
        }
    }
}// namespace Sketching

#endif//SEQUENCE_SKETCHING_OMH_HPP