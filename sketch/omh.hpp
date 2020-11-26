#pragma once

#include "util/utils.hpp"

namespace ts { // ts = Tensor Sketch

struct OMHParams {
    int alphabet_size;
    int max_len;
    int embed_dim;
    int tup_len;

    Vec2D<int> perms;

    void init_rand() {
        std::random_device rd;
        auto gen = std::mt19937(rd());
        // Dimensions: #perms X #sig X max-len
        std::vector<int> dims { max_len, alphabet_size, embed_dim };
        int total_len = alphabet_size * max_len;
        perms = Vec2D<int>(embed_dim, Vec<int>(total_len, 0));
        //            perms.init(dims, 0);
        for (int pi = 0; pi < embed_dim; pi++) {
            std::iota(perms[pi].begin(), perms[pi].end(), 0);
            std::shuffle(perms[pi].begin(), perms[pi].end(), gen);
        }
    }
};

template <class seq_type, class embed_type, class size_type = std::size_t>
void ordered_minhash(const Seq<seq_type> &seq, Vec2D<embed_type> &embed, const OMHParams &params) {
    for (int pi = 0; pi < params.embed_dim; pi++) {
        Vec<size_type> counts(params.alphabet_size, 0);
        Vec<std::pair<embed_type, size_type>> ranks;
        for (auto s : seq) {
            ranks.push_back({ params.perms[pi][s + params.alphabet_size * counts[s]], s });
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

template <class seq_type, class embed_type, class size_type = std::size_t>
void ordered_minhash_flat(const Seq<seq_type> &seq,
                          Vec<embed_type> &embed,
                          const OMHParams &params) {
    Timer::start("ordered_minhash_flat");
    Vec2D<embed_type> embed2D;
    ordered_minhash(seq, embed2D, params);
    for (const auto &tuple : embed2D) {
        int sum = 0;
        for (const auto &item : tuple) {
            sum = sum * params.alphabet_size + item;
        }
        embed.push_back(sum);
        //            for (const auto & item : col) {
        //                embed.push_back(item);
        //            }
    }

    //        embed.push_back(num);
    Timer::stop();
}
} // namespace ts
