//
// Created by Amir Joudaki on 6/18/20.
//

#ifndef SEQUENCE_SKETCHING_MINHASH_HPP
#define SEQUENCE_SKETCHING_MINHASH_HPP

namespace SeqSketch {

    struct MHParams {
        size_t embed_dim, sig_len;
        Vec2D<size_t> perms;

        void rand_init() {
            std::random_device rd;
            auto eng = std::mt19937(rd());
            perms = Vec2D<size_t>(embed_dim, Vec<size_t>(sig_len, 0));
            for (int m = 0; m < embed_dim; m++) {
                std::iota(perms[m].begin(), perms[m].end(), 0);
                std::shuffle(perms[m].begin(), perms[m].end(), eng);
            }
        }
    };

    template<class seq_type, class embed_type>
    void minhash(const Seq<seq_type> &seq, Vec<embed_type> &embed, const MHParams &params) {
        Timer::start("minhash");
        embed = Vec<embed_type>(params.embed_dim);
        for (int m = 0; m < params.embed_dim; m++) {
            seq_type min_char;
            size_t min_rank = params.sig_len + 1;
            for (auto s : seq) {
                auto r = params.perms[m][s];
                if (r < min_rank) {
                    min_rank = r;
                    min_char = s;
                }
            }
            embed[m] = min_char;
        }
        Timer::stop();
    }
}// namespace SeqSketch

#endif//SEQUENCE_SKETCHING_MINHASH_HPP
