//
// Created by Amir Joudaki on 6/11/20.
//

#ifndef SEQUENCE_SKETCHING_SKETCH_HPP
#define SEQUENCE_SKETCHING_SKETCH_HPP

#include "types.h"
#include "vec_tools.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>


namespace SeqSketch {
    using namespace Types;
    using namespace VecTools;


    template<class seq_type, class embed_type, class size_type = std::size_t>
    void seq2kmer(const Seq<seq_type> &seq, Vec<embed_type> &vec, size_type kmer_size, size_type sig_len) {
        vec = Vec<embed_type>(seq.size() - kmer_size + 1);
        for (size_type i = 0; i < vec.size(); i++) {
            size_type c = 1;
            for (size_type j = 0; j < kmer_size; j++) {
                vec[i] += c * seq[i + j];
                c *= sig_len;
            }
        }
    }

    struct MinHashParams {
        size_t embed_dim, sig_len;
        Vec2D<size_t> perms;
        MinHashParams(size_t embed_dim, size_t sig_len)
            : embed_dim(embed_dim), sig_len(sig_len) {
            std::random_device rd;
            auto eng = std::mt19937(rd());
            perms = Vec2D<size_t>(embed_dim, Vec<size_t>(sig_len, 0));
            for (int m = 0; m < embed_dim; m++) {
                std::iota(perms[m].begin(), perms[m].end(), 0);
                std::shuffle(perms[m].begin(), perms[m].end(), eng);
            }
        }
    };

    template<class seq_type>
    void minhash(const Seq<seq_type> &seq, Vec<seq_type> &embed, const MinHashParams &params) {
        embed = Vec<seq_type>(params.embed_dim);
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
    }

    struct WeightedMinHashParams {
        size_t embed_dim, sig_len, max_len;
        Vec2D<size_t> perms;
        WeightedMinHashParams(size_t embed_dim, size_t sig_len, size_t max_len)
            : embed_dim(embed_dim), sig_len(sig_len), max_len(max_len) {
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
    void weighted_minhash(const Seq<seq_type> &seq, Vec<seq_type> &embed, const WeightedMinHashParams &params) {
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

    template<class seq_type, class size_type>
    size_type subseq2ind(const Vec<seq_type> &seq, const Vec<size_type> &sub, size_type sig_len) {
        size_type ind = 0, coef = 1;
        for (size_type i = 0; i < sub.size(); i++) {
            ind += seq[sub[i]] * coef;
            coef *= sig_len;
        }
        return ind;
    }

    template<class seq_type, class embed_type, class size_type = std::size_t>
    void tup_embed(const Seq<seq_type> &seq, Vec<embed_type> &embed,
                   size_type sig_len, size_type tup_len) {
        size_type seq_len = seq.size();
        size_type cnt = 0, size = int_pow(sig_len, tup_len);
        embed = Vec<embed_type>(size, 0);
        Vec<size_type> sub(tup_len, 0);
        do {
            if (is_ascending(sub)) {
                auto ind = subseq2ind(seq, sub, sig_len);
                embed[ind]++;
            }
        } while (increment_sub(sub, seq_len));
    }

    struct OMP_Params {
        size_t sig_len,
                max_seq_len,
                embed_dim,
                tup_len;
    };

    template<class size_type = std::size_t>
    MultiVec<size_type, size_type> omp_rand_perms(const OMP_Params &params) {
        std::random_device rd;
        std::mt19937 gen = std::mt19937(rd());

        // Dimensions: #perms X #sig X max-len
        std::vector<size_type> dims{params.max_seq_len, params.sig_len, params.embed_dim};
        MultiVec<size_type, size_type> perms(dims, 0);
        for (int pi = 0; pi < params.embed_dim; pi++) {
            std::iota(perms[pi].begin(), perms[pi].end(), 0);
            std::shuffle(perms[pi].begin(), perms[pi].end(), gen);
        }
        return perms;
    }

    template<class seq_type, class embed_type, class size_type = std::size_t>
    void omp_sketch(const Seq<seq_type> &seq, Vec2D<embed_type> &embed,
                    MultiVec<size_type, size_type> &perms, const OMP_Params &params) {
        for (int pi = 0; pi < params.embed_dim; pi++) {
            Vec<size_type> counts(params.sig_len, 0);
            Vec<std::pair<embed_type, size_type>> ranks;
            for (auto s : seq) {
                ranks.push_back({perms[pi][s][counts[s]], s});
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


    struct TensorParams {
        size_t sig_len,
                embed_dim,
                num_phases,
                num_bins,
                tup_len;

        Vec3D<int> iphase;
        Vec2D<double> icdf;
        Vec<double> bins;

        virtual void rand_init() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> unif_iphase(0, num_phases - 1);
            std::uniform_real_distribution<double> unif(0, 1);
            double pie = std::atan(1) * 4;

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
                    double phase = std::fmod(p + bias, (double) num_phases) / num_phases;
                    icdf[m][p] = std::tan(pie * (phase - 0.5));
                }
            }
            num_bins -= 2;
            bins = Vec<double>(num_bins);
            for (int b = 0; b < num_bins; b++) {
                bins[b] = std::tan(pie * (((double) b + .5) / num_bins - .5));
            }
            bins.push_back(std::numeric_limits<double>::max());
            bins.insert(bins.begin(), -std::numeric_limits<double>::min());
            num_bins += 2;
        }
    };


    template<class seq_type, class embed_type>
    void tensor_sketch(const Seq<seq_type> &seq, Vec<embed_type> &embedding, const TensorParams &params) {
        embedding = Vec<embed_type>(params.embed_dim, 0);
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
            auto prod = std::inner_product(params.icdf[m].begin(), params.icdf[m].end(), top_cnt.begin(), (double) 0);
            double norm = l1(top_cnt);
            prod = prod / norm;
            embed_type bin = std::upper_bound(params.bins.begin(), params.bins.begin() + params.num_bins, prod) - params.bins.begin();
            embedding[m] = bin;
        }
    }

    struct TensorSlideParams : public TensorParams {
        size_t win_len,
                stride;

        void rand_init() {
            TensorParams::rand_init();
        }
    };
    template<class seq_type, class embed_type>
    void tensor_sketch_slide(const Seq<seq_type> &seq, Vec2D<embed_type> &embedding, const TensorSlideParams &params) {
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
                const auto &top_cnt = cnt[0][params.tup_len - 1];
                auto prod = std::inner_product(params.icdf[m].begin(), params.icdf[m].end(), top_cnt.begin(), (double) 0);
                auto norm = l1(top_cnt);
                prod = prod / norm;
                embed_type bin = std::upper_bound(params.bins.begin(), params.bins.begin() + params.num_bins, prod) - params.bins.begin();
                if ((i + 1) % params.stride == 0 or i == (seq.size() - 1)) {
                    if (norm != 0)
                        embedding[m].push_back(bin);
                    else
                        embedding[m].push_back(params.num_bins / 2);
                }
            }
        }
    }

    template<class seq_type, class embed_type>
    void tensor_sketch_slide2(const Vec<Seq<seq_type>> &seq2D, Vec2D<embed_type> &embedding, const TensorSlideParams &params) {
        assert(seq2D.size() == params.embed_dim);
        embedding = Vec2D<embed_type>(params.embed_dim, Vec<embed_type>());
        for (int m = 0; m < params.embed_dim; m++) {
            const auto &seq = seq2D[m];
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
                const auto &top_cnt = cnt[0][params.tup_len - 1];
                auto prod = std::inner_product(params.icdf[m].begin(), params.icdf[m].end(), top_cnt.begin(), (double) 0);
                auto norm = l1(top_cnt);
                prod = prod / norm;
                embed_type bin = std::upper_bound(params.bins.begin(), params.bins.begin() + params.num_bins, prod) - params.bins.begin();
                if ((i + 1) % params.stride == 0 or i == (seq.size() - 1)) {
                    if (norm != 0)
                        embedding[m].push_back(bin);
                    else
                        embedding[m].push_back(params.num_bins / 2);
                }
            }
        }
    }

}// namespace SeqSketch


#endif//SEQUENCE_SKETCHING_SKETCH_HPP
