//
// Created by Amir Joudaki on 6/18/20.
//

#ifndef SEQUENCE_SKETCHING_TENSOR_HPP
#define SEQUENCE_SKETCHING_TENSOR_HPP

namespace Sketching {


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
        int win_len;
        int stride;
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


}// namespace Sketching

#endif//SEQUENCE_SKETCHING_TENSOR_HPP
