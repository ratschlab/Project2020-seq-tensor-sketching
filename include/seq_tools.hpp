//
// Created by Amir Joudaki on 6/9/20.
//
#ifndef SEQUENCE_SKETCHING_UTILS_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <random>

#include "common.hpp"

namespace Sketching {
    using namespace Types;
    using string = std::string;

    struct SeqGen {
        std::random_device rd;
        std::mt19937 gen = std::mt19937(rd());

        int sig_len;
        bool fix_len;
        int max_num_blocks;
        int min_num_blocks;
        int num_seqs;
        int seq_len;
        float mutation_rate;
        float block_mutate_rate;


        template<class T>
        void block_permute(Seq<T> &seq) {
            std::uniform_real_distribution<double> mute(0, 1);
            if (mute(gen) > block_mutate_rate) { return; }
            std::uniform_int_distribution<T> unif(0, sig_len - 1), blocks(min_num_blocks, max_num_blocks);
            int num_blocks = blocks(gen);
            Vec<Index> perm(num_blocks);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), gen);
            while (seq.size() % num_blocks != 0) {// make length divisible by num_blocks
                seq.push_back(unif(gen));
            }
            Seq<T> res(seq.size());
            Index block_size = seq.size() / num_blocks;
            for (size_t i = 0; i < block_size; i++) {
                for (int pi = 0; pi < num_blocks; pi++) {
                    Index bi = pi * block_size + i,
                          bj = perm[pi] * block_size + i;
                    res[bj] = seq[bi];
                }
            }
            seq = res;
            //            res.swap(seq);
        }

        template<class T>
        void gen_seq(Seq<T> &seq) {
            seq.clear();
            std::uniform_int_distribution<T> unif(0, sig_len - 1);
            for (int i = 0; i < seq_len; i++) {
                seq.push_back(unif(gen));
            }
        }

        template<class T>
        void point_mutate(const Seq<T> &ref, Seq<T> &seq) {
            std::discrete_distribution<int> mut{1 - mutation_rate, mutation_rate / 3, mutation_rate / 3, mutation_rate / 3};
            std::uniform_int_distribution<T> unif(0, sig_len - 2);
            // TODO: add alignment
            for (auto i = 0; i < ref.size(); i++) {
                switch (mut(gen)) {
                    case 0: {// no mutation
                        seq.push_back(ref[i]);
                        break;
                    }
                    case 1: {// insert
                        seq.push_back(unif(gen));
                        i--;// init_tensor_slide_params negate the increment
                        break;
                    }
                    case 2: {// delete
                        break;
                    }
                    case 3: {// substitute
                        auto c = unif(gen);
                        c = (c >= ref[i]) ? c + 1 : c;// increment if not changed
                        seq.push_back(c);
                        break;
                    }
                }
            }
        }

        template<class T>
        void make_fix_len(Seq<T> &seq) {
            std::uniform_int_distribution<T> unif(0, sig_len - 1), blocks(min_num_blocks, max_num_blocks);
            if (seq.size() > seq_len) {
                seq = Seq<T>(seq.begin(), seq.end());
            } else if (seq.size() < seq_len) {
                while (seq.size() < seq_len) {
                    seq.push_back(unif(gen));
                }
            }
        }

        template<class T>
        void gen_seqs(Vec<Seq<T>> &seqs) {
            seqs = Vec2D<T>(num_seqs, Vec<T>());
            gen_seq(seqs[0]);
            for (int si = 1; si < num_seqs; si++) {
                point_mutate(seqs[si - 1], seqs[si]);
                block_permute(seqs[si]);
                if (fix_len)
                    make_fix_len(seqs[si]);
            }
        }
    };
    template<class seq_type>
    void print_seq(const Seq<seq_type> &seq, string alpha = "ACGT") {
        std::cout << "seq = ";
        for (auto s : seq) {
            std::cout << alpha[s];
        }
        std::cout << "\n";
    }
    template<class seq_type>
    string seq2string(const Seq<seq_type> &seq, string alpha = "ACGT") {
        string str = "";
        for (auto s : seq) {
            str += alpha[s];
        }
        return str;
    }
    template<class seq_type>
    int lcs(const Seq<seq_type> &s1, const Seq<seq_type> &s2) {
        size_t m = s1.size();
        size_t n = s2.size();
        int L[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 || j == 0) {
                    L[i][j] = 0;
                } else if (s1[i - 1] == s2[j - 1]) {
                    L[i][j] = L[i - 1][j - 1] + 1;
                } else {
                    L[i][j] = std::max(L[i - 1][j], L[i][j - 1]);
                }
            }
        }
        return L[m][n];
    }
    template<class seq_type>
    size_t lcs_distance(const Seq<seq_type> &s1, const Seq<seq_type> &s2) {
        return s1.size() + s2.size() - 2 * lcs(s1, s2);
    }
    template<class seq_type>
    size_t edit_distance(const Seq<seq_type> &s1, const Seq<seq_type> &s2) {
        const size_t m(s1.size());
        const size_t n(s2.size());

        if (m == 0) return n;
        if (n == 0) return m;

        auto costs = Seq<size_t>(n + 1);

        for (size_t k = 0; k <= n; k++) costs[k] = k;

        size_t i = 0;
        for (auto it1 = s1.begin(); it1 != s1.end(); ++it1, ++i) {
            costs[0] = i + 1;
            size_t corner = i;

            size_t j = 0;
            for (auto it2 = s2.begin(); it2 != s2.end(); ++it2, ++j) {
                size_t upper = costs[j + 1];
                if (*it1 == *it2) {
                    costs[j + 1] = corner;
                } else {
                    size_t t(upper < corner ? upper : corner);
                    costs[j + 1] = (costs[j] < t ? costs[j] : t) + 1;
                }

                corner = upper;
            }
        }

        size_t result = costs[n];

        return result;
    }
    size_t edit_distance(const string &s1, const string &s2) {
        const size_t m(s1.size());
        const size_t n(s2.size());

        if (m == 0) return n;
        if (n == 0) return m;

        size_t *costs = new size_t[n + 1];

        for (size_t k = 0; k <= n; k++) costs[k] = k;

        size_t i = 0;
        for (std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>::const_iterator it1 = s1.begin(); it1 != s1.end(); ++it1, ++i) {
            costs[0] = i + 1;
            size_t corner = i;

            size_t j = 0;
            for (std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>::const_iterator it2 = s2.begin(); it2 != s2.end(); ++it2, ++j) {
                size_t upper = costs[j + 1];
                if (*it1 == *it2) {
                    costs[j + 1] = corner;
                } else {
                    size_t t(upper < corner ? upper : corner);
                    costs[j + 1] = (costs[j] < t ? costs[j] : t) + 1;
                }

                corner = upper;
            }
        }

        size_t result = costs[n];
        delete[] costs;

        return result;
    }

}// namespace Sketching


#define SEQUENCE_SKETCHING_UTILS_H

#endif//SEQUENCE_SKETCHING_UTILS_H
