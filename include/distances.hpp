//
// Created by Amir Joudaki on 6/18/20.
//

#ifndef SEQUENCE_SKETCHING_DISTANCES_HPP
#define SEQUENCE_SKETCHING_DISTANCES_HPP

#include "args.hpp"

namespace SeqSketch {
    using namespace BasicTypes;
    template<class T>
    T l1(const Vec<T> &vec) {
        T sum = 0;
        for (auto v : vec) sum += (v >= 0) ? v : -v;
        return sum;
    }

    template<class T>
    T l1_dist(const Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        T res = 0;
        for (int i = 0; i < a.size(); i++)
            res += (a[i] - b[i] >= 0) ? (a[i] - b[i]) : (b[i] - a[i]);
        return res;
    }

    template<class T>
    T l1_dist2D(const Vec2D<T> &a, const Vec2D<T> &b) {
        assert(a.size() == b.size());
        T res = 0;
        for (int i = 0; i < a.size(); i++)
            res += l1_dist(a[i], b[i]);
        return res;
    }


    //    template<class T>
    //    T l1_dist2D_minlen(const Vec2D<T> &a, const Vec2D<T> &b) {
    //        assert(a.size() == b.size());
    //        auto len = std::min(a.size(), b.size());
    //        T result = 0;
    //        for (size_t i = 0; i < len; i++) {
    //            result += l1_dist_minlen(a[i], b[i]);
    //        }
    //        return result;
    //    }

    template<class T>
    T l1_dist2D_mean(const Vec2D<T> &a, const Vec2D<T> &b) {
        int len = a[0].size();
        T diff;
        for (int j = 0; j < len; j++) {
            T A = 0, B = 0;
            for (int i = 0; i < a.size(); i++) {
                A += (double) a[i][j] / a.size();
            }
            for (int i = 0; i < b.size(); i++) {
                B += (double) b[i][j] / b.size();
            }
            diff += (A - B) ? (A - B) : (B - A);
        }
        return diff;
    }

    template<class T>
    T l2_sq(const Vec<T> &vec) {
        T sum = 0;
        for (auto v : vec) sum += v * v;
        return sum;
    }

    template<class T>
    T l2_sq_dist(const Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        T sum = 0;
        for (int i = 0; i < a.size(); i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sum;
    }

    template<class T>
    T l1_dist2D_minlen(const Vec2D<T> &a, const Vec2D<T> &b) {
        auto len = std::min(a.size(), b.size());
        T val = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < a[i].size() and j < b[i].size(); j++) {
                val += (a[i][j] - b[i][j]) * ((a[i][j] - b[i][j] > 0) ? 1 : -1);
            }
        }
        return val;
    }

    template<class T>
    T l2_dist2D_minlen(const Vec2D<T> &a, const Vec2D<T> &b) {
        auto len = std::min(a.size(), b.size());
        T val = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < a[i].size() and j < b[i].size(); j++) {
                val += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]);
            }
        }
        return val;
    }


    template<class T>
    T ip_sim(const Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        T sum = 0;
        for (int i = 0; i < a.size(); i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    template<class T>
    T cosine_sim(const Vec<T> &a, const Vec<T> &b) {
        T val = ip_sim(a, b);
        val = val * val / l2_sq(a) / l2_sq(b);
        return val;
    }

    template<class T>
    T median(Vec<T> v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size()];
    }

    template<class T>
    T median_dist(const Vec<T> &a, const Vec<T> &b) {
        //        Vec<T> res;
        //        res.reserve(a.size());
        //        std::transform(a.begin(), a.end(), b.begin(), res.begin(), std::minus<T>());
        auto res = a - b;
        std::transform(res.begin(), res.end(), res.begin(), [](const T a) { return (a > 0) ? a : -a; });
        std::sort(res.begin(), res.end());
        std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
        return res[res.size() / 2];
    }

    template<class T>
    T hamming_dist(const Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        T diff = 0;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) {
                diff++;
            }
        }
        return diff;
    }

    template<class T>
    T hamming_dist2D(const Vec2D<T> &a, const Vec2D<T> &b) {
        assert(a.size() == b.size());
        T diff = 0;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) {
                diff++;
            }
        }
        return diff;
    }

    template<class seq_type>
    int lcs(const Seq<seq_type> &s1, const Seq<seq_type> &s2) {
        size_t m = s1.size();
        size_t n = s2.size();
        //        int L[m + 1][n + 1];
        Vec2D<int> L(m + 1, Vec<int>(n + 1, 0));
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
    size_t edit_distance(const std::string &s1, const std::string &s2) {
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
}// namespace SeqSketch
#endif//SEQUENCE_SKETCHING_DISTANCES_HPP
