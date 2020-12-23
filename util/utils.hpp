#pragma once

#include "util/multivec.hpp"
#include "util/Timer.hpp"

#include <gflags/gflags.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <cmath>

namespace ts { // ts = Tensor Sketch


template <class T, class = is_u_integral<T>>
T int_pow(T x, T pow) {
    T result = 1;
    for (;;) {
        if (pow & 1)
            result *= x;
        pow >>= 1;
        if (!pow)
            break;
        x *= x;
    }

    return result;
}

/**
 * Extracts k-mers from a sequence. The k-mer is treated as a number in base alphabet_size and then
 * converted to decimal, i.e. the sequence s1...sk is converted to s1*S^(k-1) + s2*S^(k-2) + ... +
 * sk, where k is the k-mer size.
 * @tparam chr types of elements in the sequence
 * @tparam kmer type that stores a kmer
 * @param seq the sequence to extract kmers from
 * @param kmer_size number of characters in a kmer
 * @param alphabet_size size of the alphabet
 * @return the extracted kmers, as integers converted from base #alphabet_size
 */
template <class chr, class kmer>
std::vector<kmer> seq2kmer(const std::vector<chr> &seq, size_t kmer_size, size_t alphabet_size) {
    Timer timer("seq2kmer");
    if (seq.size() < (size_t)kmer_size) {
        return std::vector<kmer>();
    }

    std::vector<kmer> result(seq.size() - kmer_size + 1, 0);
    for (size_t i=0; i<result.size(); i++) {
        size_t c = 1;
        for (size_t j=0; j<kmer_size; j++) {
            result[i] += seq[i+j]*c;
            c *= alphabet_size;
        }
    }

    return result;
}


/***
 * sign function: -1, 0, +1 for negative, 0, and positive values
 * @tparam T type
 * @param val: input value
 * @return sign of val
 */
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <class T>
T l1_dist(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    T res = 0;
    for (size_t i = 0; i < a.size(); i++) {
        auto el = std::abs(a[i] - b[i]);
        res += el;
    }
    return res;
}

template <class T>
T l1_dist2D(const Vec2D<T> &a, const Vec2D<T> &b) {
    assert(a.size() == b.size());
    T res = 0;
    for (int i = 0; i < a.size(); i++)
        res += l1_dist(a[i], b[i]);
    return res;
}


template <class T>
T l1_dist2D_mean(const Vec2D<T> &a, const Vec2D<T> &b) {
    int len = a[0].size();
    T diff;
    for (int j = 0; j < len; j++) {
        T A = 0, B = 0;
        for (int i = 0; i < a.size(); i++) {
            A += (double)a[i][j] / a.size();
        }
        for (int i = 0; i < b.size(); i++) {
            B += (double)b[i][j] / b.size();
        }
        diff += (A - B) ? (A - B) : (B - A);
    }
    return diff;
}

template <class T>
T l2_sq(const std::vector<T> &vec) {
    T sum = 0;
    for (auto v : vec)
        sum += v * v;
    return sum;
}

template <class T>
T l2_sq_dist(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    T sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

template <class T>
T l1_dist2D_minlen(const Vec2D<T> &a, const Vec2D<T> &b) {
    auto len = std::min(a.size(), b.size());
    T val = 0;
    for (size_t i = 0; i < len; i++) {
        for (size_t j = 0; j < a[i].size() and j < b[i].size(); j++) {
            auto el  = std::abs(a[i][j] - b[i][j]);
            val += el;
        }
    }
    return val;
}

template <class T>
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


template <class T>
T ip_sim(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    T sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <class T>
T cosine_sim(const std::vector<T> &a, const std::vector<T> &b) {
    T val = ip_sim(a, b);
    val = val * val / l2_sq(a) / l2_sq(b);
    return val;
}

template <class T>
T median(std::vector<T> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size()];
}

template <class T>
T median_dist(const std::vector<T> &a, const std::vector<T> &b) {
    auto res = a - b;
    std::transform(res.begin(), res.end(), res.begin(), [](const T a) { return (a > 0) ? a : -a; });
    std::sort(res.begin(), res.end());
    std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
    return res[res.size() / 2];
}

template <class T>
T hamming_dist(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    T diff = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            diff++;
        }
    }
    return diff;
}

template <class T>
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

template <class seq_type>
int lcs(const std::vector<seq_type> &s1, const std::vector<seq_type> &s2) {
    size_t m = s1.size();
    size_t n = s2.size();
    //        int L[m + 1][n + 1];
    Vec2D<int> L(m + 1, std::vector<int>(n + 1, 0));
    for (size_t i = 0; i <= m; i++) {
        for (size_t j = 0; j <= n; j++) {
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

template <class seq_type>
size_t lcs_distance(const std::vector<seq_type> &s1, const std::vector<seq_type> &s2) {
    return s1.size() + s2.size() - 2 * lcs(s1, s2);
}

template <class seq_type>
size_t edit_distance(const std::vector<seq_type> &s1, const std::vector<seq_type> &s2) {
    Timer timer("edit_distance");
    const size_t m(s1.size());
    const size_t n(s2.size());

    if (m == 0)
        return n;
    if (n == 0)
        return m;

    auto costs = std::vector<size_t>(n + 1);

    for (size_t k = 0; k <= n; k++)
        costs[k] = k;

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



std::string flag_values();


std::string legacy_config();

} // namespace ts
