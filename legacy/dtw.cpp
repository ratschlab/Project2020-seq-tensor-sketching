//
// Created by Amir Joudaki on 6/16/20.
//

#ifndef SEQUENCE_SKETCHING_DTW_H
#define SEQUENCE_SKETCHING_DTW_H

#include <vector>
// TODO test skeching with l1 over vectors

template <class T, class result_type>
void dtw(std::vector<T> &a, const std::vector<T> &b) {
    int n = a.size();
    int m = b.size();
    std::vector<std::vector<T>> DTW(n + 1, std::vector<T>(m, 0));

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            T cost = std::abs(a[i] - b[j]);
            DTW[i][j] = cost + std::min({ DTW[i - 1][j], DTW[i][j - 1], DTW[i - 1][j - 1] });
        }
    }
    return DTW[n][m];
}

#endif // SEQUENCE_SKETCHING_DTW_H
