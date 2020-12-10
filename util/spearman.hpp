#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Function returns the rank vector of a set of observations v
template <typename T>
std::vector<double> rankify(std::vector<T> &v) {
    std::vector<double> result(v.size());

    for (size_t i = 0; i < v.size(); i++) {
        size_t r = 1, s = 1;

        for (size_t j = 0; j < v.size(); j++) {
            if (v[j] < v[i])
                r++;
            if (v[j] == v[i])
                s++;
        }

        // Use Fractional Rank formula fractional_rank = r + (n-1)/2
        result[i] = r + (s - 1) * 0.5;
    }

    return result;
}

/* Compute the Pearson correlation coefficient of a and b */
template <typename T>
double pearson(std::vector<T> &a, std::vector<T> &b) {
    T sum_a = 0, sum_b = 0, sum_ab = 0;
    T square_sum_a = 0, square_sum_b = 0;

    for (size_t i = 0; i < a.size(); i++) {
        // sum of elements of array X.
        sum_a = sum_a + a[i];

        // sum of elements of array Y.
        sum_b = sum_b + b[i];

        // sum of X[i] * Y[i].
        sum_ab = sum_ab + a[i] * b[i];

        // sum of square of array elements.
        square_sum_a = square_sum_a + a[i] * a[i];
        square_sum_b = square_sum_b + b[i] * b[i];
    }

    // use formula for calculating correlation coefficient.
    return (a.size() * sum_ab - sum_a * sum_b)
            / std::sqrt((a.size() * square_sum_a - sum_a * sum_a)
                        * (a.size() * square_sum_b - sum_b * sum_b));
}

template <typename T>
double spearman(std::vector<T> &a, std::vector<T> &b) {
    std::vector<double> rank1 = rankify(a);
    std::vector<double> rank2 = rankify(b);
    return pearson(rank1, rank2);
}
