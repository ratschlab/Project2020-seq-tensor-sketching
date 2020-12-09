#pragma once

#include "util/utils.hpp"

namespace ts { // ts = Tensor Sketch
template <class T>
auto new2D(int d1, int d2, T val = 0) {
    return Vec2D<T>(d1, std::vector<T>(d2, val));
}
template <class T>
auto new3D(int d1, int d2, int d3, T val = 0) {
    return Vec3D<T>(d1, new2D(d2, d3, val));
}

// utility functions for len^D-dimensional tensor
template <class T, class = is_u_integral<T>>
bool increment_sub(std::vector<T> &sub, T len) {
    sub[0]++;
    T i = 0;
    while (sub[i] >= len) {
        sub[i++] = 0;
        if (i >= sub.size())
            return false;
        sub[i]++;
    }
    return true;
}

template <class T, class = is_u_integral<T>>
T sub2ind(const std::vector<T> &sub, T len) {
    T ind = 0, coef = 1;
    for (size_t i = 0; i < sub.size(); i++) {
        ind += sub[i] * coef;
        coef *= len;
    }
    return ind;
}

} // namespace ts
