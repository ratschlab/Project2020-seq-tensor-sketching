#pragma once

#include <type_traits>
#include <vector>

namespace ts { // ts = Tensor Sketch

template <class T>
using is_u_integral = typename std::enable_if<std::is_unsigned<T>::value>::type;

template <class T>
using Vec2D = std::vector<std::vector<T>>;

template <class T>
using Vec3D = std::vector<Vec2D<T>>;

template <class T>
using Vec4D = std::vector<Vec3D<T>>;


template <class T>
auto new2D(int d1, int d2, T val = 0) {
    return Vec2D<T>(d1, std::vector<T>(d2, val));
}
template <class T>
auto new3D(int d1, int d2, int d3, T val = 0) {
    return Vec3D<T>(d1, new2D(d2, d3, val));
}

} // namespace ts
