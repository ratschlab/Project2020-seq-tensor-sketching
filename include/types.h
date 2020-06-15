//
// Created by Amir Joudaki on 6/9/20.
//

#ifndef SEQUENCE_SKETCHING_TYPES_H

#include <vector>

namespace Types {
    template<class T>
    using is_u_integral = typename std::enable_if<std::is_unsigned<T>::value>::type;
    using Index = std::size_t;
    using Size_t = std::size_t;
    template<class T>
    using Vec = std::vector<T>;
    template<class T>
    using Vec2D = Vec<Vec<T>>;
    template<class T>
    using Vec3D = Vec<Vec2D<T>>;
    template<class T>
    using Seq = std::vector<T>;

}// namespace Types

#define SEQUENCE_SKETCHING_TYPES_H

#endif//SEQUENCE_SKETCHING_TYPES_H
