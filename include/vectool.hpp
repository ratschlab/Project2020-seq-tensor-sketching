//
// Created by Amir Joudaki on 6/9/20.
//

#ifndef SEQUENCE_SKETCHING_MULTIVEC_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <cstdlib>
#include <cmath>

#include "args.hpp"

namespace SeqSketch {
    using namespace BasicTypes;

    template<class T>
    auto new2D(int d1, int d2, T val = 0) {
        return Vec2D<T>(d1, Vec<T>(d2, 0));
    }
    template<class T>
    auto new3D(int d1, int d2, int d3, T val = 0) {
        return Vec3D<T>(d1, new2D(d2, d3, val));
    }

    // utility functions for len^D-dimensional tensor
    template<class T, class = is_u_integral<T>>
    bool increment_sub(Vec<T> &sub, T len) {
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

    template<class T, class = is_u_integral<T>>
    T sub2ind(const Vec<T> &sub, T len) {
        T ind = 0, coef = 1;
        for (size_t i = 0; i < sub.size(); i++) {
            ind += sub[i] * coef;
            coef *= len;
        }
        return ind;
    }

    template<class T, class = is_u_integral<T>>
    T int_pow(T x, T pow) {
        if (pow == 0) return 1;
        if (pow == 1) return x;

        T tmp = int_pow(x, pow / 2);
        if (pow % 2 == 0) return tmp * tmp;
        else
            return x * tmp * tmp;
    }

/*
    template<class T>
    bool is_ascending(const Vec<T> &v) {
        auto next = std::adjacent_find(v.begin(), v.end(), std::__1::less_equal<T>());
        return (next == v.end());
    }

    template<class T>
    Vec<T> &operator-=(Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::minus<T>());
        return a;
    }

    template<class T>
    Vec<T> &operator+=(Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T>());
        return a;
    }

    template<class T>
    Vec<T> &operator*=(Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::multiplies<T>());
        return a;
    }

    template<class T>
    Vec<T> operator-(const Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        Vec<T> res(a);
        std::transform(a.begin(), a.end(), b.begin(), res.begin(), std::minus<T>());
        return res;
    }

    template<class T>
    Vec<T> operator+(const Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        Vec<T> res(a);
        std::transform(a.begin(), a.end(), b.begin(), res.begin(), std::plus<T>());
        return res;
    }

    template<class T>
    Vec<T> operator*(const Vec<T> &a, const Vec<T> &b) {
        assert(a.size() == b.size());
        Vec<T> res(a);
        std::transform(a.begin(), a.end(), b.begin(), res.begin(), std::multiplies<T>());
        return res;
    }




    // a class for maulti-dimensional behavior, including variable dimensional tensors
    // it provides op[][]..[#dim] access init_tensor_slide_params elements, with some safety measures (cheks for out of bound)
    // there are also limited vectorized access, such as op[i] as a slice of the tensor
    // internally it has only a reference init_tensor_slide_params the data, which is a linear vector
    template<class value_type, class size_type = std::size_t>
    struct MultiView {
        template<class T>
        using Vec = BasicTypes::Vec<T>;
        template<class T>
        using is_u_integral = typename std::__1::enable_if<std::is_unsigned<T>::value>::type;

        using Dims = Vec<size_type>;
        using Data = Vec<value_type>;
        using BinOp = std::__1::binary_function<value_type, value_type, value_type>;

        size_type ind;
        size_type num_dim;
        Dims &dims;
        Data &data;

        // initializer and constructors
        void alloc(value_type val) {
            size_type prod = 1;
            for (auto dim : dims) prod *= dim;
            data = Data(prod, val);
        }
        void init() {
            ind = 0;
            num_dim = dims.size();
        }
        MultiView(Dims &dims, Data &data) : dims(dims), data(data) { init(); };
        MultiView(const MultiView &mv) : ind(mv.ind), dims(mv.dims), data(data) { init(); }
        MultiView(size_type ind, size_type num_dim, Dims &dims, Data &data) : ind(ind), dims(dims), data(data), num_dim(num_dim) {}

        // indexing
        inline Vec<size_type> ind2sub(size_type ind) {
            Vec<size_type> sub;
            for (int d = 0; d < dims.size(); d++) {
                sub[d] = ind % dims[d];
                ind = ind / dims[d];
            }
            return sub;
        }
        inline MultiView operator[](size_type i) {
            assert(num_dim >= 0 and num_dim <= dims.size());// number of dimensions out of bound
            assert(i >= 0 and i < dims[num_dim - 1]);       // index out of bound"
            return MultiView(i + ind * dims[num_dim - 1], num_dim - 1, dims, data);
        }

        // constant
        inline MultiView operator[](size_type i) const {
            assert(num_dim >= 0 and num_dim <= dims.size());// number of dimensions out of bound
            assert(i >= 0 and i < dims[num_dim - 1]);       // index out of bound"
            return MultiView(i + ind * dims[num_dim - 1], num_dim - 1, dims, data);
        }

        // cast init_tensor_slide_params value if scalar (only one element)
        operator value_type() const {
            assert(num_dim == 0);// Assigning value before indexing is complete"
            return data[ind];
        }
        operator value_type &() {
            assert(num_dim == 0);// Assigning value before indexing is complete"
            return data[ind];
        }

        // size related
        inline size_type size(int dim) const {
            assert(dim >= 0 and dim < dims.size());
            return dims[dim];
        }
        inline size_type numel() const {
            return data.size();
        }
        inline size_type range() const {
            size_type range_len = 1;
            for (int di = 0; di < num_dim; di++) range_len *= dims[di];
            return range_len;
        }
        // iterator-like functions
        inline typename Data::iterator begin() {
            return data.begin() + ind * range();
        }
        inline typename Data::iterator end() {
            return data.begin() + (ind + 1) * range();
        }
        inline value_type &front() {
            return data[ind * range()];
        }
        inline value_type &back() {
            return data[ind + (ind + 1) * range() - 1];
        }
        // iterator-like functions are const
        inline typename Data::iterator begin() const {
            return data.begin() + ind * range();
        }
        inline typename Data::iterator end() const {
            return data.begin() + (ind + 1) * range();
        }
        inline value_type front() const {
            return data[ind * range()];
        }
        inline value_type back() const {
            return data[ind + (ind + 1) * range() - 1];
        }

        // operators with a scalar
        MultiView &operator=(value_type val) {
            // for efficiency, check if it is quasi-scalar (indexed init_tensor_slide_params one element)
            if (num_dim == 0) {
                data[ind] = val;
            } else {
                std::fill(begin(), end(), val);
            }
            return *this;
        }
        MultiView &operator+=(value_type val) {
            std::transform(begin(), end(), begin(), std::bind(std::__1::plus<value_type>(), std::placeholders::_1, val));
            return *this;
        }
        MultiView &operator-=(value_type val) {
            std::transform(begin(), end(), begin(), std::bind(std::__1::minus<value_type>(), std::placeholders::_1, val));
            return *this;
        }
        MultiView &operator*=(value_type val) {
            std::transform(begin(), end(), begin(), std::bind(std::__1::multiplies<value_type>(), std::placeholders::_1, val));
            return *this;
        }

        MultiView &operator=(const MultiView &mv) {
            assert(range() == mv.range());
            std::copy(mv.begin(), mv.end(), begin());
            return *this;
        }
        MultiView &operator+=(const MultiView &mv) {
            std::transform(begin(), end(), mv.begin(), begin(), std::__1::plus<value_type>());
            return *this;
        }
        MultiView &operator-=(const MultiView &mv) {
            std::transform(begin(), end(), mv.begin(), begin(), std::__1::minus<value_type>());
            return *this;
        }
        MultiView &operator*=(const MultiView &mv) {
            std::transform(begin(), end(), mv.begin(), begin(), std::__1::multiplies<value_type>());
            return *this;
        }
    };
    // Wraps around MultiView, with storage containers
    // Copy constructs is a deep copy, same for operator= therefore they're independent after
    template<class value_type, class size_type = std::size_t>
    struct MultiVec : public MultiView<value_type, size_type> {
        template<class T>
        using Vec = BasicTypes::Vec<T>;
        using View = MultiView<value_type, size_type>;
        Vec<value_type> inner_data;
        Vec<size_type> inner_dims;

        MultiVec() : View(inner_dims, inner_data) {}

        void init(const Vec<size_type> &dims, value_type val = 0) {
            inner_dims = dims;
            View::init();
            View::alloc(val);
        }
        MultiVec(const Vec<size_type> &dims, value_type val = 0)
            : View(inner_dims, inner_data) {
            inner_dims = dims;
            View::init();
            View::alloc(val);
        }
        MultiVec(const View &mv)
            : View(mv.ind, mv.num_dim, inner_dims, inner_data) {
            inner_data = Vec<value_type>(mv.begin(), mv.end());
            inner_dims = Vec<size_type>(mv.dims.begin(), mv.dims.begin() + mv.num_dim);
        }
        MultiVec(const MultiVec &mv)
            : View(mv.ind, mv.num_dim, inner_dims, inner_data) {
            inner_data = mv.inner_data;
            inner_dims = mv.inner_dims;
            View::init();
        }
        MultiVec(size_type ind, size_type num_dim, const Vec<size_type> &dims, const Vec<value_type> &data)
            : View(ind, num_dim, inner_dims, inner_data) {
            inner_data = data;
            inner_dims = dims;
        }
    };
    template<class value_type, class size_type>
    std::ostream &operator<<(std::ostream &os, const MultiView<value_type, size_type> &mv) {
        if (mv.num_dim >= 2) {
            for (size_type d = 0; d < mv.dims[mv.num_dim - 1]; d++) {
                std::string pre("");
                pre.append("\t\t", std::max<size_type>(mv.dims.size() - mv.num_dim, 0));
                auto seperator = (mv.num_dim == 2) ? ")\t" : ")\n";
                os << pre << "(d" << mv.num_dim - 1 << "=" << d << seperator << mv[d] << "\n";
            }
        } else {
            for (auto it = mv.begin(); it != mv.end(); it++) {
                os << (*it) << "\t";
            }
        }
        return os;
    }
*/

}// namespace VecTools
#define SEQUENCE_SKETCHING_MULTIVEC_H

#endif//SEQUENCE_SKETCHING_MULTIVEC_H
