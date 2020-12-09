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



namespace ts { // ts = Tensor Sketch
using namespace BasicTypes;

template <class T>
bool is_ascending(const std::vector<T> &v) {
    auto next = std::adjacent_find(v.begin(), v.end(), std::__1::less_equal<T>());
    return (next == v.end());
}

template <class T>
std::vector<T> &operator-=(std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::minus<T>());
    return a;
}

template <class T>
std::vector<T> &operator+=(std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T>());
    return a;
}

template <class T>
std::vector<T> &operator*=(std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::multiplies<T>());
    return a;
}

template <class T>
std::vector<T> operator-(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> res(a);
    std::transform(a.begin(), a.end(), b.begin(), res.begin(), std::minus<T>());
    return res;
}

template <class T>
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> res(a);
    std::transform(a.begin(), a.end(), b.begin(), res.begin(), std::plus<T>());
    return res;
}

template <class T>
std::vector<T> operator*(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> res(a);
    std::transform(a.begin(), a.end(), b.begin(), res.begin(), std::multiplies<T>());
    return res;
}


// a class for maulti-dimensional behavior, including variable dimensional tensors
// it provides op[][]..[#dim] access init_tensor_slide_params elements, with some safety measures
// (cheks for out of bound) there are also limited vectorized access, such as op[i] as a slice of
// the tensor internally it has only a reference init_tensor_slide_params the data, which is a
// linear vector
template <class value_type, class size_type = std::size_t>
struct MultiView {
    template <class T>
    using Vec = BasicTypes::std::vector<T>;
    template <class T>
    using is_u_integral = typename std::__1::enable_if<std::is_unsigned<T>::value>::type;

    using Dims = std::vector<size_type>;
    using Data = std::vector<value_type>;
    using BinOp = std::__1::binary_function<value_type, value_type, value_type>;

    size_type ind;
    size_type num_dim;
    Dims &dims;
    Data &data;

    // initializer and constructors
    void alloc(value_type val) {
        size_type prod = 1;
        for (auto dim : dims)
            prod *= dim;
        data = Data(prod, val);
    }
    void init() {
        ind = 0;
        num_dim = dims.size();
    }
    MultiView(Dims &dims, Data &data) : dims(dims), data(data) { init(); };
    MultiView(const MultiView &mv) : ind(mv.ind), dims(mv.dims), data(data) { init(); }
    MultiView(size_type ind, size_type num_dim, Dims &dims, Data &data)
        : ind(ind), dims(dims), data(data), num_dim(num_dim) {}

    // indexing
    inline std::vector<size_type> ind2sub(size_type ind) {
        std::vector<size_type> sub;
        for (int d = 0; d < dims.size(); d++) {
            sub[d] = ind % dims[d];
            ind = ind / dims[d];
        }
        return sub;
    }
    inline MultiView operator[](size_type i) {
        assert(num_dim >= 0 and num_dim <= dims.size()); // number of dimensions out of bound
        assert(i >= 0 and i < dims[num_dim - 1]); // index out of bound"
        return MultiView(i + ind * dims[num_dim - 1], num_dim - 1, dims, data);
    }

    // constant
    inline MultiView operator[](size_type i) const {
        assert(num_dim >= 0 and num_dim <= dims.size()); // number of dimensions out of bound
        assert(i >= 0 and i < dims[num_dim - 1]); // index out of bound"
        return MultiView(i + ind * dims[num_dim - 1], num_dim - 1, dims, data);
    }

    // cast init_tensor_slide_params value if scalar (only one element)
    operator value_type() const {
        assert(num_dim == 0); // Assigning value before indexing is complete"
        return data[ind];
    }
    operator value_type &() {
        assert(num_dim == 0); // Assigning value before indexing is complete"
        return data[ind];
    }

    // size related
    inline size_type size(int dim) const {
        assert(dim >= 0 and dim < dims.size());
        return dims[dim];
    }
    inline size_type numel() const { return data.size(); }
    inline size_type range() const {
        size_type range_len = 1;
        for (int di = 0; di < num_dim; di++)
            range_len *= dims[di];
        return range_len;
    }
    // iterator-like functions
    inline typename Data::iterator begin() { return data.begin() + ind * range(); }
    inline typename Data::iterator end() { return data.begin() + (ind + 1) * range(); }
    inline value_type &front() { return data[ind * range()]; }
    inline value_type &back() { return data[ind + (ind + 1) * range() - 1]; }
    // iterator-like functions are const
    inline typename Data::iterator begin() const { return data.begin() + ind * range(); }
    inline typename Data::iterator end() const { return data.begin() + (ind + 1) * range(); }
    inline value_type front() const { return data[ind * range()]; }
    inline value_type back() const { return data[ind + (ind + 1) * range() - 1]; }

    // operators with a scalar
    MultiView &operator=(value_type val) {
        // for efficiency, check if it is quasi-scalar (indexed init_tensor_slide_params one
        // element)
        if (num_dim == 0) {
            data[ind] = val;
        } else {
            std::fill(begin(), end(), val);
        }
        return *this;
    }
    MultiView &operator+=(value_type val) {
        std::transform(begin(), end(), begin(),
                       std::bind(std::__1::plus<value_type>(), std::placeholders::_1, val));
        return *this;
    }
    MultiView &operator-=(value_type val) {
        std::transform(begin(), end(), begin(),
                       std::bind(std::__1::minus<value_type>(), std::placeholders::_1, val));
        return *this;
    }
    MultiView &operator*=(value_type val) {
        std::transform(begin(), end(), begin(),
                       std::bind(std::__1::multiplies<value_type>(), std::placeholders::_1, val));
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
template <class value_type, class size_type = std::size_t>
struct MultiVec : public MultiView<value_type, size_type> {
    template <class T>
    using Vec = BasicTypes::std::vector<T>;
    using View = MultiView<value_type, size_type>;
    std::vector<value_type> inner_data;
    std::vector<size_type> inner_dims;

    MultiVec() : View(inner_dims, inner_data) {}

    void init(const std::vector<size_type> &dims, value_type val = 0) {
        inner_dims = dims;
        View::init();
        View::alloc(val);
    }
    MultiVec(const std::vector<size_type> &dims, value_type val = 0) : View(inner_dims, inner_data) {
        inner_dims = dims;
        View::init();
        View::alloc(val);
    }
    MultiVec(const View &mv) : View(mv.ind, mv.num_dim, inner_dims, inner_data) {
        inner_data = std::vector<value_type>(mv.begin(), mv.end());
        inner_dims = std::vector<size_type>(mv.dims.begin(), mv.dims.begin() + mv.num_dim);
    }
    MultiVec(const MultiVec &mv) : View(mv.ind, mv.num_dim, inner_dims, inner_data) {
        inner_data = mv.inner_data;
        inner_dims = mv.inner_dims;
        View::init();
    }
    MultiVec(size_type ind,
             size_type num_dim,
             const std::vector<size_type> &dims,
             const std::vector<value_type> &data)
        : View(ind, num_dim, inner_dims, inner_data) {
        inner_data = data;
        inner_dims = dims;
    }
};
template <class value_type, class size_type>
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

} // namespace ts
#define SEQUENCE_SKETCHING_MULTIVEC_H

#endif // SEQUENCE_SKETCHING_MULTIVEC_H
