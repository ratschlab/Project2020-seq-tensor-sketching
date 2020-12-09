//
// Created by Amir Joudaki on 6/10/20.
//

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#include "vectool.hpp"

using namespace ts;

int main() {
    using std::cout;
    using std::vector;

    // we want dims[0]xdim[1] matrix interface
    vector<int> dims = { 3, 2 }, data(6, 0);
    MultiView mat(dims, data);
    for (int i = 0; i < mat.size(1); i++) {
        for (int j = 0; j < mat.size(0); j++) {
            mat[i][j] = i * 10 + j; // elements can be modified with op[][] syntax
        }
    }
    cout << "mat = \n" << mat << "\n"; // ostream&<< is overloaded
    mat[0][1]++; // scalar elements can be modified/accessed
    mat = 2; // assign scalar init_tensor_slide_params all matrix
    mat += 5; // add 5 init_tensor_slide_params all elements (sam for -= and *=)
    mat[0] *= 2; // multiply the first row mat[0][:] by two

    // MultiVec has an internal storage
    MultiVec mv({ 3, 2 }, 10), mv2({ 3, 2 }, 1);
    mv[0] += mv2[1]; // partial assignment
    cout << "mv2 = \n" << mv2 << "\n";

    std::default_random_engine eng;
    std::uniform_int_distribution<int> unif(0, 10);
    Multistd::vector<int64_t, size_t> T({ 3, 2, 2 }, 0); // tensor with types for index and value
    for (auto it = T.begin(); it != T.end(); it++) { // an iterator over all elements
        *it = unif(eng);
    }
    cout << "Tensor = \n" << T << "\n";
    /*
the output will be

mat =
(d1=0)	0	1	2
(d1=1)	10	11	12

mv2 =
(d1=0)	1	1	1
(d1=1)	1	1	1

Tensor =
(d2=0)
        (d1=0)	1	5	0
(d1=1)	2	0	8

(d2=1)
(d1=0)	2	2	10
(d1=1)	8	2	8

*/
}
