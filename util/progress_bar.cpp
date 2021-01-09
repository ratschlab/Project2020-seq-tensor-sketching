//
// Created by amir on 1/9/21.
//
#include <iostream>
#include "util/progress.hpp"

namespace ts {

size_t progress_bar::it;
size_t progress_bar::step;
size_t progress_bar::total;

void progress_bar::init(size_t total_iterations, size_t bar_len ) {
    it = 0;
    total = total_iterations;
    step = (total + bar_len -1) / bar_len; // iterations/bar-length, rounded up
}

void progress_bar::iter() {
    if (it % step == 0) {
        std::cout << "#" << std::flush;
    }
#pragma omp atomic
    it++;
}

};


