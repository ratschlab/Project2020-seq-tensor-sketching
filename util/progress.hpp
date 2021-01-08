//
// Created by amir on 18/12/2020.
//
#pragma once
#include <istream>

namespace ts {

size_t global_iter;
size_t global_iter_step;

void PB_init(size_t total_iterations, size_t bar_len = 100) {
    global_iter = 0;
    // iterations divided by bar length, rounded up
    global_iter_step = (total_iterations + bar_len -1) / bar_len;
}

void PB_iter() {
    if (global_iter % global_iter_step == 0) {
        std::cout << "#" << std::flush;
    }
#pragma omp atomic
    global_iter++;
}

}
