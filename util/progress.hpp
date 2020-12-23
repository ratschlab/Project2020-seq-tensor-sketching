//
// Created by amir on 18/12/2020.
//
#pragma once
#include <istream>

namespace ts {

size_t global_num;
size_t global_iter;
size_t global_bar;

void pbar_start(size_t num, size_t bar_len = 100) {
    if (num < bar_len)
        bar_len= num;
    global_num = num;
    global_iter = 0;
    global_bar = bar_len;
}

void pbar_inc() {
    // create aliases
    auto &iter = global_iter, num=global_num, bar = global_bar;

    size_t chunk = (num + bar - 1) / bar; // round up num/bar
    size_t i = iter / chunk, p = (iter-1)/chunk;
    if (i>p)
        std::cout << "#" << std::flush;
#pragma omp atomic
        global_iter++;
//        if (i>0) {
//            for (size_t j=0; j<bar; j++)
//                std::cout << "\b" << std::flush;
//        }
//        for (size_t j=0; j<i; j++)
//            std::cout << ">" << std::flush;
//        for (size_t j=0; j<bar-i; j++)
//            std::cout << "#" << std::flush;
//        std::cout << std::flush;
//
//#pragma omp atomic
//    global_iter++;
}

}
