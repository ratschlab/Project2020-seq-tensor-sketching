//
// Created by amir on 18/12/2020.
//
#pragma once
#include <istream>

void print_progress(size_t iter, size_t num, size_t bar = 100) {
    size_t chunk = (num+bar-1)/bar; // round up num/bar
    size_t i = iter/chunk, p = (iter-1)/chunk;
    if (i>p) {
        std::cout << "#" << std::flush;
    }
}

