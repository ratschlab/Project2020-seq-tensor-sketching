//
// Created by amir on 18/12/2020.
//
#pragma once
#include <istream>

namespace ts {


struct progress_bar {
    static size_t it;
    static size_t step;
    static size_t total;

    static void init(size_t total_iterations, size_t bar_len = 100);
    static void iter() ;
};


};

