cmake_minimum_required(VERSION 3.16)
project(sequence_sketching)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-O3)
#add_executable(sequence_sketching types.h main.cpp)
add_executable(dists cpp/dists_pairwise.cpp)
add_executable(multiview cpp/multivec.cpp)
add_executable(dtw cpp/DTW.cpp)
