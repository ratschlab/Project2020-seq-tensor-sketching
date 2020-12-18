cmake_minimum_required(VERSION 3.13)
project(sequence_sketching)
set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

include_directories(.)

#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0")

# Google Flags Library
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/gflags EXCLUDE_FROM_ALL)

file(GLOB util_files "util/*.cpp")
add_library(util ${util_files})
target_link_libraries(util gflags)

file(GLOB sequence_files "sequence/*.cpp")
add_library(sequence ${sequence_files})
target_link_libraries(sequence gflags)


add_executable(exp_pairwise experiments_main.cpp )
target_link_libraries(exp_pairwise sequence util OpenMP::OpenMP_CXX)

add_executable(grid_search grid_search_main.cpp)
target_link_libraries(grid_search sequence util)

add_executable(sketch sketch_main.cpp)
target_link_libraries(sketch sequence util)

add_executable(seqgen sequence_generator_main.cpp)
target_link_libraries(seqgen sequence util)

# TESTS
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Werror")

enable_testing()
include(GoogleTest)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
target_compile_options(gtest_main PRIVATE -w)
target_compile_options(gtest PRIVATE -w)

file(GLOB test_files "tests/**/*.cpp")

add_executable(tests ${test_files})
target_link_libraries(tests gtest_main gtest gmock util)
target_include_directories(tests PRIVATE "include")

gtest_discover_tests(tests)

