cmake_minimum_required(VERSION 3.13)
project(sequence_sketching)
include_directories(.)
set(CMAKE_CXX_STANDARD 20)

add_compile_options(-O0)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(exp_pairwise experiments_main.cpp)

add_executable(sketch tensor_sketch_main.cpp)

add_executable(seqgen util/seqgen.cpp)

# TESTS
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Werror")

enable_testing()
include(GoogleTest)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
target_compile_options(gtest_main PRIVATE -w)
target_compile_options(gtest PRIVATE -w)

file(GLOB test_files "tests/**/*.cpp")

add_executable(tests ${test_files})
target_link_libraries(tests gtest_main gtest gmock)
target_include_directories(tests PRIVATE "include")

gtest_discover_tests(tests)

