cmake_minimum_required(VERSION 3.13)
project(sequence_sketching)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
find_package(OpenMP REQUIRED)

include_directories(.)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Google Flags Library
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/gflags EXCLUDE_FROM_ALL)

# Murmur
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/murmur_hash EXCLUDE_FROM_ALL)

file(GLOB util_files "util/*.cpp")
add_library(util ${util_files})
target_link_libraries(util gflags OpenMP::OpenMP_CXX)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(util stdc++fs)
endif()

file(GLOB sequence_files "sequence/*.cpp")
add_library(sequence ${sequence_files})
target_link_libraries(sequence gflags OpenMP::OpenMP_CXX)

file(GLOB sketch_files "sketch/*.cpp")
add_library(sketch_lib ${sketch_files})
target_link_libraries(sketch_lib murmur_lib OpenMP::OpenMP_CXX)

file(GLOB phylogeny_files "phylogeny/*.cpp")
add_library(phylogeny_lib ${phylogeny_files})

add_executable(experiments experiments_main.cpp )
target_link_libraries(experiments sequence util sketch_lib)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(experiments stdc++fs)
endif()

add_executable(sketch sketch_main.cpp)
target_link_libraries(sketch sequence util sketch_lib)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(sketch stdc++fs)
endif()

add_executable(seqgen sequence_generator_main.cpp)
target_link_libraries(seqgen sequence util sketch_lib)

# TESTS
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Werror -Wfatal-errors -msse4")


enable_testing()
include(GoogleTest)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
target_compile_options(gtest_main PRIVATE -w)
target_compile_options(gtest PRIVATE -w)

file(GLOB test_files "tests/**/*.cpp")

add_executable(tests ${test_files})
target_link_libraries(tests gtest_main gtest gmock util sketch_lib phylogeny_lib)
target_include_directories(tests PRIVATE "include")

gtest_discover_tests(tests)
