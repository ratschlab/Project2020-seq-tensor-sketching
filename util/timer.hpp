#pragma once

#include <chrono>
#include <map>
#include <string>
#include <utility>
#include <cassert>

namespace ts { // ts = Tensor Sketch

namespace Timer {

//extern std::map<std::string, std::chrono::nanoseconds> durations;

void start(std::string func_name);

void stop();

std::string summary(uint32_t num_seqs);

} // namespace Timer
} // namespace ts
