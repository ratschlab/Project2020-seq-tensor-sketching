#pragma once

#include <chrono>
#include <map>
#include <string>
#include <utility>
#include <cassert>

namespace ts { // ts = Tensor Sketch

namespace Timer {
using namespace std::chrono;

std::map<std::string, nanoseconds> durations;

auto last_time = high_resolution_clock::now();

std::string last_func;

void start(std::string func_name) {
    assert(last_func.empty());
    last_time = high_resolution_clock::now();
    last_func = std::move(func_name);
}

void stop() {
    auto curr_time = high_resolution_clock::now();
    if (durations.find(last_func) != durations.end()) {
        durations[last_func] += duration_cast<nanoseconds>(curr_time - last_time);
    } else {
        durations[last_func] = duration_cast<nanoseconds>(curr_time - last_time);
    }
    last_func = "";
}

std::string summary() {
    start("edit_distance");
    std::map<std::string, std::string> trans
            = { { "edit_distance", "ED" },        { "minhash", "MH" },
                { "weighted_minhash", "WMH" },    { "ordered_minhash_flat", "OMH" },
                { "tensor_sketch", "TenSketch" }, { "tensor_slide_sketch", "TenSlide" } };
    std::string str;
    for (auto const &[arg_name, arg] : durations) {
        auto count = arg.count();
        if (arg_name.find("hash") != std::string::npos) {
            count += durations["seq2kmer"].count();
        }
        str += " " + arg_name + ",\t" + trans[arg_name] + ",\t" + std::to_string(count) + '\n';
    }
    return str;
}

} // namespace Timer
} // namespace ts
