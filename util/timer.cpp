#include "timer.hpp"
#include <omp.h>
#include <vector>

namespace ts {
namespace Timer {

using namespace std::chrono;

//std::string last_func;
//auto last_time = std::chrono::high_resolution_clock::now();
//std::map<std::string, std::chrono::nanoseconds> durations;

std::vector<std::string> last_func_vec(100);
std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> last_time_vec(100);
std::vector<std::map<std::string, std::chrono::nanoseconds>> durations_vec(100);

void start(std::string func_name) {
    auto tid = omp_get_thread_num();
    auto &last_time = last_time_vec[tid];
    auto &last_func = last_func_vec[tid];
    assert(last_func.empty());
    last_time = high_resolution_clock::now();
    last_func = std::move(func_name);
}

void stop() {
    auto tid = omp_get_thread_num();
    auto &last_time = last_time_vec[tid];
    auto &last_func = last_func_vec[tid];
    auto &durations = durations_vec[tid];

    auto curr_time = high_resolution_clock::now();
    if (durations.find(last_func) != durations.end()) {
        durations[last_func] += duration_cast<nanoseconds>(curr_time - last_time);
    } else {
        durations[last_func] = duration_cast<nanoseconds>(curr_time - last_time);
    }
    last_func = "";
}

std::string summary(uint32_t num_seqs) {
    start("edit_distance");
    std::map<std::string, std::string> trans
            = { { "edit_distance", "ED" },        { "minhash", "MH" },
                { "weighted_minhash", "WMH" },    { "ordered_minhash_flat", "OMH" },
                { "tensor_sketch", "TenSketch" }, { "tensor_slide_sketch", "TenSlide" } };
    std::string str;
    std::map<std::string, double> acc;
    for (auto &durations : durations_vec) {
        for (auto const &[arg_name, arg] : durations) {
            if (acc.contains(arg_name)) {
                acc[arg_name] += arg.count();
            } else {
                acc[arg_name] = arg.count();
            }
        }
    }
        for (auto const &[arg_name, arg] : acc) {
            auto count = arg;
            if (arg_name.find("hash") != std::string::npos) {
                count += acc["seq2kmer"];
            }
            // dont print seq2kmer, it's part of other functions
            if (arg_name.find("seq2kmer") != std::string::npos)
                continue;
            str += " " + arg_name + ",\t" + trans[arg_name] + ",\t" + std::to_string(count/1e6/num_seqs) + '\n';
        }
    return str;
}

} // namespace Timer
} // namespace ts
