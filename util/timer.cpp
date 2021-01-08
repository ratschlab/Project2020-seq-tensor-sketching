#include <omp.h>
#include <utility>
#include <vector>
#include "timer.hpp"

namespace ts {

using namespace std::chrono;

std::vector<std::map<std::string, nanoseconds>> durations_vec(100);


void timer_add_duration(const std::string &func_name, nanoseconds dur) {
    auto tid = omp_get_thread_num();
    auto &durations = durations_vec[tid];

    if (durations.find(func_name) != durations.end()) {
        durations[func_name] += dur;
    } else {
        durations[func_name] = dur;
    }
}

std::string timer_summary(uint32_t num_seqs) {
    std::map<std::string, std::string> trans= {
            { "edit_distance", "ED" },
            { "minhash", "MH" },
            { "weighted_minhash", "WMH" },
            { "ordered_minhash_flat", "OMH" },
            { "tensor_sketch", "TenSketch" },
            { "tensor_slide_sketch", "TenSlide" },
            {"seq2kmer", "S2K"}
    };
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
            // add seq2kmer time to the sketch time of MH* methods
            if (arg_name.find("hash") != std::string::npos) {
                count += acc["seq2kmer"];
            }
            if (arg_name == "edit_distance") {
                count = count/1e6/num_seqs/(num_seqs-1);
            } else if (arg_name == "main_func") {
              count = count/1e6;
            } else {
                count = count/1e6/num_seqs;
            }
            str += " " + arg_name + ",\t" + trans[arg_name] + ",\t" + std::to_string(count) + '\n';
        }
    return str;
}

} // namespace ts
