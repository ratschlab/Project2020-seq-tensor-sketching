#include <omp.h>
#include <utility>
#include <vector>
#include "timer.hpp"

namespace ts {

using namespace std::chrono;

auto Timer::durations_vec = std::vector<std::map<std::string, nanoseconds>>(100);


void Timer::add_duration(const std::string &func_name, nanoseconds dur) {
    int tid = omp_get_thread_num();
    std::map<std::string, std::chrono::nanoseconds> &durations = durations_vec[tid];

    if (durations.find(func_name) != durations.end()) {
        durations[func_name] += dur;
    } else {
        durations[func_name] = dur;
    }
}


std::string Timer::summary(uint32_t num_seqs, uint32_t num_pairs) {
    std::map<std::string, std::string> trans= {
            { "edit_distance", "ED" },
            { "minhash", "MH" },
            { "weighted_minhash", "WMH" },
            { "ordered_minhash", "OMH" },
            { "tensor_sketch", "TS" },
            { "tensor_slide_sketch", "TSS" },
            {"Int32Flattener", "I32FLAT"},
            {"DoubleFlattener", "FLAT"},
            {"seq2kmer", "S2K"}
    };
    std::string str = "long name,short name, time, time sketch, time dist\n";
    std::map<std::string, double> acc;
    for (auto &durations : Timer::durations_vec) {
        for (auto const &[arg_name, arg] : durations) {
            if (acc.find(arg_name) != acc.end()) {
                acc[arg_name] += arg.count();
            } else {
                acc[arg_name] = arg.count();
            }
        }
    }
    for (auto const &[arg_name, arg] : acc) {
        double count = (double)arg, count2;
        if (arg_name.find("hash") != std::string::npos && // contains *hash*
            arg_name.find("dist") == std::string::npos) { // doesn't contain *dist*
            count += acc["seq2kmer"]; // add kmer computation time to MH* methods
        }
        if (arg_name == "edit_distance") {
            count = count/1e6/num_pairs;
            str += arg_name + "," + trans[arg_name] + "," + std::to_string(count) + ",0,0\n";
        } else if (arg_name.find("dist") == std::string::npos && arg_name!="seq2kmer") {
            count = count/1e6/num_seqs ;    // mean sketching time (ms)
            count2 = acc[arg_name + "_dist"]/1e6/num_pairs; // mean distance computation time (ms)
            str += arg_name + "," + trans[arg_name] +
                  "," + std::to_string(count+count2) +
                  "," + std::to_string(count) +
                  "," + std::to_string(count2) + '\n';
        }
    }
    return str;
}

} // namespace ts