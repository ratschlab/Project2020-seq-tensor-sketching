#include "utils.hpp"

#include <gflags/gflags.h>
#include <numeric>

namespace ts {

std::string flag_values(char delimiter, bool skip_empty) {
    std::vector<gflags::CommandLineFlagInfo> flags;
    gflags::GetAllFlags(&flags);
    std::string result;
    for (const auto &flag : flags) {
        if (skip_empty && flag.current_value.empty())
            continue;
        result += "--" + flag.name + "=" + flag.current_value + delimiter;
    }
    return result;
}

std::pair<double, double> avg_stddev(const std::vector<double> &v) {
    if (v.empty())
        return { 0, 0 };
    const double sum = std::accumulate(begin(v), end(v), 0.0);
    const double avg = sum / v.size();

    double var = 0;
    for (const auto &x : v)
        var += (x - avg) * (x - avg);

    return { avg, sqrt(var / v.size()) };
}

} // namespace ts
