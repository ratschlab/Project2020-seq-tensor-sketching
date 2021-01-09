#include "utils.hpp"

namespace ts {

std::string flag_values(char delimeter, bool skip_empty) {
    std::vector<gflags::CommandLineFlagInfo> flags;
    gflags::GetAllFlags(&flags);
    std::string result;
    for (const auto &flag : flags) {
        if (skip_empty && flag.current_value.empty())
            continue;
        result += "--" + flag.name + "=" + flag.current_value + delimeter;
    }
    return result;
}


} // namespace ts
