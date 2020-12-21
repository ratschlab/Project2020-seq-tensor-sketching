#include "utils.hpp"

namespace ts {
std::string flag_values() {
    std::vector<gflags::CommandLineFlagInfo> flags;
    gflags::GetAllFlags(&flags);
    std::string result;
    for (const auto &flag : flags) {
        result += flag.name + "=" + flag.current_value + " ";
    }
    return result;
}

std::string legacy_config() {
    std::string str;
    std::vector<gflags::CommandLineFlagInfo> flags;
    gflags::GetAllFlags(&flags);
    for (const auto &flag : flags) {
        str += " " + flag.name + ",\t" + flag.type + ",\t" + flag.current_value + '\n';
    }
    return str;
}

} // namespace ts
