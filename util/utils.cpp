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

} // namespace ts
