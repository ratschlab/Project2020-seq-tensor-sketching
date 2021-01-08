#include "utils.hpp"

namespace ts {
std::string flag_values(char delimiter) {
    std::vector<gflags::CommandLineFlagInfo> flags;
    gflags::GetAllFlags(&flags);
    std::string result;
    for (const auto &flag : flags) {
        if (flag.current_value.size()>0) { // omit empty flags
            result += "--" + flag.name + "=" + flag.current_value + delimiter;
        }
    }
    return result;
}


} // namespace ts
