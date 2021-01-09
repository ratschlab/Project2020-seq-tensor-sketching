#pragma once

#include <chrono>
#include <map>
#include <string>
#include <utility>
#include <cassert>
#include <vector>


namespace ts { // ts = Tensor Sketch

using namespace std::chrono;

// TODO create a macro to create timer object

// TODO make this static members of Timer
extern std::vector<std::map<std::string, std::chrono::nanoseconds>> durations_vec;

// TODO make this static members of Timer
void timer_add_duration(const std::string &func_name, std::chrono::nanoseconds dur) ;

std::string timer_summary(uint32_t num_seqs, uint32_t num_pairs);

class Timer {
  public:
    Timer(std::string name) :
            name(std::move(name)),
            birth(high_resolution_clock::now()){}

    Timer(const Timer &tt) :
            name(tt.name),
            birth(high_resolution_clock::now()){}
    ~Timer() {
        auto dur = high_resolution_clock::now() - birth;
        timer_add_duration(name, dur);
    }

  private:
    std::string name;
    high_resolution_clock::time_point birth;
};

} // namespace ts
