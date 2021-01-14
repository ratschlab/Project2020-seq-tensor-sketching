#pragma once

#include <exception>
#include <string>
#include <utility>
namespace ts {

template <typename sketch_type_, bool kmer_input_>
class SketchBase {
  public:
    // The type that the compute function returns.
    using sketch_type = sketch_type_;

    // Whether the compute function takes a list of kmers.
    constexpr static bool kmer_input = kmer_input_;

    // The name of the sketching algorithm.
    const std::string name;

    explicit SketchBase(std::string name) : name(std::move(name)) {}
};

} // namespace ts
