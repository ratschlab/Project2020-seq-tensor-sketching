#include <exception>
#include <string>
#include <utility>
namespace ts {

template <typename sketch_type_, bool kmer_input_>
class SketchBase {
  public:
    using sketch_type = sketch_type_;
    constexpr static bool kmer_input = kmer_input_;
    const std::string name;

    explicit SketchBase(std::string name) : name(std::move(name)) {}
};

} // namespace ts
