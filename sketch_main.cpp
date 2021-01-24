#include "sequence/fasta_io.hpp"
#include "util/multivec.hpp"
#include "util/utils.hpp"

#include <gflags/gflags.h>

#include <filesystem>
#include <memory>
#include <random>
#include <sstream>
#include <utility>

DEFINE_string(o, "", "Output file");

DEFINE_string(i, "", "Input directory");

DEFINE_int32(t, 4, "Thread count");

DEFINE_int32(b, 32, "Block size");

static bool ValidateInput(const char * /*unused*/, const std::string &value) {
    if (!value.empty()) {
        return true;
    }
    std::cerr << "Please specify a fasta input file using '-i <input_file>'" << std::endl;
    return false;
}
DEFINE_validator(i, &ValidateInput);


using namespace ts;

namespace fs = std::filesystem;
using Tensor23 = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;
Tensor23 tensor23(std::vector<uint8_t> &seq) {
    Tensor23 result({ std::vector<uint32_t>(16), std::vector<uint32_t>(64) });
    std::transform(seq.begin(), seq.end(), seq.begin(), [](uint8_t a) { return (a > 3) ? 0 : a; });
    for (uint32_t i = 0; i < seq.size(); ++i) {
        const uint32_t idx1 = 16 * seq[i];
        for (uint32_t j = i + 1; j < seq.size(); ++j) {
            const uint32_t idx2 = idx1 + 4 * seq[j];
            for (uint32_t k = j + 1; k < seq.size(); ++k) {
                result.second[idx2 + seq[k]]++;
            }
            result.first[(seq[i] << 2) + seq[j]]++;
        }
    }
    return result;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    init_alphabet("DNA4");

    std::vector<std::string> names;
    for (auto &p : fs::directory_iterator(FLAGS_i)) {
        if (p.is_directory()
            || (p.path().filename().string().find(".fa") == std::string::npos
                && p.path().filename().string().find(".fna.gz") == std::string::npos)) {
            continue;
        }
        names.push_back(p.path().string());
    }
    std::cout << "Found " << names.size() << " fasta files" << std::endl;
    std::vector<std::pair<std::vector<uint8_t>, std::string>> sequences(names.size());
#pragma omp parallel for num_threads(FLAGS_t)
    for (uint32_t i = 0; i < names.size(); ++i) {
        FastaFile<uint8_t> ff = read_fasta<uint8_t>(names[i], "fasta");
        sequences[i] = { ff.sequences[0], fs::path(names[i]).filename() };
    }
    std::sort(sequences.begin(), sequences.end(),
              [](auto &a, auto &b) { return a.second < b.second; });
    std::cout << "Read " << sequences.size() << " fasta files" << std::endl;

    if (sequences.empty()) {
        std::exit(0);
    }

    fs::remove_all(FLAGS_o);
    const size_t BLOCK_SIZE = FLAGS_b;
    std::vector<Tensor23> tensors(sequences.size());
    for (uint32_t epoch = 0; epoch < (sequences.size() - 1) / BLOCK_SIZE + 1; ++epoch) {
        size_t max_ind = std::min((epoch + 1) * BLOCK_SIZE, sequences.size());
#pragma omp parallel for num_threads(FLAGS_t)
        for (uint32_t i = epoch * BLOCK_SIZE; i < max_ind; ++i) {
            tensors[i] = tensor23(sequences[i].first);
        }
        std::ofstream f2(FLAGS_o + "2");
        std::ofstream f3(FLAGS_o + "3");
        for (uint32_t j = 0; j < max_ind; ++j) {
            for (uint32_t k = j + 1; k < max_ind; ++k) {
                f2 << sequences[j].second << "," << sequences[k].second << ", "
                   << l2_dist(tensors[j].first, tensors[k].first) << std::endl;
                f3 << sequences[j].second << "," << sequences[k].second << ", "
                   << l2_dist(tensors[j].second, tensors[k].second) << std::endl;
            }
        }
        std::cout << "Epoch " << epoch << " done." << std::endl;
    }
}
