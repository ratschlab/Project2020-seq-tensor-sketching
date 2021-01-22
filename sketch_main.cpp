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

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    init_alphabet("DNA5");

    std::vector<std::pair<std::vector<uint8_t>, std::string>> sequences;

    std::cout << "Reading fasta files..." << std::endl;
    for (auto &p : fs::directory_iterator(FLAGS_i)) {
        if (p.is_directory()) {
            continue;
        }
        std::cout << p.path().filename() << std::endl;
        FastaFile<uint8_t> ff = read_fasta<uint8_t>(p.path().string(), "fasta");
        sequences.push_back({ ff.sequences[0], p.path().filename() });
    }
    std::sort(sequences.begin(), sequences.end(),
              [](auto &a, auto &b) { return a.second < b.second; });
    std::cout << "Read " << sequences.size() << " files" << std::endl;

    fs::remove_all(FLAGS_o);
    constexpr size_t BLOCK_SIZE = 4;
    for (uint32_t epoch = 0; epoch < (sequences.size() - 1) / BLOCK_SIZE + 1; ++epoch) {
        std::vector<size_t> distances(epoch * BLOCK_SIZE * BLOCK_SIZE);
        size_t max_ind = std::min((epoch + 1) * BLOCK_SIZE, sequences.size());
#pragma omp parallel for
        for (uint32_t i = 0; i < epoch * BLOCK_SIZE; ++i) {
            for (uint32_t j = epoch * BLOCK_SIZE; j < max_ind; ++j) {
                distances[i * BLOCK_SIZE + j - epoch * BLOCK_SIZE]
                        = edit_distance(sequences[i].first, sequences[j].first);
            }
        }
        std::cout << ".";
        std::vector<std::vector<size_t>> distances2(BLOCK_SIZE, std::vector<size_t>(BLOCK_SIZE));
#pragma omp parallel for
        for (uint32_t i = epoch * BLOCK_SIZE; i < max_ind; ++i) {
            for (uint32_t j = i + 1; j < max_ind; ++j) {
                distances2[(i - epoch * BLOCK_SIZE)][j - epoch * BLOCK_SIZE]
                        = edit_distance(sequences[i].first, sequences[j].first);
            }
        }
        std::ofstream f(FLAGS_o, std::ios::app);
        for (uint32_t i = 0; i < epoch * BLOCK_SIZE; ++i) {
            for (uint32_t j = epoch * BLOCK_SIZE; j < max_ind; ++j) {
                f << sequences[i].second << "," << sequences[j].second << ", "
                  << distances[i * BLOCK_SIZE + j - epoch * BLOCK_SIZE] << std::endl;
            }
        }
        for (uint32_t i = epoch * BLOCK_SIZE; i < max_ind; ++i) {
            for (uint32_t j = i + 1; j < max_ind; ++j) {
                f << sequences[i].second << "," << sequences[j].second << ", "
                  << distances2[(i - epoch * BLOCK_SIZE)][j - epoch * BLOCK_SIZE] << std::endl;
            }
        }
        f.close();

        std::cout << ".";
    }
}
