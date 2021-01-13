#pragma once

#include "util/timer.hpp"
#include "util/utils.hpp"

#include <algorithm>
#include <immintrin.h>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace ts {

enum class HashAlgorithm { uniform, crc32};

template <typename T>
class HashBase {
  public:
    HashBase() {}

    HashBase(T set_size, size_t sketch_dim, size_t hash_size, std::string hash_algorithm)
        : set_size(set_size),
          sketch_dim(sketch_dim),
          hash_size(2 * hash_size),
          hash_algorithm(to_hash_algorithm(hash_algorithm)),
          hashes(std::vector<std::unordered_map<T, T>>(sketch_dim)),
          hash_values(std::vector<std::unordered_set<T>>(sketch_dim)) {
        std::random_device rd;
        rng = std::mt19937(rd());
        rand = std::uniform_int_distribution<T>(0, this->hash_size - 1);
        crc32_base = rand(rng);
    }

    void set_hashes_for_testing(const std::vector<std::unordered_map<T, T>> &h) { hashes = h; }

    HashAlgorithm to_hash_algorithm(std::string name) {
        if (name == "uniform") {
            hash_algorithm = HashAlgorithm::uniform;
        } else if (name == "crc32") {
            hash_algorithm = HashAlgorithm::crc32;
        }
        return hash_algorithm;
    }


  protected:
    T set_size;
    size_t sketch_dim;
    size_t hash_size;

    /**
     * Returns the hash value for the index-th hash function.
     * Since the Hashes are generated on demand.
     */
    uint32_t hash(size_t index, size_t key) {
        T val = -1;
        switch (hash_algorithm) {
            T random_hash;
            case HashAlgorithm::uniform:
                // TODO multiple read Semaphor instead of critical
#pragma omp critical
            {
                if (hashes[index].find(key) != hashes[index].end()) {
                    random_hash = hashes[index][key];
                } else {
                    while (true) {
                        random_hash = rand(rng);
                        if (hash_values[index].find(random_hash) == hash_values[index].end()) {
                            hashes[index][key] = random_hash;
                            hash_values[index].insert(random_hash);
                            break;
                        }
                    }
                }
                assert(random_hash>=0 && random_hash<hash_size && " Hash values are not in [0,set_size-1] range");
                val = random_hash;
            }
                break;
            case HashAlgorithm::crc32:
                val = _mm_crc32_u32((unsigned int)crc32_base,(unsigned int)key);
                val = _mm_crc32_u32((unsigned int)val,(unsigned int)index);
                return val;
        }


        return val;
    }

  private:
    HashAlgorithm hash_algorithm;

    /** Contains the sketch_dim permutations (hashes) that are used to compute the min-hash */
    std::vector<std::unordered_map<T, T>> hashes;
    /** Contains the values used so far for each on-demand permutation */
    std::vector<std::unordered_set<T>> hash_values;
    std::uniform_int_distribution<T> rand;
    std::mt19937 rng;
    uint32_t crc32_base = 341234;
};




} // namespace ts
