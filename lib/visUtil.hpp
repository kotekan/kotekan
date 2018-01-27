#ifndef VIS_UTIL_HPP
#define VIS_UTIL_HPP

#include <complex>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include "json.hpp"
#include "Config.hpp"


using json = nlohmann::json;

// Structs to represent the datatypes of the index maps
struct freq_ctype {
    double centre;
    double width;
};

struct input_ctype {
    // Allow initialisation from a std::string
    input_ctype(uint16_t id, std::string serial);

    uint16_t chan_id;
    char correlator_input[32];
};

struct time_ctype {
    uint64_t fpga_count;
    double ctime;
};

struct prod_ctype {
    uint16_t input_a;
    uint16_t input_b;
};

// Functions for indexing into the buffer of data
inline uint32_t cmap(uint32_t i, uint32_t j, uint32_t n) {
    return (n * (n + 1) / 2) - ((n - i) * (n - i + 1) / 2) + (j - i);
}

inline prod_ctype icmap(uint32_t k, uint16_t n) {
    uint16_t ii = 0;
    for (ii; ii < n; ii++) {
        if (cmap(ii, n - 1, n) >= k) {
            break;
        }
    }

    uint16_t j = k - cmap(ii, ii, n) + ii;
    return {ii, j};
}

inline uint32_t prod_index(uint32_t i, uint32_t j, uint32_t block, uint32_t N) {
    uint32_t b_ix = cmap(i / block, j / block, N / block);

    return block * block * b_ix + (i % block) * block + (j % block);
}

inline double tv_to_double(const timeval & tv) {
    return (tv.tv_sec + 1e-6 * tv.tv_usec);
}

inline double ts_to_double(const timespec & ts) {
    return (ts.tv_sec + 1e-9 * ts.tv_nsec);
}


// Copy the visibility triangle into a contiguous array
// ... either a preallocated one
void copy_vis_triangle(
    const int32_t * buf, const std::vector<uint32_t>& inputmap,
    size_t block, size_t n, std::complex<float> * output
);

// ... or allocate a vector for it
std::vector<std::complex<float>> copy_vis_triangle(
    const int32_t * buf, const std::vector<uint32_t>& inputmap,
    size_t block, size_t N
);

// Parse the reordering configuration section
std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> parse_reorder(json& j);
std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> default_reorder(size_t num_elements);
std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> parse_reorder_default(Config& config, const std::string base_path);
#endif
