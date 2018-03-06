/*****************************************
@file
@brief Miscellaneous utils for the receiver code.
- Types for index_maps in the HDF5 output
- Routines for dealing with times as doubles. This is typically more than enough precision.
- Decoding the GPU buffer, and copying out the data into packed form.
- Parsing the input_reorder block in the config files.
- Figuring out struct alignments
- Calculating moving averages.
*****************************************/
#ifndef VIS_UTIL_HPP
#define VIS_UTIL_HPP

#include <complex>
#include <cstdint>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <vector>

#include "gsl-lite.hpp"
#include "json.hpp"

#include "Config.hpp"
#include "buffer.h"

using json = nlohmann::json;

/// Define an alias for the single precision complex type
using cfloat = typename std::complex<float>;

/// Aliased type for storing the layout of members in a struct
using struct_layout = typename std::map<std::string, std::pair<size_t, size_t>>;


/**
 * @brief Frequency index map type
*/
struct freq_ctype {
    /// Centre of frequency channel in MHz
    double centre;
    /// Width of frequency channel in MHz
    double width;
};

/**
 * @brief Correlator input index map
 */
struct input_ctype {
    /**
     * @brief Allow initialisation from a std::string
     * @param id     Input ID
     * @param serial Input serial number
     */
    input_ctype(uint16_t id, std::string serial);

    /// Input ID
    uint16_t chan_id;
    /// Correlator input serial number
    char correlator_input[32];
};

/**
 * @brief Time index map
 */
struct time_ctype {
    /// FPGA sequence number
    uint64_t fpga_count;
    /// UNIX time
    double ctime;
};

/**
 * @brief Product index map type.
 */
struct prod_ctype {
    /// Index of input A
    uint16_t input_a;
    /// Index of input B
    uint16_t input_b;
};

/**
 * @brief Index into a flattened upper matrix triangle.
 * @param  i Row index.
 * @param  j Column index.
 * @param  n Size of matrix.
 * @return   Index into flattend matrix.
 */
inline uint32_t cmap(uint32_t i, uint32_t j, uint32_t n) {
    return (n * (n + 1) / 2) - ((n - i) * (n - i + 1) / 2) + (j - i);
}

/**
 * Get the index of a particular product into the GPU blocked output.
 * @param  i     Row index.
 * @param  j     Column index.
 * @param  block Block size.
 * @param  N     Number if inputs.
 * @return       Index into blocked array.
 */
inline uint32_t prod_index(uint32_t i, uint32_t j, uint32_t block, uint32_t N) {
    uint32_t b_ix = cmap(i / block, j / block, N / block);

    return block * block * b_ix + (i % block) * block + (j % block);
}


/**
 * @brief Convert timeval type into UNIX time as a double.
 * @param  tv Time as timeval.
 * @return    Time as double.
 */
inline double tv_to_double(const timeval & tv) {
    return (tv.tv_sec + 1e-6 * tv.tv_usec);
}


/**
 * @brief Convert timespec type into UNIX time as a double.
 * @param  ts Time as timespec.
 * @return    Time as double.
 */
inline double ts_to_double(const timespec & ts) {
    return (ts.tv_sec + 1e-9 * ts.tv_nsec);
}


/**
 * @brief Convert a UNIX time as double into a timespec.
 * @param  dtime  Time as double.
 * @return        Time as timespec.
 **/
inline timespec double_to_ts(double dtime) {
    return {(int64_t)dtime, (int64_t)(fmod(dtime, 1.0) * 1e9)};
}


/**
 * @brief Get the current UNIX time as a double.
 * @return  UNIX time as double.
 **/
inline double current_time() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts_to_double(ts);
}


/**
 * @brief Copy the visibility triangle into a contiguous array.
 * @param inputdata Input data to copy out.
 * @param inputmap  Vector of feed indices to extract.
 * @param block     Block size.
 * @param N         Number of inputs in input data.
 * @param output    Region of memory to write into.
 */
void copy_vis_triangle(
    const int32_t * inputdata, const std::vector<uint32_t>& inputmap,
    size_t block, size_t N, gsl::span<cfloat> output
);


/**
 * @brief Parse the reordering configuration section
 * @param config    Configuration handle.
 * @param base_path Path into YAML file to search from.
 * @return          Tuple containing a vector of the input reorder map, and a
 *                  vector of the input labels for the index map.
 */
std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> parse_reorder_default(Config& config, const std::string base_path);

/**
 * @brief Return the next aligned location for a given type size
 * @param  offset Start offset.
 * @param  size   Item size.
 * @return        Next aligned offset.
 */
size_t _member_alignment(size_t offset, size_t size);

 /**
  * @brief Calculate the alignment of members in a struct and its total size.
  *
  * @param  members  A vector of tupeles of `name`, `element_size` and `num_elements`.
  * @return          A map of member name to start and end in bytes of each
  *                  member. The total size is packed into `"_struct"`.
  */
struct_layout struct_alignment(
    std::vector<std::tuple<std::string, size_t, size_t>> members
);


/**
 * @class movingAverage
 * @brief Calculate an exponentially weighted moving average of a time series.
 * 
 * @author Richard Shaw
 **/
class movingAverage {

public:

    /**
     * @brief Create a moving average calculation.
     * 
     * @param  length  The length scale to average over. This is defined as
     *                 the lag at which all newer samples carry the same weight as all earlier
     *                 samples. Or equivalently the distance at which the weight per sample has
     *                 decreased by a factor of two.
     **/
    movingAverage(double length=4.0);

    /**
     * @brief Add a new sample in the time series.
     * 
     * @param  value  The sample to add.
     **/
    void add_sample(double value);

    /**
     * @brief Return the moving average of the current set of samples.
     * 
     * @returns  The current moving average.
     **/
    double average();

private:
    double current_value;
    double alpha;

    bool initialised = false;
};


#endif
