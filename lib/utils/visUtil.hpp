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
#include <functional>

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
     * @brief Default constructor.
     **/
    input_ctype();

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
 * @brief Stack index map type (stack -> product)
 */
struct stack_ctype {
    /// Index of an example product
    uint32_t prod;
    /// Conjugate before stack
    bool conjugate;
};

/**
 * @brief Reverse stack map (product -> stack)
 */
struct rstack_ctype {
    /// Index of stack this product goes into
    uint32_t stack;
    /// Conjugate before stack
    bool conjugate;
};

/// Comparison operator for stacks
bool operator!=(const rstack_ctype& lhs, const rstack_ctype& rhs);


// Conversions of the index types to json
void to_json(json& j, const freq_ctype& f);
void to_json(json& j, const input_ctype& f);
void to_json(json& j, const prod_ctype& f);
void to_json(json& j, const time_ctype& f);
void to_json(json& j, const stack_ctype& f);
void to_json(json& j, const rstack_ctype& f);

void from_json(const json& j, freq_ctype& f);
void from_json(const json& j, input_ctype& f);
void from_json(const json& j, prod_ctype& f);
void from_json(const json& j, time_ctype& f);
void from_json(const json& j, stack_ctype& f);
void from_json(const json& j, rstack_ctype& f);

// Conversion of std::complex<T> to and from json
namespace std {
    template<class T>
    void to_json(json& j, const std::complex<T>& p) {
        j = json{{"real", p.real()}, {"imag", p.imag()}};
    }

    template<class T>
    void from_json(const json& j, std::complex<T>& p) {
        p = std::complex<T>{j.at("real").get<T>(), j.at("imag").get<T>()};
    }
}

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
 * @brief Convert a product index to an input pair.
 * @param k Product index.
 * @param n Total number of inputs.
 * @return Product pair indices.
 *
 * @todo This is super inefficient.
 **/
inline prod_ctype icmap(uint32_t k, uint16_t n) {
    uint16_t ii;
    for (ii = 0; ii < n; ii++) {
        if (cmap(ii, n - 1, n) >= k) {
            break;
        }
    }

    uint16_t j = k - cmap(ii, ii, n) + ii;
    return {ii, j};
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
 * @brief Subtraction of two timespec structs.
 * @param  a  Time as timespec.
 * @param  b  Time as timespec.
 * @return    a - b as timespec.
 **/
inline timespec operator-(const timespec & a, const timespec & b)
{
    if(a.tv_nsec < b.tv_nsec)
        return {a.tv_sec - b.tv_sec - 1, 1000000000 + a.tv_nsec - b.tv_nsec};
    else
        return {a.tv_sec - b.tv_sec, a.tv_nsec - b.tv_nsec};
}


/**
 * @brief Comparison of two timespec structs.
 * @param  a  Time as timespec.
 * @param  b  Time as timespec.
 * @return    True if (a == b), False otherwise.
 **/
inline bool operator==(const timespec & a, const timespec & b) {
    return (a.tv_sec == b.tv_sec && a.tv_nsec == b.tv_nsec);
}


/**
 * @brief Comparison of two timespec structs.
 * @param  a  Time as timespec.
 * @param  b  Time as timespec.
 * @return    True if (a > b), False otherwise.
 **/
inline bool operator>(const timespec & a,const timespec & b) {
    return (a.tv_sec > b.tv_sec ||
            (a.tv_sec == b.tv_sec && a.tv_nsec > b.tv_nsec));
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
 * @brief Apply a function over the visibility triangle.
 *
 * This function is best by passing a lambda function/closure that does the
 * computation you want. It allows you to avoid doing the index computation
 * yourself.
 *
 * @param inputmap  Vector of feed indices to use.
 * @param block     Block size.
 * @param N         Number of inputs in input data.
 * @param f         Function to apply. It takes three arguments.
 *                    - The product index into the correlation triangle.
 *                    - The same product in the GPU packed data.
 *                    - Whether we need to conjugate to map between the two.
 */
void map_vis_triangle(const std::vector<uint32_t>& inputmap,
    size_t block, size_t N, std::function<void(int32_t, int32_t, bool)> f
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
 * @brief Calculate the norm of a complex number (i.e. |z|^2).
 *
 * In theory std::norm should do this, but the version in libstdc++ is super
 * slow.
 *
 * @param z  Number to find the norm of.
 * @returns  Norm of z.
 **/
template<typename T>
inline T fast_norm(const std::complex<T>& z)
{
    T r = std::real(z);
    T i = std::imag(z);
    return (r * r + i * i);
}


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

// Zip, unzip adapted from https://gist.github.com/yig/32fe51874f3911d1c612
// TODO: write a more generalised version with variadic arguments
/**
 * @brief Zip together two vectors to create a vector of the pairs.
 *
 * This is similar to using Python's zip(first, second). It will zip up until
 * the point that one of the vectors ends.
 *
 * @param first  The first vector to zip.
 * @param second The second vector to zip.
 *
 * @returns A vector of the zipped pairs.
 **/
template< typename T, typename U >
inline std::vector< std::pair< T, U > > zip( const std::vector< T >& first, const std::vector< U >& second )
{
    size_t min_size = std::min(first.size(), second.size());
    std::vector< std::pair<T, U> > result;
    result.reserve(min_size);

    for( unsigned int i = 0; i < min_size; ++i )
    {
        result.push_back({first[i], second[i]});
    }
    return result;
}

/**
 * @brief Split a vector of pairs into a pair of vectors.
 *
 * This is similar to using Python's zip(*both).
 *
 * @param both A vector of pairs.
 *
 * @returns A pair of the unzipped vectors.
 **/
template< typename T, typename U >
inline std::pair< std::vector<T>, std::vector<U> > unzip(
    const std::vector< std::pair<T, U> >& both)
{
    std::pair< std::vector< T >, std::vector< U > > result;
    result.first.reserve(both.size());
    result.second.reserve(both.size());

    for (auto& p : both) {
        result.first.push_back(p.first);
        result.second.push_back(p.second);
    }
    return result;
}

/**
 * @brief Apply a function 1->1 over a vector.
 *
 * @param vec  Vector to use.
 * @param func Function to apply.
 *
 * @returns Vector with the mapped elements.
 **/
template<typename T, typename U>
inline std::vector<U> func_map(const std::vector<T>& vec,
                               std::function<U(const T&)> func)
{
    std::vector<U> ret;
    ret.reserve(vec.size());

    for(const T& x : vec) {
        ret.push_back(func(x));
    }
    return ret;
}

/**
 * @brief Splits a string based on a regex delimiter
 *
 * Aside: how is something like this not in std::string?
 *
 * @param input The string to split
 * @param reg The regex string delimiter
 * @return A vector of strings as split by the delimiter
 */
std::vector<std::string> regex_split(const std::string input, const std::string reg);


#endif
