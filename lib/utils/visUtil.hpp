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


#include "Config.hpp" // for Config
#include "buffer.hpp" // for Buffer

#include "fmt.hpp"      // for format_context, formatter
#include "gsl-lite.hpp" // for span
#include "json.hpp"     // for json

#include <algorithm> // for max
#include <chrono>
#include <complex>     // for complex, imag, real
#include <cstdint>     // for uint32_t, uint16_t, int64_t, int32_t, uint64_t
#include <cstdlib>     // for size_t, (anonymous), div
#include <deque>       // for deque
#include <functional>  // for function
#include <iosfwd>      // for ostream
#include <map>         // for map
#include <math.h>      // for fmod
#include <memory>      // for unique_ptr
#include <mutex>       // for mutex, lock_guard
#include <string>      // for string
#include <sys/time.h>  // for timeval, CLOCK_REALTIME
#include <sys/types.h> // for __syscall_slong_t, suseconds_t, time_t
#include <time.h>      // for timespec, clock_gettime
#include <tuple>       // for tuple, tie
#include <type_traits> // for enable_if_t, is_integral, make_unsigned
#include <utility>     // for pair
#include <vector>      // for vector

/// Define an alias for the single precision complex type
using cfloat = typename std::complex<float>;

#if defined(WITH_CUDA)
#include <cuda_fp16.h>
using float16_t = __half;
#define KOTEKAN_FLOAT16 1
#else
#include <float.h>
#if defined __FLT16_MAX__
using float16_t = _Float16;
#else
#define KOTEKAN_FLOAT16 0
#endif
#endif

/// Aliased type for storing the layout of members in a struct
/// The first element of the pair is the total struct size, the second is a map
/// associating the type T member labels with their offsets
template<typename T>
using struct_layout = typename std::pair<size_t, std::map<T, std::pair<size_t, size_t>>>;


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

/// Comparison operator for products
inline bool operator==(const prod_ctype& lhs, const prod_ctype& rhs) {
    return (lhs.input_a == rhs.input_a) && (lhs.input_b == rhs.input_b);
}

/**
 * @brief Comparison of two time_ctype structs.
 *
 * Note this compares only the FPGA counts.
 *
 * @param  a  Time a.
 * @param  b  Time b.
 * @return    The comparison result.
 **/
inline bool operator<(const time_ctype& a, const time_ctype& b) {
    return (a.fpga_count < b.fpga_count);
}


/**
 * @brief Comparison of two time_ctype structs.
 *
 * Note this compares only the FPGA counts.
 *
 * @param  a  Time a.
 * @param  b  Time b.
 * @return    The comparison result.
 **/
inline bool operator>(const time_ctype& a, const time_ctype& b) {
    return (a.fpga_count > b.fpga_count);
}

// Conversions of the index types to json
void to_json(nlohmann::json& j, const freq_ctype& f);
void to_json(nlohmann::json& j, const input_ctype& f);
void to_json(nlohmann::json& j, const prod_ctype& f);
void to_json(nlohmann::json& j, const time_ctype& f);
void to_json(nlohmann::json& j, const stack_ctype& f);
void to_json(nlohmann::json& j, const rstack_ctype& f);

void from_json(const nlohmann::json& j, freq_ctype& f);
void from_json(const nlohmann::json& j, input_ctype& f);
void from_json(const nlohmann::json& j, prod_ctype& f);
void from_json(const nlohmann::json& j, time_ctype& f);
void from_json(const nlohmann::json& j, stack_ctype& f);
void from_json(const nlohmann::json& j, rstack_ctype& f);

// Conversion of std::complex<T> to and from json
namespace std {
template<class T>
void to_json(nlohmann::json& j, const std::complex<T>& p) {
    j = nlohmann::json{p.real(), p.imag()};
}

template<class T>
void from_json(const nlohmann::json& j, std::complex<T>& p) {
    p = std::complex<T>{j.at(0).get<T>(), j.at(1).get<T>()};
}
} // namespace std

/**
 * @brief Get type name of a JSON value.
 * Returns a string with the name of the given json value type. Can be one of:
 * integer, float or value.type_name().
 * @param value A JSON value.
 * @return Type name.
 */
std::string json_type_name(nlohmann::json& value);

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
    uint32_t num_blocks1 = ((N - 1) / block) + 1; // Blocks needed to tile 1D
    uint32_t b_ix = cmap(i / block, j / block, num_blocks1);

    return block * block * b_ix + (i % block) * block + (j % block);
}


/**
 * @brief Convert timeval type into UNIX time as a double.
 * @param  tv Time as timeval.
 * @return    Time as double.
 */
inline double tv_to_double(const timeval& tv) {
    return (tv.tv_sec + 1e-6 * tv.tv_usec);
}


/**
 * @brief Convert timespec type into UNIX time as a double.
 * @param  ts Time as timespec.
 * @return    Time as double.
 */
inline double ts_to_double(const timespec& ts) {
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
 * @brief Convert a UNIX time as double into a timeval.
 * @param  dtime  Time as double.
 * @return        Time as timeval.
 **/
inline timeval double_to_tv(double dtime) {
    return {(time_t)dtime, (suseconds_t)(fmod(dtime, 1.0) * 1e6)};
}

/**
 * @brief Division and positive modulus of two integers.
 *
 * @param  a  Dividend.
 * @param  n  Divisor.
 *
 * @return    Pair of (a / n, a mod n) with a/n defined to round down.
 **/
template<typename T>
inline std::pair<T, T> divmod_pos(T a, T n) {
    T d = a / n; // Compiler can usually optimise these into a single instruction
    T m = a % n;

    return {d - (a < 0), (m + n) % n};
}

/**
 * @brief Add an offset to a timespec.
 * @param t     timespec to modify.
 * @param nsec  Number of nsec to add (can be negative).
 * @return      Modified timespec.
 **/
inline timespec add_nsec(const timespec& t, const long nsec) {
    auto dm = divmod_pos(t.tv_nsec + nsec, 1000000000L);
    return {t.tv_sec + dm.first, dm.second};
}

/**
 * @brief Subtraction of two timespec structs.
 * @param  a  Time as timespec.
 * @param  b  Time as timespec.
 * @return    a - b as timespec.
 **/
inline timespec operator-(const timespec& a, const timespec& b) {
    auto dm = divmod_pos(a.tv_nsec - b.tv_nsec, 1000000000L);
    return {a.tv_sec - b.tv_sec + dm.first, dm.second};
}

/**
 * @brief Addition of two timespec structs.
 * @param  a  Time as timespec.
 * @param  b  Time as timespec.
 * @return    a + b as timespec.
 **/
inline timespec operator+(const timespec& a, const timespec& b) {
    // Use std::div instead of divmod_pos to save the extra instructions.
    auto ns_div = std::div(a.tv_nsec + b.tv_nsec, 1000000000L);
    return {a.tv_sec + b.tv_sec + ns_div.quot, ns_div.rem};
}

/**
 * @brief Comparison of two timespec structs.
 * @param  a  Time as timespec.
 * @param  b  Time as timespec.
 * @return    True if (a == b), False otherwise.
 **/
inline bool operator==(const timespec& a, const timespec& b) {
    return (a.tv_sec == b.tv_sec && a.tv_nsec == b.tv_nsec);
}


/**
 * @brief Comparison of two timespec structs.
 * @param  a  Time as timespec.
 * @param  b  Time as timespec.
 * @return    True if (a > b), False otherwise.
 **/
inline bool operator>(const timespec& a, const timespec& b) {
    return (a.tv_sec > b.tv_sec || (a.tv_sec == b.tv_sec && a.tv_nsec > b.tv_nsec));
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
 * @brief Calculate the size of GPU packed data.
 *
 * @note This is the length of a *single* frequency.
 *
 * @param  N      The number of elements.
 * @param  block  The size of a block.
 * @return        The size of the packd GPU data.
 **/
inline constexpr uint32_t gpu_N2_size(uint32_t N, uint32_t block) {
    const auto num_blocks1 = ((N - 1) / block) + 1;               // Blocks per side
    const auto num_blocks2 = num_blocks1 * (num_blocks1 + 1) / 2; // ... triangle
    return (num_blocks2 * block * block);                         // Total size
}


/**
 * @brief Copy the visibility triangle into a contiguous array.
 * @param inputdata Input data to copy out.
 * @param inputmap  Vector of feed indices to extract.
 * @param block     Block size.
 * @param N         Number of inputs in input data.
 * @param output    Region of memory to write into.
 */
void copy_vis_triangle(const int32_t* inputdata, const std::vector<uint32_t>& inputmap,
                       size_t block, size_t N, gsl::span<cfloat> output);


/**
 * @brief Apply a function over the visibility triangle.
 *
 * This will call a function for every set of indices in a *GPU output buffer*.
 * The function is given the index into the GPU buffer and the correlation
 * triangle. To actually process any data use the a lambda/closure and bind the
 * data you want to be able to access.
 *
 * To support multi-frequency GPU buffer this takes a frequency index which
 * will change the GPU buffer offset. As the visibility buffers are single
 * frequency that offset will be unaffected.
 *
 * @param inputmap  Vector of feed indices to use.
 * @param block     Block size.
 * @param N         Number of inputs in input data.
 * @param freq      Frequency index to use. This just gives an offset into the
 *                  visibility triangle.
 * @param f         Function to apply. It takes three arguments.
 *                    - The product index into the correlation triangle.
 *                    - The same product in the GPU packed data.
 *                    - Whether we need to conjugate to map between the two.
 */
void map_vis_triangle(const std::vector<uint32_t>& inputmap, size_t block, size_t N, uint32_t freq,
                      std::function<void(int32_t, int32_t, bool)> f);


/**
 * @brief Parse the reordering configuration section
 * @param config    Configuration handle.
 * @param base_path Path into YAML file to search from.
 * @return          Tuple containing a vector of the input reorder map, and a
 *                  vector of the input labels for the index map.
 */
std::tuple<std::vector<uint32_t>, std::vector<input_ctype>>
parse_reorder_default(kotekan::Config& config, const std::string base_path);

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
 * @param  members  A vector of tuples of a `label` for the member (can be
 *                  any type, but must be unique per member), `element_size`
 *                  and `num_elements`. `name` can be of any type.
 * @return          A pair, of the total size and the struct layout. The
 *                  layout is a map of member name to start and end in bytes
 *                  of each member.
 */
template<typename T>
struct_layout<T> struct_alignment(std::vector<std::tuple<T, size_t, size_t>> members) {

    T label;
    size_t size, num, end = 0, max_size = 0;

    std::map<T, std::pair<size_t, size_t>> layout;

    for (auto member : members) {
        std::tie(label, size, num) = member;

        // Uses the end of the *last* member
        size_t start = _member_alignment(end, size);
        end = start + size * num;
        max_size = std::max(max_size, size);

        layout[label] = {start, end};
    }

    size_t struct_size = _member_alignment(end, max_size);

    return {struct_size, layout};
}


/**
 * @brief Calculate the norm of a complex number (i.e. |x|^2).
 *
 * In theory std::norm should do this, but the version in libstdc++ is super
 * slow.
 *
 * @param x  Number to find the norm of.
 * @returns  Norm of x.
 **/
template<typename T>
inline T fast_norm(const T& x) {
    return x * x;
}

template<typename T>
inline T fast_norm(const std::complex<T>& z) {
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
    movingAverage(double length = 4.0);

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

/**
 * @class SlidingWindowMinMax
 *
 * @brief Use two deques to keep tracking minimum and maximum values.
 *
 * This class is based on a modified version of:
 * https://www.nayuki.io/page/sliding-window-minimum-maximum-algorithm
 **/
class SlidingWindowMinMax {

public:
    /**
     * @brief Get the current minimum value from the front of min_deque.
     *
     * @return The current minimum value.
     **/
    double get_min();

    /**
     * @brief Get the current maximum value from the front of max_deque.
     *
     * @return The current maximum value.
     **/
    double get_max();

    /**
     * @brief Add a new value to both min_deque and max_deque.
     *        The insert position is based on the input value.
     *        All values greater than the input are removed in min_deque;
     *        All values less than the input are removed in max_deque;
     **/
    void add_tail(double val);

    /**
     * @brief Remove the given value from the front of min_deque or max_deque.
     **/
    void remove_head(double val);

private:
    std::deque<double> min_deque;
    std::deque<double> max_deque;
};

/**
 * @class StatTracker
 *
 * @brief Store samples and compute statistics.
 *
 * There are two ways in this class to implement min/max (i.e., sliding window and brute force).
 * The sliding window approach uses O(1) time to get min/max but spends more time when a sample is
 *added. The brute force approach uses O(n) time to traverse the entire buffer without overhead in
 *add_sample(). "is_optimized" is the flag to switch between two methods (true: sliding window;
 *false: brute force). Normally, if get_min/max() is called often, "is_optimized" should be set to
 *true.
 **/
class StatTracker {

public:
    /**
     * @brief Create a ring buffer.
     *
     * @param name The statistic's name.
     * @param unit Sample unit.
     * @param size The size of the ring buffer.
     * @param is_optimized Flag of min/max optimization.
     **/
    explicit StatTracker(std::string name, std::string unit, size_t size = 100,
                         bool is_optimized = true);

    /**
     * @brief Add a new sample value to the buffer.
     *        if the buffer is full, the new sample will overwrite the earliest one.
     *
     * @param new_val The sample to add.
     **/
    void add_sample(double new_val);

    /**
     * @brief Return the maximum sample based on all values stored in buffer.
     *
     * @return The current maximum sample.
     **/
    double get_max();

    /**
     * @brief Return the minimum sample based on all values stored in buffer.
     *
     * @return The current minimum sample.
     **/
    double get_min();

    /**
     * @brief Return the average value based on all samples stored in buffer.
     *
     * @return The current average value.
     **/
    double get_avg();

    /**
     * @brief Return the standard deviation based on all samples stored in buffer.
     *
     * @return The current standard deviation.
     **/
    double get_std_dev();

    /**
     * @brief Return the last value added, will be NAN if no samples have been added
     *
     * @return The last sample or NAN
     **/
    double get_current();

    /**
     * @brief Return tracker content in json format.
     *
     * @return A json object of tracker content.
     **/
    nlohmann::json get_json();

    /**
     * @brief Return tracker stats in json format.
     *
     * @return A json object of tracker min,max,avg,std.
     **/
    nlohmann::json get_current_json();

private:
    SlidingWindowMinMax min_max;

    struct sample {
        double value;
        std::chrono::system_clock::time_point timestamp;
    };
    std::unique_ptr<sample[]> rbuf;
    size_t end;
    size_t buf_size;
    size_t count;

    double avg;
    double dist;
    double var;
    double std_dev;

    std::string name;
    std::string unit;
    bool is_optimized;

    std::recursive_mutex tracker_lock;
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
template<typename T, typename U>
inline std::vector<std::pair<T, U>> zip(const std::vector<T>& first, const std::vector<U>& second) {
    size_t min_size = std::min(first.size(), second.size());
    std::vector<std::pair<T, U>> result;
    result.reserve(min_size);

    for (unsigned int i = 0; i < min_size; ++i) {
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
template<typename T, typename U>
inline std::pair<std::vector<T>, std::vector<U>> unzip(const std::vector<std::pair<T, U>>& both) {
    std::pair<std::vector<T>, std::vector<U>> result;
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
inline std::vector<U> func_map(const std::vector<T>& vec, std::function<U(const T&)> func) {
    std::vector<U> ret;
    ret.reserve(vec.size());

    for (const T& x : vec) {
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


/**
 * @brief A class for modular arithmetic. Used for holding ring buffer indices.
 *
 * This implements comparison and arithmetic operators for modular arithmetic.
 *
 * @note The binary arithmetic operators only work adding/subtracting normal numbers
 *       to a modular number. They are also *asymmetric*.
 **/
template<typename T>
class modulo {

public:
    // Use an unsigned type for the base
    using Tu = typename std::make_unsigned<T>::type;

    /**
     * @brief Create a new modular number.
     **/
    modulo(Tu n) : _n(n){};

    // Default constructor
    modulo() : modulo(0){};

    /// Assignment of a number into the modular number.
    modulo<T>& operator=(const T& i) {
        _i = i;
        return *this;
    }

    // Increment and decrement
    modulo<T>& operator++() {
        _i++;
        return *this;
    }
    modulo<T>& operator--() {
        _i--;
        return *this;
    }
    modulo<T> operator++(int) {
        modulo<T> t(*this);
        operator++();
        return t;
    }
    modulo<T> operator--(int) {
        modulo<T> t(*this);
        operator--();
        return t;
    }

    template<typename V, typename std::enable_if_t<std::is_integral<V>::value>* = nullptr>
    modulo<T>& operator+=(const V& rhs) {
        _i += rhs;
        return *this;
    }

    template<typename V, typename std::enable_if_t<std::is_integral<V>::value>* = nullptr>
    modulo<T>& operator-=(const V& rhs) {
        _i -= rhs;
        return *this;
    }

    // Add and subtract are *asymmetric*. Must be always be modulo<T> +/- T
    template<typename V, typename std::enable_if_t<std::is_integral<V>::value>* = nullptr>
    friend modulo<T> operator+(modulo<T> lhs, const V& rhs) {
        modulo<T> t(lhs);
        t += rhs;
        return t;
    }
    template<typename V, typename std::enable_if_t<std::is_integral<V>::value>* = nullptr>
    friend modulo<T> operator-(modulo<T> lhs, const V& rhs) {
        modulo<T> t(lhs);
        t -= rhs;
        return t;
    }

    // Comparisons are always false if the bases don't match
    friend bool operator==(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() == rhs.norm());
    }
    friend bool operator!=(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() != rhs.norm());
    }
    friend bool operator<(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() < rhs.norm());
    }
    friend bool operator>(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() > rhs.norm());
    }
    friend bool operator<=(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() <= rhs.norm());
    }
    friend bool operator>=(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() >= rhs.norm());
    }

    /**
     * @brief Return the normalised modular number.
     *
     * @returns The modular number.
     **/
    T norm() const {
        return _i % _n;
    }

    /// Conversion back to type T
    operator T() const {
        return norm();
    }

private:
    // Internally we don't actually keep bother mod'ing the number when
    // we do arithmetic, only at output time.
    T _i = 0;

    // The modular base.
    Tu _n;
};

/// Stream output for modular types
template<typename T>
std::ostream& operator<<(std::ostream& os, const modulo<T>& m) {
    return (os << m.norm());
}


/**
 * @brief Class to hold buffer indices.
 **/
class frameID : public modulo<int> {
public:
    /**
     * @brief Create a frameID for a given buffer.
     *
     * @param buf   Buffer to use.
     **/
    frameID(const Buffer* buf) : modulo<int>(buf->num_frames) {}
};

/**
 *@brief FMT formatter that casts frameIDs to int so that `format("{:d}", frame_id)` works.
 */
namespace fmt {
template<>
struct formatter<frameID> : formatter<int> {
    auto format(const frameID id, format_context& ctx) {
        return formatter<int>::format((int)id, ctx);
    }
};
} // namespace fmt

/**
 * @brief Pretty-printing for terminal or literal python code.
 */
inline std::string format_nice_string(uint8_t x) {
    return fmt::format("{} = 0x{:x}", x, x);
}
inline std::string format_nice_string(uint32_t x) {
    return fmt::format("{} = 0x{:x}", x, x);
}
inline std::string format_nice_string(int x) {
    return fmt::format("{} = 0x{:x}", x, x);
}
#if KOTEKAN_FLOAT16
inline std::string format_nice_string(float16_t x) {
    return fmt::format("{}", (float)x);
}
#endif
inline std::string format_nice_string(float x) {
    return fmt::format("{}", x);
}

inline std::string format_python_string(uint8_t x) {
    return fmt::format("{}", x);
}
inline std::string format_python_string(uint32_t x) {
    return fmt::format("{}", x);
}
#if KOTEKAN_FLOAT16
inline std::string format_python_string(float16_t x) {
    return fmt::format("{}", (float)x);
}
#endif

/**
 * @brief Degrees to radians, as floats
 */
inline float deg2radf(float d) {
    return d * (float)(M_PI / 180.);
}

/**
 * @brief Trig functions for angles in degrees, as floats
 */
inline float cosdf(float d) {
    return cosf(deg2radf(d));
}
inline float sindf(float d) {
    return sinf(deg2radf(d));
}

/**
 * @brief Returns time of day in floating-point seconds.  Useful if you want to measure the time
 * between two events.
 */
inline double gettime() {
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return tp.tv_sec + tp.tv_usec / 1.0e+6;
}

typedef uint8_t int4x2_t;

// Compose a 4+4-bit complex integer
inline constexpr int4x2_t set4(const int8_t lo, const int8_t hi) {
    return (uint8_t(lo) & 0x0f) | ((uint8_t(hi) << 4) & 0xf0);
}
// Compose a 4+4-bit complex integer
inline constexpr int4x2_t set4(const std::array<int8_t, 2> a) {
    return set4(a[0], a[1]);
}

// Decompose a 4+4-bit complex integer
inline constexpr std::array<int8_t, 2> get4(const int4x2_t i) {
    return {int8_t(int8_t((i + 0x08) & 0x0f) - 0x08),
            int8_t(int8_t(((i >> 4) + 0x08) & 0x0f) - 0x08)};
}

#endif
