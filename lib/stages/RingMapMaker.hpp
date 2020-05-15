
#ifndef RINGMAP_HPP
#define RINGMAP_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t
#include "datasetState.hpp"    // for stackState, prodState
#include "restServer.hpp"      // for connectionInstance
#include "visUtil.hpp"         // for input_ctype, prod_ctype, time_ctype, stack_ctype, cfloat

#include "fmt.hpp"  // for format
#include "json.hpp" // for json

#include <algorithm> // for copy, max
#include <map>       // for map
#include <math.h>    // for cos
#include <mutex>     // for mutex
#include <stddef.h>  // for size_t
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint32_t, int64_t, uint8_t
#include <string>    // for string
#include <utility>   // for pair
#include <vector>    // for vector

/**
 * @brief Generate a ringmap from a real-time stream of data.
 *
 * Expects frames from the stacked dataset.
 *
 * @par buffers
 * @buffer in_buf The buffer to read from.
 *        @buffer_format VisFrameView
 *        @buffer_metadata VisMetadata
 *
 * @conf feed_sep       Float, default 0.3048. The separation between feeds (in m)
 * @conf apodization    String, default nuttall. The type of window to use for apodization.
 * @conf exclude_autos  Bool, default true. Exclude the autos from the maps.
 *
 *
 * @author Tristan Pinsonneault-Marotte
 */
class RingMapMaker : public kotekan::Stage {

public:
    // Default constructor
    RingMapMaker(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    // Main loop for the process
    void main_thread() override;

    /// REST endpoint to request a map
    void rest_callback(kotekan::connectionInstance& conn, nlohmann::json& json);
    void rest_callback_get(kotekan::connectionInstance& conn);

    /// Abbreviation for RingMap type
    typedef std::vector<std::vector<cfloat>> RingMap;

private:
    void change_dataset_state(dset_id_t ds_id);

    bool setup(size_t frame_id);

    void gen_matrices();

    int64_t resolve_time(time_ctype t);

    inline float wl(float freq) {
        return 299.792458 / freq;
    };

    // Matrix from visibilities to map for every freq (same for each pol)
    std::map<uint32_t, std::vector<cfloat>> vis2map;
    std::map<uint32_t, std::vector<float>> wgt2map;
    // Store the maps and weight maps for every frequency
    std::map<uint32_t, std::vector<std::vector<float>>> map;
    std::map<uint32_t, std::vector<std::vector<float>>> wgt;

    // Visibilities specs
    std::vector<stack_ctype> stacks;
    std::vector<prod_ctype> prods;
    std::vector<input_ctype> inputs;
    std::vector<std::pair<uint32_t, freq_ctype>> freqs;
    std::vector<float> ns_baselines;

    // Dimensions
    uint32_t num_time;
    uint32_t num_pix;
    uint8_t num_pol;
    uint32_t num_stack;
    uint32_t num_bl;

    // Map dimensions and time keeping
    std::vector<float> sinza;
    std::vector<time_ctype> times;
    std::map<double, size_t> times_map;
    modulo<size_t> latest;
    double max_ctime, min_ctime;

    // Dataset ID of incoming stream
    dset_id_t ds_id;

    // Configurable
    float feed_sep;
    std::string apodization;
    bool exclude_autos;

    // Mutex for reading and writing to maps
    std::mutex mtx;

    // Buffer to read from
    Buffer* in_buf;
};

/**
 * @brief Complete stack over redundant baselines from previously stacked data.
 *
 * This is based on baselineCompression.
 *
 * @todo merge this with baselineCompression
 *
 * @par buffers
 * @buffer in_buf The buffer to read from.
 *        @buffer_format VisFrameView
 *        @buffer_metadata VisMetadata
 * @buffer out_buf The buffer to write to.
 *        @buffer_format VisFrameView
 *        @buffer_metadata VisMetadata
 *
 *
 * @author Tristan Pinsonneault-Marotte
 */
class RedundantStack : public kotekan::Stage {

public:
    RedundantStack(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    void main_thread();

private:
    void change_dataset_state(dset_id_t ds_id);

    // dataset states and IDs
    dset_id_t output_dset_id;
    dset_id_t input_dset_id;
    const stackState* old_stack_state_ptr;
    const stackState* new_stack_state_ptr;

    // Buffers
    Buffer* in_buf;
    Buffer* out_buf;
};

/**
 * @brief Complete the redundant stacking.
 *
 * @param inputs The set of inputs.
 * @param prods  The products we are stacking.
 *
 * @returns Stack definition.
 **/
std::pair<uint32_t, std::vector<rstack_ctype>>
full_redundant(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods);

const float pi = std::acos(-1);

// Apodization windows.
// Based on https://github.com/radiocosmology/draco/blob/hybrid-beamform/draco/util/tools.py#L370
const std::map<std::string, std::vector<float>> apod_param = {
    {"uniform", {1., 0., 0., 0.}},
    {"hanning", {0.5, -0.5, 0., 0.}},
    {"hamming", {0.53836, -0.46164, 0., 0.}},
    {"blackman", {0.42, -0.5, 0.08, 0.}},
    {"nuttall", {0.355768, -0.487396, 0.144232, -0.012604}},
    {"blackman_nuttall", {0.3635819, -0.4891775, 0.1365995, -0.0106411}},
    {"blackman_harris", {0.35875, -0.48829, 0.14128, -0.01168}},
};

/**
 * @brief Generate an apodization function over some range.
 *
 * Evaluates a window function that is symmetric around zero.
 * Available functions are listed in the `apod_param` variable.
 *
 * @param x        Vector of float. The locations at which to evaluate the window.
 * @param width    Float, default 1. The distance from zero beyond which window will be set to 0.
 *                      The x parameter will be normalized by this value before evaluating.
 * @param win      String, default nuttall. The type of window to use.
 *
 * @author Tristan Pinsoneault-Marotte, Richard Shaw
 **/
inline std::vector<float> apod(std::vector<float>& x, float width = 1.,
                               const std::string& win = "nuttall") {

    if (width <= 0)
        throw std::runtime_error(fmt::format("Apodization width {:f} <= 0", width));

    std::vector<float> coeff = apod_param.at(win);
    std::vector<float> w;
    for (auto xi : x) {
        // Accept a range of +/- 1
        if (xi < -1 * width || xi > width) {
            w.push_back(0.);
        } else {
            float y = 0.;
            for (unsigned short n = 0; n < coeff.size(); n++) {
                y += coeff[n] * std::cos(pi * (xi / width + 1) * n);
            }
            w.push_back(y);
        }
    }

    return w;
};

#endif
