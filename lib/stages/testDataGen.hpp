#ifndef TEST_DATA_GEN_H
#define TEST_DATA_GEN_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "Telescope.hpp"       // for stream_t
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance
#include "visUtil.hpp"         // for StatTracker

#include "json.hpp" // for json

#include <memory>   // for shared_ptr
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class testDataGen
 * @brief Generate test data as a standin for DPDK.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill
 *         @buffer_format any format
 *         @buffer_metadata chimeMetadata
 *
 * @conf  type                  String. "const", "random", "random_signed", "ramp", or "tpluse".
 * @conf  value                 Int. Required for type "const" and "ramp".
 * @conf  values                Vector of ints. Only used for type "onehot" - sets the array element
 * to a different value for each frame; loops through the values.
 * @conf  seed                  Int. For type "random", "random_signed", and "onehot".  If non-zero,
 * seeds the random number generator on startup for reproducible results.
 * @conf  reuse_random          Bool, default False.  For "random" types, only generate each random
 * block once, then keep re-using it.
 * @conf  wait                  Bool, default True. Produce data a set cadence.
 *                              Otherwise just as fast as possible.
 * @conf  samples_per_data_set  Int. How often to produce data.
 * @conf  stream_id             Int.
 * @conf  num_frames            Int. How many frames to produce. Default inf.
 * @conf  num_freq_in_frame     Int. Number of frequencies in each GPU frame.
 * @conf  rest_mode             String. "none" (default), "start", or "step.
 *                              How to interact with rest commands to trigger
 *                              data production.
 * @conf  num_links             Int.  How many links are being simulated, impacts
 *                              the rate of data generated in the wait = true mode.
 *
 * @par REST Endpoints
 * @endpoint /\<unique_name\>/generate_test_data
 *             ``POST`` Triggers the generation of data.
 *              Requires json values  `num_frames` (integer)
 *
 * If `rest_mode` is "start" or "step", data generation will not start right away
 * but will wait for a rest trigger providing `num_frames`. In "start" mode this
 * will initiate the stream of data. In "step" mode this will trigger `num_frames`
 * frames to be generated.
 *
 * @author Andre Renard, Kiyoshi Masui
 */
class testDataGen : public kotekan::Stage {
public:
    testDataGen(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~testDataGen();
    void main_thread() override;

private:
    void rest_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    bool can_i_go(int frame_id_abs);
    Buffer* buf;
    std::string type;
    std::string endpoint;
    int value;
    std::vector<int> _value_array;
    float fvalue;
    std::vector<float> _fvalue_array;
    int step_to_frame;
    bool _pathfinder_test_mode;
    int samples_per_data_set;
    bool wait;
    std::string rest_mode;
    int num_frames;
    bool _reuse_random;
    size_t _num_freq_in_frame;
    stream_t stream_id;
    uint32_t _first_frame_index;
    uint32_t num_links;
    int _seed;
    std::vector<int> _array_shape;
    std::vector<std::string> _dim_name;

    // kotekan trackers example
    std::shared_ptr<StatTracker> timer;
};

#endif
