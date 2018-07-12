#ifndef TEST_DATA_GEN_H
#define TEST_DATA_GEN_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "restServer.hpp"

// Type: one of "random", "const"
// Value: the value of the constant
//

/**
 * @class testDataGen
 * @brief Generate test data as a standin for DPDK.
 *
 * @par Buffers
 * @buffer network_out_buf Buffer to fill
 *         @buffer_format any format
 *         @buffer_metadata chimeMetadata
 *
 * @conf  type                  String. "const", "random", or "ramp"
 * @conf  value                 Int.
 * @conf  wait                  Bool, default True. Produce data a set cadence.
 *                              Otherwise just as fast as possible.
 * @conf  samples_per_data_set  Int. How often to produce data.
 * @conf  stream_id             Int.
 * @conf  num_frames            Int. How many frames to produce. Default inf.
 * @conf  rest_mode             String. "none" (default), "start", or "step.
 *                              How to interact with rest commands to trigger
 *                              data production.
 *
 * @par REST Endpoints
 * @endpoint <unique_name>/generate_test_data
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
class testDataGen : public KotekanProcess {
public:
    testDataGen(Config& config, const string& unique_name,
                bufferContainer &buffer_container);
    ~testDataGen();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;
private:
    void rest_callback(connectionInstance& conn, nlohmann::json& request);
    bool can_i_go(int frame_id_abs);
    struct Buffer *buf;
    std::string type;
    std::string endpoint;
    int value;
    int step_to_frame;
    bool _pathfinder_test_mode;
    int samples_per_data_set;
    bool wait;
    std::string rest_mode;
    int num_frames;
    int stream_id;
};

#endif
