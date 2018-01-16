#include "airspyMode.hpp"
#include "buffers.h"
#include "chrxUplink.hpp"
#include "gpuPostProcess.hpp"
#include "networkOutputSim.hpp"
#include "nullProcess.hpp"
#include "vdifStream.hpp"
#include "network_dpdk.h"
#include "util.h"
#include "testDataCheck.hpp"
#include "testDataGen.hpp"
#include "rawFileRead.hpp"
#include "rawFileWrite.hpp"
#include "pyPlotResult.hpp"
#include "simVdifData.hpp"
#include "computeDualpolPower.hpp"
#include "networkPowerStream.hpp"
#include "vdif_functions.h"
#include "dpdkWrapper.hpp"
#include "processFactory.hpp"

#include <vector>
#include <string>
#include <libairspy/airspy.h>

using std::string;
using std::vector;

airspyMode::airspyMode(Config& config) : kotekanMode(config) {
}

airspyMode::~airspyMode() {
}

void airspyMode::initalize_processes() {

    // Config values:
    int num_total_freq = config.get_int("/", "num_freq");
    int num_elements = config.get_int("/", "num_elements");
    int buffer_depth = config.get_int("/", "buffer_depth");
    int num_fpga_links = config.get_int("/", "num_links");
    int integration_length = config.get_int("/", "integration_length");
    int timesteps_in = config.get_int("/", "samples_per_data_set");
    int timesteps_out = timesteps_in / integration_length;

    bufferContainer buffer_container;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool *pool;
    pool = (struct InfoObjectPool *)malloc(sizeof(struct InfoObjectPool));
    add_info_object_pool(pool);
    create_info_pool(pool, 5 * buffer_depth, num_total_freq, num_elements);

    DEBUG("Creating buffers...");
    // Create buffers.

    struct Buffer *airspy_input_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(airspy_input_buffer);
    create_buffer(airspy_input_buffer,
                  buffer_depth,
                  (65536*4),
                  pool,
                  "airspy_input_buf");
    buffer_container.add_buffer("airspy_input_buf", airspy_input_buffer);

    processFactory process_factory(config, buffer_container);
    vector<KotekanProcess *> processes = process_factory.build_processes();

    for (auto process: processes) {
        add_process(process);
    }
}
