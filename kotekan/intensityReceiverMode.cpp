#include "intensityReceiverMode.hpp"
#include "buffers.h"
#include "chrxUplink.hpp"
#include "gpuPostProcess.hpp"
#include "networkOutputSim.hpp"
#include "nullProcess.hpp"
#include "vdifStream.hpp"
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
#include "streamSingleDishVDIF.hpp"
#include "networkInputPowerStream.hpp"
#include "processFactory.hpp"

#include <vector>
#include <string>

using std::string;
using std::vector;

intensityReceiverMode::intensityReceiverMode(Config& config) : kotekanMode(config) {
}

intensityReceiverMode::~intensityReceiverMode() {
}

void intensityReceiverMode::initalize_processes() {

    // Config values:
    int freqs = config.get_int("/","num_freq");
    int elems = config.get_int("/","num_elements");
    int buffer_depth = config.get_int("/","buffer_depth" );

    int integration_length = config.get_int("/","integration_length");
    int timesteps_in = config.get_int("/","samples_per_data_set");
    int timesteps_out = timesteps_in / integration_length;

    bufferContainer buffer_container;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool *pool;
    pool = (struct InfoObjectPool *)malloc(sizeof(struct InfoObjectPool));
    add_info_object_pool(pool);
    create_info_pool(pool, 5*buffer_depth, freqs, elems);

    DEBUG("Creating buffers...");
    // Create buffers.

    int buffer_size = sizeof(IntensityPacketHeader) + sizeof(uint)*freqs*elems;

    struct Buffer *input_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(input_buffer);
    create_buffer(input_buffer,
                  buffer_depth,
                  buffer_size,
                  pool,
                  "input_power_buf");
    buffer_container.add_buffer("input_power_buf", input_buffer);

    struct Buffer *output_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(output_buffer);
    create_buffer(output_buffer,
                  buffer_depth,
                  buffer_size,
                  pool,
                  "input_power_buf");
    buffer_container.add_buffer("output_power_buf", output_buffer);


    processFactory process_factory(config, buffer_container);
    vector<KotekanProcess *> processes = process_factory.build_processes();

    for (auto process: processes) {
        add_process(process);
    }

}
