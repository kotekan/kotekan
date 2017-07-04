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
    int buffer_depth = config.get_int("/processing/buffer_depth");
    int freqs = config.get_int("/processing/num_frequencies");
    int elems = config.get_int("/processing/num_elements");

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
                  1,
                  1,
                  pool,
                  "input_power_buf");
    host_buffers.add_buffer("input_power_buf", input_buffer);

    add_process((KotekanProcess*) new networkInputPowerStream(config, *input_buffer));
    add_process((KotekanProcess*) new nullProcess(config, *input_buffer));
    //add_process((KotekanProcess*) new networkPowerStream(config, *output_buffer));


}
