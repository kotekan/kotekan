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
    int num_total_freq = config.get_int("/processing","num_total_freq");
    int num_elements = config.get_int("/processing","num_elements");
    int buffer_depth = config.get_int("/processing","buffer_depth");
    int num_disks = config.get_int("/raw_capture","num_disks");

    int integration_length = config.get_int("/raw_capture","integration_length");
    int timesteps_in = config.get_int("/processing","samples_per_data_set");
    int timesteps_out = timesteps_in / integration_length;
    string instrument_name = config.get_string("/raw_capture","instrument_name");

    // TODO This needs to move outside of this function, but that
    // requires some more refactoring of the nDisk thread.
    char data_time[64];
    char data_set[150];
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = gmtime(&rawtime);

    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(data_set, sizeof(data_set), "%s_%s_raw", data_time, instrument_name.c_str());

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
