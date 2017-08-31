#include "singleDishMode.hpp"
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

using std::string;
using std::vector;

singleDishMode::singleDishMode(Config& config) : kotekanMode(config) {
}

singleDishMode::~singleDishMode() {
}

void singleDishMode::initalize_processes() {

    // Config values:
    int num_total_freq = config.get_int("/", "num_freq");
    int num_elements = config.get_int("/", "num_elements");
    int buffer_depth = config.get_int("/", "buffer_depth");
    int num_fpga_links = config.get_int("/", "num_links");
    int num_disks = config.get_int("/raw_capture", "num_disks");

    int integration_length = config.get_int("/", "integration_length");
    int timesteps_in = config.get_int("/", "samples_per_data_set");
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

    bufferContainer buffer_container;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool *pool;
    pool = (struct InfoObjectPool *)malloc(sizeof(struct InfoObjectPool));
    add_info_object_pool(pool);
    create_info_pool(pool, 5 * num_disks * buffer_depth, num_total_freq, num_elements);

    DEBUG("Creating buffers...");
    // Create buffers.

    struct Buffer *vdif_input_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(vdif_input_buffer);
    create_buffer(vdif_input_buffer,
                  buffer_depth * num_disks,
                  timesteps_in * num_elements * (num_total_freq + sizeof(VDIFHeader)),
                  pool,
                  "vdif_input_buf");
    buffer_container.add_buffer("vdif_input_buf", vdif_input_buffer);

    struct Buffer *output_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(output_buffer);
    create_buffer(output_buffer,
                  buffer_depth,
                  timesteps_out * (num_total_freq + 1) * num_elements * sizeof(float),
                  pool,
                  "output_power_buf");
    buffer_container.add_buffer("output_power_buf", output_buffer);

    processFactory process_factory(config, buffer_container);
    vector<KotekanProcess *> processes = process_factory.build_processes();

    for (auto process: processes) {
        add_process(process);
    }
}
