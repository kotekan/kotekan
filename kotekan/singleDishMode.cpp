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
    config.dump_config();
//    config.find_parameter("dualpol_sumsq","integration_length");

    int test_value = config.get_int("/vdif_stream/", "num_disks");
    INFO("TEST VALUE: %d", test_value);

    // Config values:
    int num_total_freq = config.get_int("/processing", "num_total_freq");
    int num_elements = config.get_int("/processing", "num_elements");
    int buffer_depth = config.get_int("/processing", "buffer_depth");
    int num_fpga_links = config.get_int("/dpdk", "num_links");
    int num_disks = config.get_int("/raw_capture", "num_disks");

    int integration_length = config.get_int("/raw_capture", "integration_length");
    int timesteps_in = config.get_int("/processing", "samples_per_data_set");
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
    create_info_pool(pool, 5 * num_disks * buffer_depth, num_total_freq, num_elements);

    DEBUG("Creating buffers...");
    // Create buffers.

    struct Buffer *vdif_input_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(vdif_input_buffer);
    create_buffer(vdif_input_buffer,
                  buffer_depth * num_disks,
                  timesteps_in * num_elements * (num_total_freq + sizeof(VDIFHeader)),
                  num_fpga_links,
                  3,
                  pool,
                  "vdif_buf");
    host_buffers.add_buffer("vdif_buf", vdif_input_buffer);

    struct Buffer *output_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(output_buffer);
    create_buffer(output_buffer,
                  buffer_depth,
                  timesteps_out * (num_total_freq + 1) * num_elements * sizeof(float),
                  1, //producers
                  2, //consumers
                  pool,
                  "output_power_buf");
    host_buffers.add_buffer("output_power_buf", output_buffer);

    /*add_process((KotekanProcess *) new dpdkWrapper(config, "vdif_capture", &vdif_input_buffer, "vdif"));
    //add_process((KotekanProcess*) new simVdifData(config, *vdif_input_buffer));
    //add_process((KotekanProcess*) new nullProcess(config, *vdif_input_buffer));
    add_process((KotekanProcess*) new streamSingleDishVDIF(config, *vdif_input_buffer));
    for (int i = 0; i < num_disks; ++i) {
        // See nDiskFileWrite.cpp, this will be changed to just one process.
        INFO("Adding nDiskFileWrite with ID %d", i);
        add_process((KotekanProcess*) new nDiskFileWrite(config, "disk_writer_i", *vdif_input_buffer, i, "test_dataset"));
    }
    add_process((KotekanProcess*) new computeDualpolPower(config, "compute_power", *vdif_input_buffer, *output_buffer));
    //add_process((KotekanProcess*) new nullProcess(config, *output_buffer));
    add_process((KotekanProcess*) new networkPowerStream(config, "stream_power", *output_buffer));
     */

//    string instrument_name = config.get_string("/raw_capture/instrument_name");
//    for (int i = 0; i<2; i++){
//      add_process((KotekanProcess*) new networkPowerStream(config, i, *output_buffer));
//    }
}
