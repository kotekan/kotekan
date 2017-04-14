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
    int num_local_freq = config.get_int("/processing/num_local_freq");
    int num_elements = config.get_int("/processing/num_elements");
    int buffer_depth = config.get_int("/processing/buffer_depth");

    int integration_length = config.get_int("/raw_capture/integration_length");
    int timesteps_in = config.get_int("/processing/samples_per_data_set");
    int timesteps_out = timesteps_in / integration_length;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool *pool[1];
    for (int i = 0; i < 1; ++i) {
        pool[i] = (struct InfoObjectPool *)malloc(sizeof(struct InfoObjectPool));
        add_info_object_pool(pool[i]);
    }
    create_info_pool(pool[0], 5 * buffer_depth, num_local_freq, num_elements);




    DEBUG("Creating buffers...");
    // Create buffers.

    struct Buffer *vdif_input_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(vdif_input_buffer);
    create_buffer(vdif_input_buffer,
                  buffer_depth,
                  timesteps_in * num_elements * (num_local_freq + sizeof(VDIFHeader)),
                  1,
                  1,
                  pool[0],
                  "vdif_buf");
    host_buffers.add_buffer("vdif_buf", vdif_input_buffer);

    struct Buffer *output_buffer = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(output_buffer);
    create_buffer(output_buffer,
                  buffer_depth,
                  num_local_freq * 2 * sizeof(float) * timesteps_out,
                  1,
                  1,
                  pool[0],
                  "output_power_buf");
    host_buffers.add_buffer("output_power_buf", output_buffer);

    add_process((KotekanProcess*) new simVdifData(config, *vdif_input_buffer));
//    add_process((KotekanProcess*) new nullProcess(config, *vdif_input_buffer));
    add_process((KotekanProcess*) new computeDualpolPower(config, *vdif_input_buffer, *output_buffer));
//    add_process((KotekanProcess*) new nullProcess(config, *output_buffer));
    add_process((KotekanProcess*) new networkPowerStream(config, *output_buffer));

    /*snprintf(buffer_name, 100, "beamform_output_buf_%d", i);
    create_buffer(beamform_output_buffer[i],
                  links_per_gpu * buffer_depth,
                  2 * timesteps_in * num_adjusted_elements * sizeof(float),
                  1,
                  1,
                  pool[0],
                  buffer_name);
    host_buffers[i].add_buffer("beamform_output_buf", beamform_output_buffer[i]);

    snprintf(buffer_name, 100, "beamform_output_simulate_buf_%d", i);
    create_buffer(beamform_output_simulate_buffer[i],
                  links_per_gpu * buffer_depth,
                  2 * timesteps_in * num_adjusted_elements * sizeof(float),
                  1,
                  1,
                  pool[0],
                  buffer_name); */

    // TODO better management of the buffers so this list doesn't have to change size...

    // Beamform test
    //add_process((KotekanProcess*) new testDataGen(config, *network_input_buffer[i]));
    //add_process((KotekanProcess*) new gpuHSAThread(config, host_buffers[i], i));
    //add_process((KotekanProcess*) new rawFileRead(config, *beamform_output_simulate_buffer[i], false, true,
    //                                                 "/data/test_data/", "beamform_gpu_output_99", "dat"));
    ////add_process((KotekanProcess*) new gpuBeamformSimulate(config, *network_input_buffer[i], *beamform_output_simulate_buffer[i]));
    //add_process((KotekanProcess*) new testDataCheck<float>(config, *beamform_output_buffer[i], *beamform_output_simulate_buffer[i]));
    //add_process((KotekanProcess*) new rawFileWrite(config, *beamform_output_buffer[i], "/data/test_data/", "beamform_gpu_output_99", "dat"));

    // GPU Test
    //add_process((KotekanProcess*) new rawFileRead(config, *network_input_buffer[i], true, false,
    //                                                "/data/test_data/", "gpu_input_frame_const", "dat"));
//    add_process((KotekanProcess*) new testDataGen(config, *network_input_buffer[i]));
//    add_process((KotekanProcess*) new gpuHSAThread(config, host_buffers[i], i));
//    add_process((KotekanProcess*) new rawFileRead(config, *simulate_output_buffer[i], false, false,
//                                                    "/data/test_data/", "gpu_sim_output_frame_const", "dat"));
//    add_process((KotekanProcess*) new pyPlotResult(config, *gpu_output_buffer[i], i, "/data/test_data/", "beamform_gpu_output_99", "dat") );
//        add_process((KotekanProcess*) new pyPlotOutputError(config, *gpu_output_buffer[i], *simulate_output_buffer[i] ) );
//    add_process((KotekanProcess*) new testDataCheck<int>(config, *gpu_output_buffer[i], *simulate_output_buffer[i] ) );

    // Processes to generate test data
    //add_process((KotekanProcess*) new testDataGen(config, *network_input_buffer[i]));
    //add_process((KotekanProcess*) new gpuSimulate(config, *network_input_buffer[i], *simulate_output_buffer[i]));
    //add_process((KotekanProcess*) new rawFileWrite(config, *network_input_buffer[i], "/data/test_data/", "gpu_input_frame_const", "dat"));
    //add_process((KotekanProcess*) new rawFileWrite(config, *simulate_output_buffer[i], "/data/test_data/", "gpu_sim_output_frame_const", "dat"));
}
