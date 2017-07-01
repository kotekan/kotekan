#include "buffers.h"
#include "chrxUplink.hpp"
#include "gpuPostProcess.hpp"
#include "networkOutputSim.hpp"
#include "nullProcess.hpp"
#include "vdifStream.hpp"
#include "network_dpdk.h"
#include "util.h"
#include "packetCapMode.hpp"
#include "fullPacketDump.hpp"
#include "dpdkWrapper.hpp"
#include "processFactory.hpp"
#ifdef WITH_HSA
#include "hsaBase.h"
#endif

#include <vector>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


using std::string;
using std::vector;

packetCapMode::packetCapMode(Config& config) : kotekanMode(config) {

}

packetCapMode::~packetCapMode() {

}

void packetCapMode::initalize_processes() {

    INFO("starting packet cap");

    #ifdef WITH_HSA
        kotekan_hsa_start();
    #endif

    // Config values:
    int32_t fpga_packet_lenght = config.get_int("/", "udp_packet_size");
    int32_t samples_per_data_set = config.get_int("/", "samples_per_data_set");
    int32_t buffer_depth = config.get_int("/", "buffer_depth");
    int32_t num_fpga_links = config.get_int("/", "num_links");

    // Create dump folder
    // Needs to be moved to process
    /*if (dump_to_disk) {
        char dir_name[100];
        snprintf(dir_name, 100, "%s/%s", file_base.c_str(), data_set.c_str());

        struct stat st = {0};

        if (stat(dir_name, &st) == -1) {
            if (mkdir(dir_name, 0777) == -1) {
                ERROR("Could not create directory %s", dir_name);
                return;
            }
        } else {
            INFO("Note: directory %s already exists", dir_name);
            // return;
        }
    }*/

    bufferContainer buffer_container;

    // Create buffers.
    struct Buffer * network_input_buffer[num_fpga_links];
    for (int i = 0; i < num_fpga_links; ++i) {
        network_input_buffer[i] = (struct Buffer *)malloc(sizeof(struct Buffer));
        add_buffer(network_input_buffer[i]);

    }

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool * pool[num_fpga_links];
    for (int i = 0; i < num_fpga_links; ++i) {
        pool[i] = (struct InfoObjectPool *)malloc(sizeof(struct InfoObjectPool));
        add_info_object_pool(pool[i]);
    }

    INFO("Creating input buffers...");
    char buffer_name[100];

    for (int i = 0; i < num_fpga_links; ++i) {
        create_info_pool(pool[i], 2 * buffer_depth, 0, 0);

        snprintf(buffer_name, 100, "network_input_buffer_%d", i);
        create_buffer(network_input_buffer[i],
                      buffer_depth,
                      fpga_packet_lenght * samples_per_data_set,
                      1,
                      1,
                      pool[i],
                      buffer_name);
        buffer_container.add_buffer(buffer_name, network_input_buffer[i]);
    }

    processFactory process_factory(config, buffer_container);
    vector<KotekanProcess *> processes = process_factory.build_processes();

    for (auto process: processes) {
        add_process(process);
    }

}
