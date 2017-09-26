#include "buffer.h"
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
#include "chimeMetadata.h"
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


    bufferContainer buffer_container;

    // Create buffers.
    struct Buffer * network_input_buffer[num_fpga_links];
    for (int i = 0; i < num_fpga_links; ++i) {
        network_input_buffer[i] = (struct Buffer *)malloc(sizeof(struct Buffer));
        add_buffer(network_input_buffer[i]);

    }

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct metadataPool * pool[num_fpga_links];
    for (int i = 0; i < num_fpga_links; ++i) {
        pool[i] = create_metadata_pool(2 * buffer_depth, sizeof(struct chimeMetadata));
        add_metadata_pool(pool[i]);
    }

    INFO("Creating input buffers...");
    char buffer_name[100];

    for (int i = 0; i < num_fpga_links; ++i) {
        snprintf(buffer_name, 100, "network_input_buffer_%d", i);
        create_buffer(network_input_buffer[i],
                      buffer_depth,
                      fpga_packet_lenght * samples_per_data_set,
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
