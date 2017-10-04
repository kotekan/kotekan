#include "gpuReplayMode.hpp"
#ifdef WITH_HSA
#include "hsaBase.h"
#endif
#include "processFactory.hpp"
#include "chimeMetadata.h"

#include <vector>
#include <string>

using std::string;
using std::vector;

gpuReplayMode::gpuReplayMode(Config& config) : kotekanMode(config) {
}

gpuReplayMode::~gpuReplayMode() {
}

void gpuReplayMode::initalize_processes() {

#ifdef WITH_HSA
    // Start HSA
    kotekan_hsa_start();
#endif

    // Config values:
    int32_t num_local_freq = config.get_int("/", "num_local_freq");
    int32_t num_elements = config.get_int("/", "num_elements");
    int32_t block_size = config.get_int("/", "block_size");
    int32_t num_blocks = (int32_t)(num_elements / block_size) *
                         (num_elements / block_size + 1) / 2.;
    int32_t num_data_sets = config.get_int("/", "num_data_sets");
    int32_t buffer_depth = config.get_int("/", "buffer_depth");

    int output_len = num_local_freq * num_blocks * (block_size*block_size)*2.;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct metadataPool *pool = create_metadata_pool(5 * buffer_depth, sizeof(struct chimeMetadata));
    add_metadata_pool(pool);

    // Input (packet) buffers.
    struct Buffer * gpu_data_buffer = create_buffer(
                                            buffer_depth,
                                            output_len * num_data_sets * sizeof(int32_t),
                                            pool,
                                            "gpu_data_buffer");
    add_buffer(gpu_data_buffer);

    processFactory process_factory(config, buffer_container);
    vector<KotekanProcess *> processes = process_factory.build_processes();

    for (auto process: processes) {
        add_process(process);
    }
}
