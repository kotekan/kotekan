#include "singleDishVDIFMode.hpp"

#include "buffers.h"
#include "errors.h"
#include "dpdkWrapper.hpp"
#include "Config.hpp"

#include <vector>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


using std::string;
using std::vector;

singleDishVDIFMode::singleDishVDIFMode(Config& config) : kotekanMode(config) {
}

singleDishVDIFMode::~singleDishVDIFMode() {
}

void singleDishVDIFMode::initalize_processes() {

    INFO("Starting single dish vdif mode");

    // Config values:
    int32_t samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    int32_t num_elements = config.get_int("/processing/num_elements");
    int32_t buffer_depth = config.get_int("/processing/buffer_depth");
    const int32_t vdif_frame_len = 1056;

    // Create buffers.
    struct Buffer * vdif_buf;

    vdif_buf = (struct Buffer *)malloc(sizeof(struct Buffer));
    add_buffer(vdif_buf);

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool * pool;
    pool = (struct InfoObjectPool *)malloc(sizeof(struct InfoObjectPool));
    create_info_pool(pool, 2 * buffer_depth, 0, 0);
    add_info_object_pool(pool);

    INFO("Creating vdif buffer...");

    create_buffer(vdif_buf,
                  buffer_depth,
                  vdif_frame_len * num_elements * samples_per_data_set,
                  1,
                  1,
                  pool,
                  "vdif_buffer");

    add_process((KotekanProcess *) new dpdkWrapper(config, &vdif_buf, "vdif"));

}