#include "hexDump.hpp"
#include "util.h"

hexDump::hexDump(Config& config,
                        const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&hexDump::main_thread, this)) {

    buf = get_buffer("buf");
    register_consumer(buf, unique_name.c_str());
    len = config.get_int(unique_name, "len");
    offset = config.get_int(unique_name, "offset");
}

hexDump::~hexDump() {
}

void hexDump::apply_config(uint64_t fpga_seq) {
}

void hexDump::main_thread() {

    int buf_id = 0;

    for (;;) {

        wait_for_full_buffer(buf, unique_name.c_str(), buf_id);
        INFO("hexDump: Got buffer %s[%d]", buf->buffer_name, buf_id);

        hex_dump(16, (void*)&buf->data[buf_id][offset], len );

        mark_buffer_empty(buf, unique_name.c_str(), buf_id);
        buf_id = (buf_id + 1) % buf->num_buffers;
    }
}