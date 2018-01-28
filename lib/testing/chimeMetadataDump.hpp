#ifndef CHIME_METADATA_DUMP_HPP
#define CHIME_METADATA_DUMP_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include <unistd.h>

/*
 * Displays CHIME metedata for a given buffer
 * "buf": String with the name of the buffer display metadata info for.
 */

class chimeMetadataDump : public KotekanProcess {
public:
    chimeMetadataDump(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~chimeMetadataDump();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;
private:
    struct Buffer *buf;
    int32_t len;
    int32_t offset;
};

#endif