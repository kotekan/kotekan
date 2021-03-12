#ifndef BASEBAND_METADATA_HPP
#define BASEBAND_METADATA_HPP

#include "Telescope.hpp"
#include "buffer.h"
#include "chimeMetadata.hpp"
#include "metadata.h"

struct BasebandMetadata {
    uint64_t event_id;
    uint64_t freq_id;
    int64_t start;
    int64_t end;
    int64_t fpga_seq;
    int32_t num_elements;
    int32_t reserved;
    int64_t valid_to;
};

#endif // BASEBAND_METADATA_HPP
