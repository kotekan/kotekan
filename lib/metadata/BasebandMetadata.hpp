#ifndef BASEBAND_METADATA_HPP
#define BASEBAND_METADATA_HPP

#include "Telescope.hpp"
#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "metadata.h"

struct BasebandMetadata {
    /// event and frequency ID
    uint64_t event_id;
    uint64_t freq_id;

    //@{
    /// Event start and end fpga seq at this frequency
    uint64_t event_start_fpga;
    uint64_t event_end_fpga;
    //@}

    //@{
    /// Timestamp of the first captured sample
    uint64_t time0_fpga;
    double time0_ctime;
    double time0_ctime_offset;
    //@}

    /// Time of arrival of the packet containing the first sample
    double first_packet_recv_time;

    /// FPGA seq of the first sample in this frame
    int64_t frame_fpga_seq;

    /// Number of valid samples in this frame
    int64_t valid_to;

    /// The time of FPGA frame=0
    uint64_t fpga0_ns;

    /// Number of inputs per sample
    int32_t num_elements;

    /// Future expansion+padding
    int32_t reserved;
};


inline void baseband_metadata_init(BasebandMetadata* c) {
    bzero(c, sizeof(BasebandMetadata));
}

inline void baseband_metadata_copy(BasebandMetadata* out, const BasebandMetadata* in) {
    memcpy(out, in, sizeof(BasebandMetadata));
}

inline bool metadata_is_baseband(Buffer* buf, int) {
    return strcmp(buf->metadata_pool->type_name, "BasebandMetadata") == 0;
}

inline bool metadata_container_is_baseband(const metadataContainer* mc) {
    return strcmp(mc->parent_pool->type_name, "BasebandMetadata") == 0;
}

inline const BasebandMetadata* get_baseband_metadata(const Buffer* buf, int frame_id) {
    return (const BasebandMetadata*)buf->metadata[frame_id]->metadata;
}

inline BasebandMetadata* get_baseband_metadata(Buffer* buf, int frame_id) {
    return (BasebandMetadata*)buf->metadata[frame_id]->metadata;
}

inline const BasebandMetadata* get_baseband_metadata(const metadataContainer* mc) {
    if (!mc)
        return nullptr;
    if (strcmp(mc->parent_pool->type_name, "BasebandMetadata")) {
        WARN_NON_OO("Expected metadata to be type \"BasebandMetadata\", got \"{:s}\".",
                    mc->parent_pool->type_name);
        return nullptr;
    }
    return static_cast<const BasebandMetadata*>(mc->metadata);
}

inline BasebandMetadata* get_baseband_metadata(metadataContainer* mc) {
    if (!mc)
        return nullptr;
    if (strcmp(mc->parent_pool->type_name, "BasebandMetadata")) {
        WARN_NON_OO("Expected metadata to be type \"BasebandMetadata\", got \"{:s}\".",
                    mc->parent_pool->type_name);
        return nullptr;
    }
    return static_cast<BasebandMetadata*>(mc->metadata);
}

#endif // BASEBAND_METADATA_HPP
