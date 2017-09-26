#ifndef CHIME_METADATA
#define CHIME_METADATA

#include "metadata.h"
#include "buffer.h"
#include "fpga_header_functions.h"
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

struct chimeMetadata {
    int64_t fpga_seq_num;
    struct timeval first_packet_recv_time;
    int32_t lost_timesamples;
    uint16_t stream_ID;
};

// Helper functions to save lots of pointer work

inline int64_t get_fpga_seq_num(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->fpga_seq_num;
}

inline int32_t get_lost_timesamples(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->lost_timesamples;
}

inline uint16_t get_stream_id(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->stream_ID;
}

inline stream_id_t get_stream_id_t(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return extract_stream_id(chime_metadata->stream_ID);
}

inline struct timeval get_first_packet_recv_time(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->first_packet_recv_time;
}

inline void atomic_add_lost_timesamples(struct Buffer * buf, int ID,
                                        int64_t num_lost_samples) {
    struct metadataContainer * mc = buf->metadata[ID];
    lock_metadata(mc);
    struct chimeMetadata * chime_metadata = (struct chimeMetadata *) mc->metadata;
    chime_metadata->lost_timesamples += num_lost_samples;
    unlock_metadata(mc);
}

// Setting functions

inline void set_fpga_seq_num(struct Buffer * buf, int ID, int64_t seq) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->fpga_seq_num = seq;
}

inline void set_stream_id(struct Buffer * buf, int ID, uint16_t stream_id) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->stream_ID = stream_id;
}

inline void set_stream_id_t(struct Buffer * buf, int ID, stream_id_t stream_id) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->stream_ID = encode_stream_id(stream_id);
}

inline void set_first_packet_recv_time(struct Buffer * buf, int ID, struct timeval time) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->first_packet_recv_time = time;
}

#ifdef __cplusplus
}
#endif

#endif