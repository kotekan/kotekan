#ifndef CHIME_METADATA
#define CHIME_METADATA

#include "metadata.h"
#include "buffer.h"
#include "fpga_header_functions.h"
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack()

struct psrCoord {
  float ra[10];
  float dec[10];
  uint32_t scaling[10];
};

struct chimeMetadata {
    /// The ICEBoard sequence number
    int64_t fpga_seq_num;
    /// The system time when the first packet in the frame was captured
    struct timeval first_packet_recv_time;
    /// The GPS time of @c fpga_seq_num.
    struct timespec gps_time;
    /// The total lost time samples for lost/currupt packets and RFI zeroing.
    /// This value only include RFI losses if RFI zeroing was used.
    int32_t lost_timesamples;
    /// The number of 2.56us samples flaged as containg RFI.
    /// NOTE: This value might contain overlap with lost samples, so it can count
    /// missing samples as samples with RFI.  For renormalization this value
    /// should NOT be used, use @c lost_timesamples instead.
    /// This value will be filled even if RFI zeroing is disabled.
    int32_t rfi_flagged_samples;
    /// The stream ID from the ICEBoard
    /// Note in the case of CHIME-2048 the normally unused section
    /// Encodes the port-shuffle freqeuncy information
    uint16_t stream_ID;
    /// The corrdinates of the pulsar beam (if applicable)
    struct psrCoord psr_coord;
    /// This value is set to 1 if the RFI containing samples were zeroed
    /// in the correlation, and 0 otherwise.
    uint32_t rfi_zeroed;
};

// Helper functions to save lots of pointer work

inline int64_t get_fpga_seq_num(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->fpga_seq_num;
}

inline struct psrCoord get_psr_coord(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->psr_coord;
}

inline uint32_t  get_rfi_zeroed(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->rfi_zeroed;
}

inline int32_t get_lost_timesamples(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->lost_timesamples;
}

/**
 * @brief Get the number of RFI flagged samples
 *
 * @param buf The buffer containing the frame
 * @param ID The frame to get metadata from
 * @return The number of RFI flagged samples
 */
inline int32_t get_rfi_flaged_samples(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->rfi_flagged_samples;
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

inline struct timespec get_gps_time(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    return chime_metadata->gps_time;
}

inline void atomic_add_lost_timesamples(struct Buffer * buf, int ID,
                                        int64_t num_lost_samples) {
    struct metadataContainer * mc = buf->metadata[ID];
    lock_metadata(mc);
    struct chimeMetadata * chime_metadata = (struct chimeMetadata *) mc->metadata;
    chime_metadata->lost_timesamples += num_lost_samples;
    unlock_metadata(mc);
}

/**
 * @brief Add RFI flagged samples to the metadata
 *
 * @param buf The buffer with the metadata
 * @param ID The frame in the buffer to add metadata too
 * @param num_flagged_samples The number of flagged samples to add
 */
inline void atomic_add_rfi_flagged_samples(struct Buffer * buf, int ID,
                                           int64_t num_flagged_samples) {
    struct metadataContainer * mc = buf->metadata[ID];
    lock_metadata(mc);
    struct chimeMetadata * chime_metadata = (struct chimeMetadata *) mc->metadata;
    chime_metadata->rfi_flagged_samples += num_flagged_samples;
    unlock_metadata(mc);
}

// Setting functions

inline void set_fpga_seq_num(struct Buffer * buf, int ID, int64_t seq) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->fpga_seq_num = seq;
}

inline void set_psr_coord(struct Buffer * buf, int ID, struct psrCoord psr_coord) {
    struct chimeMetadata * chime_metadata =
      (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->psr_coord = psr_coord;
  }

inline void set_rfi_zeroed(struct Buffer * buf, int ID, uint32_t rfi_zeroed) {
    struct chimeMetadata * chime_metadata =
      (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->rfi_zeroed = rfi_zeroed;
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

inline void set_gps_time(struct Buffer * buf, int ID, struct timespec time) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->gps_time = time;
}

/**
 * @brief Zeros the number of lost samples for the given frame metadata
 *
 * @param buf The buffer with the frame to metadata to zero
 * @param ID The frame ID
 */
inline void zero_lost_samples(struct Buffer * buf, int ID) {
    struct chimeMetadata * chime_metadata =
     (struct chimeMetadata *) buf->metadata[ID]->metadata;
    chime_metadata->lost_timesamples = 0;
}

#ifdef __cplusplus
}
#endif

#endif
