// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
 * @file   chimeMetadata.hpp
 * @brief  This file declares Chime BeamMetadata
 *         structure
 *
 * Last update by Mehdi Najafi
 * @date   08 SEP 2022
 *****************************************************/

#ifndef CHIME_METADATA
#define CHIME_METADATA

#include "Telescope.hpp"
#include "buffer.h"
#include "datasetManager.hpp"
#include "metadataFactory.hpp" // metadata registration

#include <sys/time.h>

#define MAX_NUM_BEAMS 20

#pragma pack()

struct beamCoord {
    float ra[MAX_NUM_BEAMS];
    float dec[MAX_NUM_BEAMS];
    uint32_t scaling[MAX_NUM_BEAMS];
};


struct chimeMetadata {
    /// The ICEBoard sequence number
    int64_t fpga_seq_num;
    /// The system time when the first packet in the frame was captured
    struct timeval first_packet_recv_time;
    /// The GPS time of @c fpga_seq_num.
    struct timespec gps_time;
    /// The total lost time samples for lost/corrupt packets and RFI zeroing.
    /// This value only include RFI losses if RFI zeroing was used.
    int32_t lost_timesamples;
    /// The number of FPGA frames flagged as containing RFI.
    /// NOTE: This value might contain overlap with lost samples, so it can count
    /// missing samples as samples with RFI.  For renormalization this value
    /// should NOT be used, use @c lost_timesamples instead.
    /// This value will be filled even if RFI zeroing is disabled.
    int32_t rfi_flagged_samples;
    /// This value is set to 1 if the RFI containing samples were zeroed
    /// in the correlation, and 0 otherwise.
    uint32_t rfi_zeroed;
    /// The number of bad inputs in the RFI systems bad input list.
    /// This value is mostly needed for renormalization of the SK values.
    uint32_t rfi_num_bad_inputs;
    /// The stream ID from the ICEBoard
    /// Note in the case of CHIME-2048 the normally unused section
    /// Encodes the port-shuffle frequency information
    uint16_t stream_id;
    /// ID of the dataset
    dset_id_t dataset_id;
    /// The coordinates of the tracking beam (if applicable)
    struct beamCoord beam_coord;
};

// register this metadata structure
REGISTER_KOTEKAN_METADATA(chimeMetadata)

// Helper functions to save lots of pointer work

inline int64_t get_fpga_seq_num(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->fpga_seq_num;
}

inline struct beamCoord get_beam_coord(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->beam_coord;
}

inline uint32_t get_rfi_zeroed(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->rfi_zeroed;
}

inline int32_t get_lost_timesamples(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->lost_timesamples;
}

/**
 * @brief Get the number of RFI flagged samples
 *
 * @param buf The buffer containing the frame
 * @param ID The frame to get metadata from
 * @return The number of RFI flagged samples
 */
inline int32_t get_rfi_flagged_samples(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->rfi_flagged_samples;
}

/**
 * @brief Get the number of bad inputs (elements) in the data
 *
 * @param buf The buffer containing the frame
 * @param ID The frame to get metadata from
 * @return The number of bad inputs in the input mask
 */
inline uint32_t get_rfi_num_bad_inputs(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->rfi_num_bad_inputs;
}

inline stream_t get_stream_id_from_metadata(const chimeMetadata* metadata) {
    return {(uint64_t)metadata->stream_id};
}

inline stream_t get_stream_id(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return get_stream_id_from_metadata(chime_metadata);
}

inline struct timeval get_first_packet_recv_time(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->first_packet_recv_time;
}

inline struct timespec get_gps_time(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->gps_time;
}

inline dset_id_t get_dataset_id(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return chime_metadata->dataset_id;
}

inline void atomic_add_lost_timesamples(struct Buffer* buf, int ID, int64_t num_lost_samples) {
    struct metadataContainer* mc = buf->metadata[ID];
    lock_metadata(mc);
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)mc->metadata;
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
inline void atomic_add_rfi_flagged_samples(struct Buffer* buf, int ID,
                                           int64_t num_flagged_samples) {
    struct metadataContainer* mc = buf->metadata[ID];
    lock_metadata(mc);
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)mc->metadata;
    chime_metadata->rfi_flagged_samples += num_flagged_samples;
    unlock_metadata(mc);
}

// Setting functions

inline void set_rfi_num_bad_inputs(struct Buffer* buf, int ID, uint32_t num_bad_inputs) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->rfi_num_bad_inputs = num_bad_inputs;
}

inline void set_fpga_seq_num(struct Buffer* buf, int ID, int64_t seq) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->fpga_seq_num = seq;
}

inline void set_beam_coord(struct Buffer* buf, int ID, struct beamCoord beam_coord) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->beam_coord = beam_coord;
}

inline void set_rfi_zeroed(struct Buffer* buf, int ID, uint32_t rfi_zeroed) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->rfi_zeroed = rfi_zeroed;
}

inline void set_stream_id_to_metadata(struct chimeMetadata* metadata, stream_t stream_id) {
    metadata->stream_id = (uint16_t)(stream_id.id);
}

inline void set_stream_id(struct Buffer* buf, int ID, stream_t stream_id) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    set_stream_id_to_metadata(chime_metadata, stream_id);
}

inline void set_first_packet_recv_time(struct Buffer* buf, int ID, struct timeval time) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->first_packet_recv_time = time;
}

inline void set_gps_time(struct Buffer* buf, int ID, struct timespec time) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->gps_time = time;
}

inline void set_dataset_id(struct Buffer* buf, int ID, dset_id_t dataset_id) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->dataset_id = dataset_id;
}

/**
 * @brief Zeros the number of lost samples for the given frame metadata
 *
 * @param buf The buffer with the frame to metadata to zero
 * @param ID The frame ID
 */
inline void zero_lost_samples(struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->lost_timesamples = 0;
}

#endif
