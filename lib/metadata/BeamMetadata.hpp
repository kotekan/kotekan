#ifndef BEAMMETADATA_HPP
#define BEAMMETADATA_HPP

#include "Telescope.hpp"
#include "buffer.h"
#include "chimeMetadata.hpp"
#include "dataset.hpp" // for dset_id_t
#include "metadata.h"

struct BeamMetadata {
    /// The ICEBoard sequence number
    int64_t fpga_seq_start;
    /// The GPS time of @c fpga_seq_start.
    timespec ctime;
    /// Stream identifier
    stream_t stream_id;
    /// ID of the dataset
    dset_id_t dataset_id;
    /// Beam number (e.g. which of the tracking beams is in this stream)
    uint32_t beam_number;
    /// Right ascension of the beam
    float ra;
    /// Declination of the beam
    float dec;
    /// Scaling factor applied to the beam ( typically: raw_beam/(scaling + .5) )
    uint32_t scaling;
};

struct FreqIDBeamMetadata : BeamMetadata {
    uint32_t frequency_bin;
};

struct MergedBeamMetadata {
    /// Number of the raw frame in one merged frame
    uint32_t sub_frame_pre_frame;
    /// The size of the sub frame metadata
    uint32_t sub_frame_metadata_size;
    /// The size of the sub frame
    uint32_t sub_frame_data_size;
};

#endif // BEAMMETADATA_HPP
