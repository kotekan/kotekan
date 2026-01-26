#ifndef BEAMMETADATA_HPP
#define BEAMMETADATA_HPP

#include "Telescope.hpp"
#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "dataset.hpp" // for dset_id_t
#include "metadata.h"

struct BeamMetadata {
    /// event and frequency ID
    uint64_t event_id;
    uint64_t freq_id;

    /// The ICEBoard sequence number
    int64_t fpga_seq_start;
    /// The GPS time of @c fpga_seq_start.
    timespec ctime;
    
    double time0_ctime;
    double time0_ctime_offset;

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

inline bool metadata_is_beam(Buffer* buf, int){
    return strcmp(buf->metadata_pool->type_name, "BeamMetadata") == 0;
}


#endif // BEAMMETADATA_HPP
