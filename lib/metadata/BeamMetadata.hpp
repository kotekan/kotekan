#ifndef BEAMMETADATA_HPP
#define BEAMMETADATA_HPP

#include "Telescope.hpp"
#include "buffer.h"
#include "dataset.hpp" // for dset_id_t
#include "metadata.h"
#include "chimeMetadata.hpp"

struct BeamMetadata {
    /// The ICEBoard sequence number
    int64_t fpga_seq_start;
    /// The GPS time of @c fpga_seq_start.
    timespec ctime;
    /// Stream identifier
    stream_t stream_id;
    /// ID of the dataset
    dset_id_t dataset_id;
    /// Right ascension of the beam
    float ra;
    /// Declination of the beam
    float dec;
    /// Scaling factor applied to the beam ( typically: raw_beam/(scaling + .5) )
    uint32_t scaling;
};

#endif // BEAMMETADATA_HPP
