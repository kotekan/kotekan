// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
* @file   BeamMetadata.hpp
* @brief  This file declares the basic BeamMetadata 
*         structure
*
* @author Mehdi Najafi
* @date   28 AUG 2022
*****************************************************/

#ifndef BEAMMETADATA_HPP
#define BEAMMETADATA_HPP

#include "Telescope.hpp"                  // stream_t
#include "dataset.hpp"                    // for dset_id_t
#include "metadataFactory.hpp"            // metadata registration
#include "FrequencyAssembledMetadata.hpp" // FrequencyAssembledMetadata

struct BeamMetadata {
    /// The ICEBoard sequence number
    int64_t fpga_seq_start;
    /// The GPS time of @c fpga_seq_start
    timespec gps_time;
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

// register this basic metadata structure
REGISTER_KOTEKAN_METADATA(BeamMetadata)

// make the frequency assembled form of the above structure as well
typedef FrequencyAssembledMetadata<BeamMetadata> assembledBeamMetadata;
// register this new metadata structure
REGISTER_KOTEKAN_METADATA(assembledBeamMetadata)

#endif // BEAMMETADATA_HPP
