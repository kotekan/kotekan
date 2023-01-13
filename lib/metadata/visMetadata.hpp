// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
 * @file   visMetadata.hpp
 * @brief  This file declares the VisMetadata
 *         structure
 *
 * @author Mehdi Najafi
 * @date   29 SEP 2022
 *****************************************************/

#ifndef VIS_METADATA_HPP
#define VIS_METADATA_HPP

#include "Telescope.hpp"       // for freq_id_t
#include "dataset.hpp"         // for dset_id_t
#include "metadataFactory.hpp" // for metadata registration

/**
 * @struct VisMetadata
 * @brief Metadata for the visibility style buffers
 *
 * @author Richard Shaw
 **/
struct VisMetadata {

    /// The FPGA sequence number of the integration frame
    uint64_t fpga_seq_start;
    /// The ctime of the integration frame
    timespec ctime;
    /// Nominal length of the frame in FPGA ticks
    uint64_t fpga_seq_length;
    /// Amount of data that actually went into the frame (in FPGA ticks)
    uint64_t fpga_seq_total;
    /// The number of FPGA frames flagged as containing RFI. NOTE: This value
    /// might contain overlap with lost samples, as that counts missing samples
    /// as well as RFI. For renormalization this value should NOT be used, use
    /// lost samples (= @c fpga_seq_length - @c fpga_seq_total) instead.
    uint64_t rfi_total;

    /// ID of the frequency bin
    freq_id_t freq_id;
    /// ID of the dataset
    dset_id_t dataset_id;

    /// Number of elements for data in buffer
    uint32_t num_elements;
    /// Number of products for data in buffer
    uint32_t num_prod;
    /// Number of eigenvectors and values calculated
    uint32_t num_ev;
};

// register this metadata structure
REGISTER_KOTEKAN_METADATA(VisMetadata)

#endif // VIS_METADATA_HPP
