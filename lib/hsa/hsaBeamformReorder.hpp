/**
 * @file
 * @brief Reordering input for FRB beamform
 *  - hsaBeamformReorder : public hsaCommand
 */

#ifndef HSA_BEAMFORM_REORDER_H
#define HSA_BEAMFORM_REORDER_H

#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string
#include <vector>   // for vector
/**
 * @class hsaBeamformReorder
 * @brief hsaCommand to reorder input for FRB beamform
 *
 * This is an hsaCommand that launches the kernel (reorder.hsaco) for
 * reordering input data from correlator order to cylinder
 * order with polarization index last (A0-B0-C0-D0-A1-B1-C1-D1)
 * which is convenient for FFT beamforming for the
 * downstream FRB pipeline.
 * An array of reordering index of length 512 is parsed from
 * the config file. This is sufficient to reorder the 2048
 * input elements because the inputs are `scrambled' in groups of 4,
 * i.e., every 4 consecutive inputs are still consecutive in the
 * scrambled order. See doclib #546 for details.
 *
 * @requires_kernel    reorder.hsaco
 *
 * @par GPU Memory
 * @gpu_mem input            Input data of size input_frame_len
 * @gpu_mem_type             staging
 * @gpu_mem_format           Array of @c uchar
 * @gpu_mem_metadata         chimeMetadata
 *
 * @gpu_mem reorder_map      Array of reordering index of size 512
 * @gpu_mem_type             static
 * @gpu_mem_format           Array of @c uint8
 * @gpu_mem_metadata         none
 *
 * @gpu_mem input_reordered  Reordered data
 * @gpu_mem_type             static
 * @gpu_mem_format           Array of @c uchar
 * @gpu_mem_metadata         chimeMetadata
 *
 * @conf  num_elements          Int (default 2048). Number of elements
 * @conf  samples_per_data_set  Int (default 49152). Number of time samples in a data set
 * @conf  reorder_map           Int (array of default size 512). Reordering index
 * @conf  num_local_freq        Int (default 1). Number of local frequencies.
 *
 * @todo  Change output name to something more sensible, silly to have "input" as output.
 *
 * @author Cherry Ng
 *
 */

class hsaBeamformReorder : public hsaCommand {
public:
    /// Constructor, also initializes internal variables from config and initializes the array of
    /// reordering index.
    hsaBeamformReorder(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor, cleans up local allocs.
    virtual ~hsaBeamformReorder();

    /// Allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Input length, should be nsamp x n_elem
    int32_t input_frame_len;
    /// Output length, same as input
    int32_t output_frame_len;
    /// Length of reordering index, should be 512
    int32_t map_len;

    /// Number of element, should be 2048
    int32_t _num_elements;
    /// Number of local freq, should be 1
    int32_t _num_local_freq;
    /// Number of sample per data set, current set at 128*128*3
    int32_t _samples_per_data_set;
    /// Array of reordering index
    std::vector<int32_t> _reorder_map;
    /// Array of reordering index in C style for backwards compatibility.
    int* _reorder_map_c;
};

#endif
