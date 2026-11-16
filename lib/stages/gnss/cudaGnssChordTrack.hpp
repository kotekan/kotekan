/**
 * @file
 * @brief The state cudaGnssInject's instances share.
 *  - cudaGnssChordTrackState : public cudaCommandState
 */

#ifndef CUDA_GNSS_CHORD_TRACK_HPP
#define CUDA_GNSS_CHORD_TRACK_HPP

#include "Config.hpp"              // for Config
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommandState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "gnssTrackChain.hpp"      // for TrackChain

#include <string> // for string

/**
 * @class cudaGnssChordTrackState
 * @brief The gnss::TrackChain a cudaGnssInject's instances share.
 *
 * @conf The gnss::TrackChain keys, at the command's path.
 */
class cudaGnssChordTrackState : public cudaCommandState {
public:
    cudaGnssChordTrackState(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);

    gnss::TrackChain chain;
};

#endif // CUDA_GNSS_CHORD_TRACK_HPP
