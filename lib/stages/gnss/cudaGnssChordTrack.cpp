#include "cudaGnssChordTrack.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

cudaGnssChordTrackState::cudaGnssChordTrackState(Config& config, const std::string& unique_name,
                                                 bufferContainer& host_buffers,
                                                 cudaDeviceInterface& device) :
    cudaCommandState(config, unique_name, host_buffers, device), chain(config, unique_name) {}
