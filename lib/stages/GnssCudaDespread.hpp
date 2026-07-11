#ifndef GNSS_CUDA_DESPREAD_HPP
#define GNSS_CUDA_DESPREAD_HPP

#include "gnssChannelizedDespread.hpp" // for DespreadResult
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank

#include <array>
#include <complex>
#include <memory>
#include <vector>

/**
 * Host-side driver for the fused GPU GNSS despread (lib/cuda/cudaGnssDespreadKernel.cu).
 *
 * Owns the device buffers and the per-PRN Doppler-bucketed Phi filter caches; the tracker calls
 * @c upload_window once per record, then @c despread3 per active PRN for the Early/Prompt/Late
 * triple. Replica semantics == ChannelizedReplicaBank::hoprate_stream (the validated closed form);
 * Phi tables cover ALL subband channels (a job's covering subset is a bitmask), rebuilt on the CPU
 * only when a sat's Doppler moves > refresh_hz. G1b of docs/gnss_gpu_migration.md: correctness
 * first (synchronous launches, one PRN per call); cross-PRN batching + cudaProcess stream overlap
 * is the G1c throughput step.
 *
 * Only built when USE_CUDA=ON (stages CMake guards the source + defines GNSS_CUDA for the tracker).
 */
class GnssCudaDespread {
public:
    /// @param bank the tracker's replica bank (code tables, filter builder, geometry)
    /// @param n_prn number of PRN slots  @param n_chan channels in this subband (<=64)
    /// @param refresh_hz rebuild a PRN's Phi bucket when its Doppler moves further than this
    GnssCudaDespread(gnss::ChannelizedReplicaBank& bank, int n_prn, int n_chan, int chan_offset,
                     int n_hops, double sample_rate, double f_offset, double refresh_hz = 100.0);
    ~GnssCudaDespread();

    /// Upload one record window ([hop][chan] interleaved complex float, as the tracker holds it).
    void upload_window(const std::complex<float>* window, long long window_start_sample);

    /// Despread the E/P/L triple for PRN slot @c p at the commanded code phase / carrier against
    /// the uploaded window. @c covering = global channel ids minus chan_offset (this subband's
    /// local indices). Results ordered {early, prompt, late}, channel-summed, matching
    /// gnss::channelized_despread's fields.
    std::array<gnss::DespreadResult, 3> despread3(int p, double cp_seed, double spacing_chips,
                                                  double doppler_hz,
                                                  const std::vector<int>& covering);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

#endif
