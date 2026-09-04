#include "GnssSubbandSplit.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp" // for FATAL_ERROR
#include "visUtil.hpp"        // for frameID

#include <complex> // for complex
#include <cstring> // for memcpy

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssSubbandSplit);

GnssSubbandSplit::GnssSubbandSplit(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssSubbandSplit::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_bufs = get_buffer_array("out_bufs");
    for (Buffer* b : out_bufs)
        b->register_producer(unique_name);

    _N = config.get<int>(unique_name, "spectrum_length");
    _lo = config.get<std::vector<int>>(unique_name, "subband_lo");
    _hi = config.get<std::vector<int>>(unique_name, "subband_hi");
    if (_lo.size() != out_bufs.size() || _hi.size() != out_bufs.size()) {
        FATAL_ERROR("GnssSubbandSplit: subband_lo/hi ({:d}/{:d}) must match out_bufs ({:d})",
                    _lo.size(), _hi.size(), out_bufs.size());
        return;
    }
    for (size_t m = 0; m < _lo.size(); ++m)
        if (_lo[m] < 0 || _hi[m] > _N || _lo[m] >= _hi[m]) {
            FATAL_ERROR("GnssSubbandSplit: subband {:d} range [{:d},{:d}) invalid for N={:d}", m,
                        _lo[m], _hi[m], _N);
            return;
        }
}

void GnssSubbandSplit::main_thread() {
    frameID in_id(in_buf);
    std::vector<frameID> out_ids;
    for (Buffer* b : out_bufs)
        out_ids.emplace_back(b);

    while (!stop_thread) {
        auto* in = (std::complex<float>*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;
        const int hops = in_buf->frame_size / (int)sizeof(std::complex<float>) / _N;

        bool stopping = false;
        for (size_t m = 0; m < out_bufs.size(); ++m) {
            auto* out =
                (std::complex<float>*)out_bufs[m]->wait_for_empty_frame(unique_name, out_ids[m]);
            if (out == nullptr) {
                stopping = true;
                break;
            }
            const int w = _hi[m] - _lo[m]; // channels in this subband
            for (int h = 0; h < hops; ++h)
                memcpy(out + (size_t)h * w, in + (size_t)h * _N + _lo[m],
                       sizeof(std::complex<float>) * (size_t)w);
            // Propagate the absolute sample reference (sample_seq) into each subband, so
            // each downstream search/tracker shares one "sample 0" (the CHORD comb may
            // scatter these subbands across nodes).
            if (in_buf->metadata_pool && out_bufs[m]->metadata_pool) {
                out_bufs[m]->allocate_new_metadata_object(out_ids[m]);
                in_buf->copy_metadata(in_id, out_bufs[m], out_ids[m]);
            }
            out_bufs[m]->mark_frame_full(unique_name, out_ids[m]++);
        }
        if (stopping)
            break;
        in_buf->mark_frame_empty(unique_name, in_id++);
    }
}
