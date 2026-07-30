#include "GnssChanAlignMerge.hpp"

#include "GnssChanMetadata.hpp"  // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp"    // for FATAL_ERROR, WARN, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "visUtil.hpp"           // for frameID

#include <complex> // for complex
#include <cstring> // for memcpy
#include <numeric> // for accumulate

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using kotekan::prometheus::Metrics;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssChanAlignMerge);

GnssChanAlignMerge::GnssChanAlignMerge(Config& config, const std::string& unique_name,
                                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChanAlignMerge::main_thread, this)) {
    in_bufs = get_buffer_array("in_bufs");
    for (Buffer* b : in_bufs)
        b->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _in_chans = config.get<std::vector<int>>(unique_name, "in_channels");
    _n_hops = config.get<int>(unique_name, "samples_per_data_set");

    if (in_bufs.empty() || _in_chans.size() != in_bufs.size()) {
        FATAL_ERROR("GnssChanAlignMerge: {:d} in_bufs but {:d} in_channels entries",
                    in_bufs.size(), _in_chans.size());
        return;
    }
    _out_chan = std::accumulate(_in_chans.begin(), _in_chans.end(), 0);
    for (size_t i = 0; i < in_bufs.size(); ++i) {
        const size_t need = (size_t)_n_hops * _in_chans[i] * sizeof(cf);
        if ((size_t)in_bufs[i]->frame_size < need)
            FATAL_ERROR("GnssChanAlignMerge: in_bufs[{:d}] frame {:d} B < {:d} ({:d} hops x {:d} "
                        "chan x cfloat32)",
                        i, (size_t)in_bufs[i]->frame_size, need, _n_hops, _in_chans[i]);
    }
    const size_t out_need = (size_t)_n_hops * _out_chan * sizeof(cf);
    if ((size_t)out_buf->frame_size < out_need)
        FATAL_ERROR("GnssChanAlignMerge: out_buf frame {:d} B < {:d} ({:d} hops x {:d} chan)",
                    (size_t)out_buf->frame_size, out_need, _n_hops, _out_chan);
}

void GnssChanAlignMerge::main_thread() {
    const int n_in = (int)in_bufs.size();
    std::vector<frameID> in_ids;
    for (Buffer* b : in_bufs)
        in_ids.emplace_back(b);
    frameID out_id(out_buf);

    auto& skipped = Metrics::instance().add_counter("kotekan_gnss_merge_skipped_frames_total",
                                                    unique_name, {"input"});

    // Current frame + its epoch, per input. seq is the ABSOLUTE sample index of hop 0,
    // taken from GnssChanMetadata::sample_seq (all feeds inherit it from the F-engine's
    // global fpga_seq_num, so equality across inputs means the same sky).
    std::vector<uint8_t*> frames(n_in, nullptr);
    std::vector<int64_t> seqs(n_in, 0);

    auto acquire = [&](int i) -> bool {
        frames[i] = in_bufs[i]->wait_for_full_frame(unique_name, in_ids[i]);
        if (frames[i] == nullptr)
            return false;
        if (!metadata_is_gnss_chan(in_bufs[i])) {
            // Without sample_seq there is nothing to align on, and guessing (index lockstep)
            // is the silent-corruption mode this stage exists to prevent. Hard error.
            FATAL_ERROR("GnssChanAlignMerge: in_bufs[{:d}] has no GnssChanMetadata; alignment "
                        "is impossible",
                        i);
            return false;
        }
        seqs[i] = (int64_t)get_gnss_chan_metadata(in_bufs[i], in_ids[i])->sample_seq;
        return true;
    };

    for (int i = 0; i < n_in; ++i)
        if (!acquire(i))
            return;

    bool logged_first = false;
    while (!stop_thread) {
        // Advance every input to the maximum epoch currently held. Each release/reacquire can
        // RAISE the target (the lagging input may leapfrog if the sender dropped frames), so
        // loop until a full sweep finds all inputs equal.
        bool aligned = false;
        while (!aligned && !stop_thread) {
            int64_t target = seqs[0];
            for (int i = 1; i < n_in; ++i)
                target = std::max(target, seqs[i]);
            aligned = true;
            for (int i = 0; i < n_in; ++i) {
                while (seqs[i] < target) {
                    skipped.labels({std::to_string(i)}).inc();
                    in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);
                    if (!acquire(i))
                        return;
                    aligned = false; // epochs moved; re-sweep with a possibly higher target
                }
            }
        }
        if (stop_thread)
            break;

        cf* out = (cf*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;

        // [hop][sum chan]: concatenate each input's row per hop, input order.
        for (int m = 0; m < _n_hops; ++m) {
            cf* dst = out + (size_t)m * _out_chan;
            for (int i = 0; i < n_in; ++i) {
                const cf* src = (const cf*)frames[i] + (size_t)m * _in_chans[i];
                std::memcpy(dst, src, (size_t)_in_chans[i] * sizeof(cf));
                dst += _in_chans[i];
            }
        }

        out_buf->allocate_new_metadata_object(out_id);
        get_gnss_chan_metadata(out_buf, out_id)->sample_seq = (uint64_t)seqs[0];
        if (!logged_first) {
            INFO("GnssChanAlignMerge[{:s}]: first aligned frame at sample_seq {:d} across {:d} "
                 "inputs ({:d} channels merged)",
                 unique_name, seqs[0], n_in, _out_chan);
            logged_first = true;
        }

        out_buf->mark_frame_full(unique_name, out_id++);
        for (int i = 0; i < n_in; ++i) {
            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);
            if (!acquire(i))
                return;
        }
    }
}
