#include "GnssChanAlignMerge.hpp"

#include "GnssChanMetadata.hpp"  // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp"    // for FATAL_ERROR, WARN, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "visUtil.hpp"           // for frameID

#include <algorithm> // for min, max
#include <chrono>    // for the warn rate limit
#include <cmath>     // for floor
#include <complex>   // for complex
#include <cstring>   // for memcpy
#include <ctime>     // for clock_gettime, timespec
#include <numeric>   // for accumulate
#include <string>    // for string (the rate-limited warn helper)

using kotekan::bufferContainer;
using kotekan::Config;
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
    _input_timeout_s = config.get_default<double>(unique_name, "input_timeout_s", 5.0);
    if (_input_timeout_s <= 0.0) {
        FATAL_ERROR("GnssChanAlignMerge[{:s}]: input_timeout_s {:g} must be > 0", unique_name,
                    _input_timeout_s);
        return;
    }

    if (in_bufs.empty() || _in_chans.size() != in_bufs.size()) {
        FATAL_ERROR("GnssChanAlignMerge: {:d} in_bufs but {:d} in_channels entries", in_bufs.size(),
                    _in_chans.size());
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

    // Current frame and its epoch, per input. seq is the absolute sample index of hop 0, from
    // GnssChanMetadata::sample_seq; every feed derives it from the F-engine's global
    // fpga_seq_num, so equal values across inputs are the same instant.
    std::vector<uint8_t*> frames(n_in, nullptr);
    std::vector<int64_t> seqs(n_in, 0);

    // Absent senders must not block the merge. The aggregator config declares one input per
    // node GPU; a feed whose node is not running never delivers, and an untimed wait would
    // block forever with no error. An input that delivers nothing for _input_timeout_s is
    // absent: its channel slice is zeroed and the merge continues on the rest.
    // Zeros rather than the last frame, because stale samples belong to another epoch and
    // would correlate into false detections, where zeros only cost sensitivity.
    //
    // Absent inputs are retried every cycle, so a restarted node rejoins on its own; `present`
    // describes only the current frame. The retry sweep calls acquire() for each input in
    // turn, so an input already absent is polled for absent_poll_s rather than waited on:
    // a full timeout each would make the output period n_absent x _input_timeout_s and leave
    // the search's detections behind the sky. A rejoining sender has frames queued, so the
    // poll picks one up at once.
    //
    // Present inputs get the full wait. Frames arrive every 41.94 ms, so a short poll of a
    // healthy input would often find nothing ready, and the timeout path would zero it.
    std::vector<uint8_t> present(n_in, 0);
    std::vector<uint8_t> announced(n_in,
                                   0); // declared absent; join/leave logs once per transition
    struct timespec ts_timeout;

    // Controller-reset guard. The alignment below advances every input to the maximum sequence
    // held, which is right for a lagging sender but stalls after an F-engine restart:
    // fpga_seq_num restarts near 0, so a feed still delivering pre-reset frames pins `target`
    // where the post-reset feeds never reach it. The loop cannot tell that from a slow sender,
    // so acquire() warns on both signs of it: one feed going backwards, and a spread across
    // feeds too large to be sender lag.
    std::vector<int64_t> last_seq(n_in, -1);
    const auto now_s = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    double last_warn = 0.0;
    const auto warn_rate_limited = [&](const std::string& msg) {
        const double now = now_s();
        if (now - last_warn < 10.0)
            return;
        last_warn = now;
        WARN("GnssChanAlignMerge[{:s}]: {:s}", unique_name, msg);
    };

    auto acquire = [&](int i) -> bool {
        // An input already declared absent is polled, not waited on (see above).
        const double wait_s = announced[i] ? absent_poll_s : _input_timeout_s;
        clock_gettime(CLOCK_REALTIME, &ts_timeout);
        const double whole = std::floor(wait_s);
        ts_timeout.tv_sec += (time_t)whole;
        ts_timeout.tv_nsec += (long)((wait_s - whole) * 1e9);
        if (ts_timeout.tv_nsec >= 1000000000L) {
            ts_timeout.tv_nsec -= 1000000000L;
            ts_timeout.tv_sec += 1;
        }
        const int rc = in_bufs[i]->wait_for_full_frame_timeout(unique_name, in_ids[i], ts_timeout);
        if (rc < 0)
            return false; // shutdown
        if (rc > 0) {     // timed out: treat as absent for this frame
            present[i] = 0;
            frames[i] = nullptr;
            if (!announced[i]) {
                announced[i] = 1;
                WARN("GnssChanAlignMerge[{:s}]: input {:d} delivered nothing in {:.1f} s -- "
                     "treating it as ABSENT and zeroing its {:d} channels. Its sender is "
                     "down, or this config declares more inputs than there are nodes "
                     "running. The merge continues on the others; it will rejoin "
                     "automatically.",
                     unique_name, i, _input_timeout_s, _in_chans[i]);
            }
            return true; // not a failure: carry on without it
        }
        frames[i] = in_bufs[i]->frames[in_ids[i]];
        if (frames[i] == nullptr)
            return false;
        present[i] = 1;
        if (announced[i]) {
            announced[i] = 0;
            INFO("GnssChanAlignMerge[{:s}]: input {:d} is back, {:d} channels rejoined",
                 unique_name, i, _in_chans[i]);
        }
        if (!metadata_is_gnss_chan(in_bufs[i])) {
            // Without sample_seq there is nothing to align on; index lockstep would silently
            // mix epochs.
            FATAL_ERROR("GnssChanAlignMerge: in_bufs[{:d}] has no GnssChanMetadata; alignment "
                        "is impossible",
                        i);
            return false;
        }
        seqs[i] = (int64_t)get_gnss_chan_metadata(in_bufs[i], in_ids[i])->sample_seq;

        // (a) This feed went backwards. A restarted sender resumes ahead of where it left off;
        // a controller reset drops the counter by orders of magnitude. Any backward step warns.
        if (last_seq[i] >= 0 && seqs[i] < last_seq[i])
            warn_rate_limited(fmt::format(
                "input {:d} sample_seq went BACKWARDS {:d} -> {:d} ({:+.3g} samples). The "
                "F-engine controller has restarted. This stage aligns to the MAXIMUM sequence "
                "held, so feeds still delivering pre-reset frames will pin the target where the "
                "post-reset feeds can never reach it and the merge will stall silently. Restart "
                "the aggregator (and any node still on the old epoch).",
                i, last_seq[i], seqs[i], (double)(seqs[i] - last_seq[i])));
        last_seq[i] = seqs[i];

        // (b) The feeds disagree by more than lag can explain. A slow sender is behind by
        // frames; a straddled reset puts feeds about an F-engine uptime apart. 2^40 samples is
        // ~5.7 min at 3.2 GS/s: far beyond any buffering, far below the 2.8e14 of one day's
        // uptime. Only present inputs count: an absent input's seq is stale, or still 0.
        int64_t lo = INT64_MAX, hi = INT64_MIN;
        int n_present = 0;
        for (int k = 0; k < n_in; ++k)
            if (present[k]) {
                lo = std::min(lo, seqs[k]);
                hi = std::max(hi, seqs[k]);
                ++n_present;
            }
        if (n_present >= 2 && hi - lo > (int64_t)1 << 40)
            warn_rate_limited(fmt::format(
                "inputs span {:.3g} samples (min {:d}, max {:d}) -- too far apart to be sender "
                "lag. Some feeds are pre-reset and some post-reset; the merge cannot align them "
                "and is producing nothing. Restart the aggregator.",
                (double)(hi - lo), lo, hi));
        return true;
    };

    for (int i = 0; i < n_in; ++i)
        if (!acquire(i))
            return;

    bool logged_first = false;
    // Every 60 s, log the output rate with the absent inputs, the usual cause of a low rate.
    long n_out = 0;
    double t_rate0 = now_s();
    while (!stop_thread) {
        // Advance every input to the maximum epoch currently held. Each release/reacquire can
        // raise the target (a lagging input leapfrogs if its sender dropped frames), so loop
        // until a full sweep finds all present inputs equal. Equality needs every feed's frames
        // to start on the same sample_seq grid.
        bool aligned = false;
        while (!aligned && !stop_thread) {
            int64_t target = INT64_MIN;
            for (int i = 0; i < n_in; ++i)
                if (present[i])
                    target = std::max(target, seqs[i]);
            if (target == INT64_MIN)
                break; // nothing present: the frame below goes out all zeros, sample_seq 0
            aligned = true;
            for (int i = 0; i < n_in; ++i) {
                while (present[i] && seqs[i] < target) {
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
                if (present[i] && frames[i] != nullptr) {
                    const cf* src = (const cf*)frames[i] + (size_t)m * _in_chans[i];
                    std::memcpy(dst, src, (size_t)_in_chans[i] * sizeof(cf));
                } else {
                    // Zero, never stale (see the absent-sender note above). std::fill rather
                    // than memset: std::complex<float> is not a trivial type.
                    std::fill(dst, dst + _in_chans[i], cf(0.0f, 0.0f));
                }
                dst += _in_chans[i];
            }
        }

        out_buf->allocate_new_metadata_object(out_id);
        int64_t out_seq = 0;
        for (int i = 0; i < n_in; ++i)
            if (present[i]) {
                out_seq = seqs[i];
                break;
            }
        get_gnss_chan_metadata(out_buf, out_id)->sample_seq = (uint64_t)out_seq;
        if (!logged_first) {
            INFO("GnssChanAlignMerge[{:s}]: first aligned frame at sample_seq {:d} across {:d} "
                 "inputs ({:d} channels merged)",
                 unique_name, seqs[0], n_in, _out_chan);
            logged_first = true;
        }

        out_buf->mark_frame_full(unique_name, out_id++);

        ++n_out;
        if (now_s() - t_rate0 >= 60.0) {
            const double el = now_s() - t_rate0;
            std::string absent;
            int n_absent = 0;
            for (int i = 0; i < n_in; ++i)
                if (announced[i]) {
                    absent += (absent.empty() ? "" : ",") + std::to_string(i);
                    ++n_absent;
                }
            const double hz = (double)n_out / el;
            // The nominal rate is one output per input frame (~24/s); far below it, the data
            // reaching the search is old as well as sparse.
            INFO("GnssChanAlignMerge[{:s}]: {:.2f} frames/s ({:d} in {:.0f}s), {:d}/{:d} inputs "
                 "absent [{:s}]{:s}",
                 unique_name, hz, n_out, el, n_absent, n_in, absent,
                 (n_absent && hz < 1.0)
                     ? " -- OUTPUT IS GATED BY ABSENT FEEDS; downstream sky is stale"
                     : "");
            n_out = 0;
            t_rate0 = now_s();
        }
        for (int i = 0; i < n_in; ++i) {
            if (present[i])
                in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);
            // Absent inputs are retried too -- that is how a restarted node rejoins.
            if (!acquire(i))
                return;
        }
    }
}
