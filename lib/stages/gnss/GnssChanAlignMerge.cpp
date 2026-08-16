#include "GnssChanAlignMerge.hpp"

#include "GnssChanMetadata.hpp"  // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp"    // for FATAL_ERROR, WARN, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "visUtil.hpp"           // for frameID

#include <cmath>   // for floor
#include <cstring> // for memcpy, memset
#include <ctime>   // for clock_gettime, timespec

#include <algorithm> // for min, max
#include <chrono>    // for the warn rate limit
#include <complex> // for complex
#include <cstring> // for memcpy
#include <numeric> // for accumulate
#include <string>  // for string (the rate-limited warn helper)

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
    // ABSENT-SENDER TIMEOUT. 0 disables (block forever, the historical behaviour).
    _input_timeout_s = config.get_default<double>(unique_name, "input_timeout_s", 5.0);
    // RE-PROBE COST for an input ALREADY known absent -- see the wait-budget note in
    // main_thread(). Must be long enough to pick up a frame that is already queued (which is
    // all a rejoining sender needs) and short enough that dead feeds are free.
    _absent_probe_s = config.get_default<double>(unique_name, "absent_probe_s", 0.002);

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

    // ABSENT SENDERS MUST NOT BLOCK THE MERGE. The aggregator config declares one input per
    // node-GPU; run fewer nodes than it was generated for and the missing feeds never deliver,
    // so acquire() blocks in wait_for_full_frame FOREVER. The stage stays "alive", the log
    // stops, no frame is ever emitted, and the search reports nothing -- observed 2026-08-08
    // with a 12-input config and 8 nodes running, and it took forty minutes to find because
    // the symptom (no detections) points at the sky, not at a config arity mismatch.
    //
    // A timed wait turns that into an absent input: its channel slice is ZEROED and the merge
    // continues on the rest. Zeroing rather than reusing the last frame is deliberate -- stale
    // samples are not neutral, they correlate against the wrong epoch and would manufacture
    // detections rather than merely lose sensitivity.
    //
    // Absence is not permanent: a timed-out input is retried every cycle, so a node that
    // restarts rejoins on its own. `present` is only about THIS frame.
    //
    // ⚠️ THE COST OF THAT RETRY IS THE WHOLE STAGE'S THROUGHPUT. The reacquire sweep at the end
    // of the loop calls acquire() for EVERY input, SEQUENTIALLY, and an input that is still
    // absent blocks the FULL _input_timeout_s before timing out again. So the merge's output
    // period is n_absent x _input_timeout_s -- it degrades with the number of DEAD feeds, which
    // is precisely the case it was written to survive.
    //
    // MEASURED 2026-08-16 (task #81): 3 absent inputs (cx43's two GPUs + cx51.1) x 5.0 s = a
    // 15.0 s merge period against a 41.94 ms frame. Predicted 15.0, measured 15.0. Downstream,
    // GnssChannelizedSearch reported "20 frames (0.1/s), blocked 100%, 0 frames discarded while
    // the worker was busy" -- 356x starved and NOT compute-bound -- and its detections ran 48.9 s
    // behind the sky. A stale detection is not merely late: the clock-bias solve is
    // median(det_dop - pred_dop) with pred at NOW, so lag x dop_rate is a pure fabricated bias
    // (~44 Hz here). gps_distributed_broker.py:3457 documents the same failure at 90 s costing
    // +68 Hz and dragging a tracker off a 55-sigma satellite.
    //
    // THE FIX: an input ALREADY KNOWN ABSENT is POLLED (_absent_probe_s), not waited on. A
    // rejoining sender has frames queued, so the poll takes one immediately and rejoin stays as
    // prompt as before -- absence costs microseconds instead of seconds.
    //
    // ⚠️ DO NOT extend this to a total per-frame wait budget that also short-polls PRESENT
    // inputs. Frames arrive every 41.94 ms; a healthy input polled 2 ms after its last frame was
    // consumed will usually NOT be ready, and the timeout path ZEROES it. That would silently
    // zero healthy feeds every frame -- turning a throughput bug into a sensitivity bug that
    // looks like the sky. A present feed must be genuinely waited for. If a SLOW-but-alive feed
    // ever needs bounding, measure that it exists first: it is not what #81 was.
    std::vector<uint8_t> present(n_in, 0);
    std::vector<uint8_t> ever(n_in, 0);      // acquired at least once -- gates the spread check
    std::vector<uint8_t> announced(n_in, 0); // so join/leave logs once per transition, not per frame
    struct timespec ts_timeout;

    // CONTROLLER-RESET GUARD. The alignment below advances every input to the MAXIMUM sequence
    // currently held, which is right for a lagging sender and catastrophic for an F-engine that
    // has restarted: fpga_seq_num goes back to ~0, so a feed still delivering pre-reset frames
    // pins `target` at a value the post-reset feeds can never reach, and the merge waits
    // forever. Nothing in the loop can tell that apart from "one sender is slow".
    //
    // The node ingest already refuses this case loudly ("CRS FPGA seq went backwards ...
    // controller likely reset, kotekan stopping") -- the aggregator had no equivalent, which is
    // how it sat through an F-engine restart on 2026-08-07 looking healthy while producing
    // nothing. Detect BOTH shapes: a single feed jumping backwards, and a spread across feeds
    // too large to be sender lag.
    std::vector<int64_t> last_seq(n_in, -1);
    const auto now_s = [] {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
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
        if (_input_timeout_s > 0.0) {
            // An input already declared ABSENT is polled, not waited on -- otherwise every dead
            // feed costs a full timeout on every output frame and the merge period becomes
            // n_absent x _input_timeout_s. See the wait-budget note above (#81).
            const double wait_s = announced[i] ? _absent_probe_s : _input_timeout_s;
            clock_gettime(CLOCK_REALTIME, &ts_timeout);
            const double whole = std::floor(wait_s);
            ts_timeout.tv_sec += (time_t)whole;
            ts_timeout.tv_nsec += (long)((wait_s - whole) * 1e9);
            if (ts_timeout.tv_nsec >= 1000000000L) {
                ts_timeout.tv_nsec -= 1000000000L;
                ts_timeout.tv_sec += 1;
            }
            const int rc = in_bufs[i]->wait_for_full_frame_timeout(unique_name, in_ids[i],
                                                                  ts_timeout);
            if (rc < 0)
                return false;            // shutdown
            if (rc > 0) {                // timed out: treat as absent for this frame
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
                return true;             // NOT a failure: carry on without it
            }
            frames[i] = in_bufs[i]->frames[in_ids[i]];
        } else {
            frames[i] = in_bufs[i]->wait_for_full_frame(unique_name, in_ids[i]);
        }
        if (frames[i] == nullptr)
            return false;
        present[i] = 1;
        ever[i] = 1;
        if (announced[i]) {
            announced[i] = 0;
            INFO("GnssChanAlignMerge[{:s}]: input {:d} is back, {:d} channels rejoined",
                 unique_name, i, _in_chans[i]);
        }
        if (!metadata_is_gnss_chan(in_bufs[i])) {
            // Without sample_seq there is nothing to align on, and guessing (index lockstep)
            // is the silent-corruption mode this stage exists to prevent. Hard error.
            FATAL_ERROR("GnssChanAlignMerge: in_bufs[{:d}] has no GnssChanMetadata; alignment "
                        "is impossible",
                        i);
            return false;
        }
        seqs[i] = (int64_t)get_gnss_chan_metadata(in_bufs[i], in_ids[i])->sample_seq;

        // (a) THIS feed went backwards. Sender restarts are normal and land near where they
        // left off; a controller reset drops the counter by orders of magnitude. Threshold at
        // one frame's worth of samples so ordinary reordering is not reported.
        if (last_seq[i] >= 0 && seqs[i] < last_seq[i])
            warn_rate_limited(fmt::format(
                "input {:d} sample_seq went BACKWARDS {:d} -> {:d} ({:+.3g} samples). The "
                "F-engine controller has restarted. This stage aligns to the MAXIMUM sequence "
                "held, so feeds still delivering pre-reset frames will pin the target where the "
                "post-reset feeds can never reach it and the merge will stall silently. Restart "
                "the aggregator (and any node still on the old epoch).",
                i, last_seq[i], seqs[i], (double)(seqs[i] - last_seq[i])));
        last_seq[i] = seqs[i];

        // (b) The feeds DISAGREE by more than lag can explain. A slow sender is behind by
        // frames; a straddled reset puts feeds ~an F-engine uptime apart. 2^40 samples is
        // ~1.6 h at 3.2 GS/s -- far beyond any buffering, far below the ~3.7e15 seen when a
        // 13-day uptime resets.
        // ONLY over inputs that have delivered at least once. Evaluating this during the
        // initial sequential acquire compared real sequences against the zeros of inputs not
        // yet read, and reported a 7.1e13 straddle that did not exist (2026-08-08) -- a false
        // alarm that sent a diagnosis looking for a stale sender for half an hour.
        int64_t lo = INT64_MAX, hi = INT64_MIN;
        int n_ever = 0;
        for (int k = 0; k < n_in; ++k)
            if (ever[k] && present[k]) {
                lo = std::min(lo, seqs[k]);
                hi = std::max(hi, seqs[k]);
                ++n_ever;
            }
        if (n_ever >= 2 && hi - lo > (int64_t)1 << 40)
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
    // THROUGHPUT VISIBILITY. #81 stayed hidden because nothing here ever said what rate this
    // stage was achieving. The search's own log DID show it starving ("0.1/s, blocked 100%"),
    // but with no counterpart at the producer there was nothing to connect it to, and the
    // symptom -- no detections, stale detections, a walking clock bias -- points at the sky.
    // The absent set is printed alongside the rate because it is the CAUSE of the rate.
    long n_out = 0;
    double t_rate0 = now_s();
    while (!stop_thread) {
        // Advance every input to the maximum epoch currently held. Each release/reacquire can
        // RAISE the target (the lagging input may leapfrog if the sender dropped frames), so
        // loop until a full sweep finds all inputs equal.
        bool aligned = false;
        while (!aligned && !stop_thread) {
            int64_t target = INT64_MIN;
            for (int i = 0; i < n_in; ++i)
                if (present[i])
                    target = std::max(target, seqs[i]);
            if (target == INT64_MIN)
                break;   // nothing present at all: nothing to align to, retry the outer loop
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
                    // ZERO, never stale. An absent feed's old samples belong to another epoch
                    // and would correlate into a false detection; zeros only cost sensitivity.
                    // std::fill, not memset: cf is std::complex<float>, a non-trivial type.
                    std::fill(dst, dst + _in_chans[i], cf(0.0f, 0.0f));
                }
                dst += _in_chans[i];
            }
        }

        out_buf->allocate_new_metadata_object(out_id);
        int64_t out_seq = 0;
        for (int i = 0; i < n_in; ++i)
            if (present[i]) { out_seq = seqs[i]; break; }
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
            // The nominal rate is one output per input frame; anything far below it means the
            // sky reaching the search is OLD, not merely sparse.
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
