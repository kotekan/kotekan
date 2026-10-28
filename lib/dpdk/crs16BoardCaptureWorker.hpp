/**
 * @file
 * @brief A capture worker/handler for CRS board packets
 */

#ifndef CRS_16BOARD_CAPTURE_WORKER_HPP
#define CRS_16BOARD_CAPTURE_WORKER_HPP

#include "Config.hpp"
#include "FramePrefetchService.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "crsUtils.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"
#include "timeUtil.hpp"
#include "util.h"
#include "visUtil.hpp"

#include "json.hpp"

#include <algorithm>
#include <array>
#include <immintrin.h>


/**
 * @brief Capture worker for the 16 CRS (F-engine) board configuration.
 *
 * Consumes the packets a @c crs16BoardDistributor placed on this worker's ring and copies
 * them into the output voltage buffer, one frame per @c time_samples_per_frame FPGA samples.
 * Packet validation (size, checksum, cookie, stream ID range) has already been done by the
 * distributor, so this handler's hot path is just a header parse and a non-temporal copy.
 *
 * Each packet's location in the output frame is computed from its FPGA sequence number (time
 * axis), stream ID (frequency axis), and source ID (element/board axis), so a frame always
 * covers an exact sequence-number range. Frames are supplied by a
 * @c kotekan::FramePrefetchService running on separate cores. Several workers (one per port)
 * write into the same frames, each covering its own subset of stream IDs.
 *
 * The 16 CRS boards feeding a node are identified by a raw source ID
 * (slot_id + 16 * crate_id). By default board data is placed along the output element axis
 * in raw source-ID order; the optional @c crs_board_remap config reorders the boards for
 * downstream consumers. A per-packet receipt bitmap is maintained alongside each data frame
 * with layout [time_long][board output slot][stream ID], one bit per packet; it uses the
 * same (possibly remapped) board axis as the data, so packet-loss accounting (e.g.
 * ProcessPacketMask) follows the output ordering. Log messages always report the raw
 * stream/source IDs as they appear in the packets.
 *
 * @par Capture scheduling
 * By default capture starts at the first frame edge a couple of seconds after packets begin
 * to flow.  Set @c capture_start_time_utc to instead hold off until a given wall clock time,
 * and @c capture_n_frames to stop after a fixed number of frames.  Both options can be set on
 * the @c dpdkCore stage itself so that every worker shares them.
 *
 * @par Buffers
 * @buffer out_buf  Kotekan buffer the packet payloads are placed in.
 *       @buffer_format uint8_t array of 4+4-bit complex voltage data
 *       @buffer_metadata chordMetadata (filled only by the worker with allocate_metadata set)
 * @buffer receipt_bitmap_buf  Kotekan buffer holding the per-packet receipt bitmap.
 *       @buffer_format uint8_t array of packet receipt bits
 *       @buffer_metadata none
 *
 * @conf  port                        Int. The DPDK port this worker accepts packets from.
 * @conf  out_buf                     String. Name of the output data buffer.
 * @conf  receipt_bitmap_buf          String. Name of the packet receipt bitmap buffer.
 * @conf  packet_size                 UInt32. Total packet size in bytes; must equal
 *                                    payload_size plus the 64-byte header.
 * @conf  payload_size                UInt32. Payload bytes per packet. Must be a multiple
 *                                    of 32, and the output frame size must be a multiple
 *                                    of payload_size * num_source_ids * num_stream_ids.
 * @conf  num_expected_stream_ids     UInt32. Number of distinct stream IDs expected on
 *                                    this port; capture starts once all have been seen.
 * @conf  frame_service_cpu_affinity  Array of Int. CPU cores the frame prefetch service
 *                                    thread may run on.
 * @conf  prefetch_depth              Int. Default 4. Number of frames the prefetch
 *                                    service keeps ready ahead of the write position.
 * @conf  allocate_metadata           Bool. Default false. Enable on the worker that
 *                                    should allocate and fill each frame's chordMetadata
 *                                    (exactly one worker per buffer).
 * @conf  capture_start_time_utc      String. Default "" (start as soon as packets flow).
 *                                    UTC time at which capture should start, e.g.
 *                                    "2026-01-31 18:45:00".  The time is converted to an
 *                                    FPGA sequence number using the GPS time of the F-engine
 *                                    (never the system clock), and rounded down to the
 *                                    nearest frame edge.  It must be at least one minute in
 *                                    the future; a start more than 12 hours out is allowed
 *                                    but warned about.
 * @conf  capture_n_frames            UInt64. Default 0 (unlimited).  Stop capturing after
 *                                    this many frames.  The DPDK threads exit but kotekan
 *                                    keeps running, so downstream stages can flush what they
 *                                    have.
 * @conf  crs_board_remap             Array of UInt32. Default empty (identity). Desired
 *                                    board order by output slot: output slot i receives
 *                                    the board whose raw source_id is crs_board_remap[i].
 *                                    Must be a permutation of [0, num_source_ids).
 *                                    Applies to both the data frame element axis and the
 *                                    receipt bitmap.
 *
 * @par Metrics
 * @metric kotekan_dpdk_crs16_out_of_window_packets_total
 *         Packets dropped for a sequence number outside the active frames and the catch-up
 *         window, labelled by port. Non-zero means a packet arrived with a sequence number out of
 *         step with the rest of the stream.
 *
 * @author Andre Renard
 */
class crs16BoardCaptureWorker : public dpdkRXhandler {
public:
    /// Default constructor
    crs16BoardCaptureWorker(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& buffer_container, int worker_id,
                            const std::vector<rte_ring*>& worker_rings);

    /// Processes the incoming packets
    int handle_packet(struct rte_mbuf* mbuf) override;

    /// Update stats, not used by this handler yet.
    virtual void update_stats() override {};

protected:
    /// Rings the distributor uses to hand packets to the workers (unused here).
    const std::vector<rte_ring*>& worker_rings;

    /// The output buffer
    Buffer* out_buf;
    /// The packet recepit bitmap buffer
    Buffer* receipt_bitmap_buf;

    /// Index of this worker within the DPDK stage.
    int worker_id;

    /// The packet size
    uint32_t packet_size;

    /// The packet payload size (packet_size minus header_size)
    uint32_t payload_size;

    /// Number of distinct stream IDs to see on this port before starting capture.
    uint32_t num_expected_stream_ids;

    /// Size of the CRS packet headers in bytes.
    const uint32_t header_size = 64;
    // Each node processes data from 16 source IDs (16 CRS boards) and 8 stream IDs (1/16 of the
    // total streams)
    static constexpr uint32_t num_source_ids = 16;
    const uint32_t num_stream_ids = 8;
    const uint32_t time_samples_per_packet = 16;

    /// Output slot in the frame for each raw CRS source_id: indexed by source_id, so
    /// board `src` writes into slot dest_slot_for_source_id[src]. Identity unless the
    /// optional `crs_board_remap` config is set, in which case this is the inverse of
    /// that list. Note the config list is indexed the other way round, by output slot:
    /// slot i receives the board whose raw source_id is crs_board_remap[i].
    std::array<uint16_t, num_source_ids> dest_slot_for_source_id;

    /// The stream IDs seen on this port during startup; packets with other IDs are dropped.
    std::vector<uint32_t> stream_ids_expected;

    /// True until the expected stream ID list is complete and the prefetch service started.
    bool first_run = true;

    /// Number of FPGA time samples covered by one output frame.
    uint64_t time_samples_per_frame;
    /// Number of packets (across all boards and streams) that make up one output frame.
    uint64_t packets_per_frame;

    /// Prefetches, zeros, and hands out output frames on separate cores.
    std::unique_ptr<kotekan::FramePrefetchService> prefetch_service;

    /// The frame currently being filled, and the next one (for out-of-order margin).
    const kotekan::FrameInfo* active_f0 = nullptr;
    const kotekan::FrameInfo* active_f1 = nullptr;

    /// Requested capture start time in ns since the UNIX epoch.
    /// Zero means "start as soon as packets are flowing".
    int64_t capture_start_time_ns = 0;

    /// Number of frames to capture before stopping, 0 = unlimited.
    uint64_t capture_n_frames;

    /// Number of frames handed to the output buffer so far.
    uint64_t frames_captured = 0;

    /// FPGA seq number of the first sample of the first captured frame.
    uint64_t capture_start_seq = 0;

    /// Set once @c capture_start_seq has been worked out from the first packets.
    bool capture_start_seq_known = false;

    /// Instrument (GPS) time at FPGA seq 0, and the length of one seq tick, both in
    /// nanoseconds. Cached from the Telescope when the start seq is computed.
    int64_t time0_ns = 0;
    int64_t seq_length_ns = 0;

    /// How far ahead of @c capture_start_seq the prefetch service is armed.
    uint64_t prefetch_lead_samples = 0;

    /// The seq number at which the next "waiting for the scheduled start" message is logged.
    uint64_t next_countdown_seq = 0;

    /// The prefetch service is armed this long before the capture start, so that the first
    /// frames are ready by the time the first packets belonging to them arrive.
    static constexpr double _prefetch_lead_s = 2.0;

    /// A scheduled start closer than this to the current F-engine time is an error, ...
    static constexpr int64_t _min_start_lead_ns = 60L * 1'000'000'000L;

    /// ... and one further out than this is allowed but warned about.
    static constexpr int64_t _max_start_lead_ns = 12L * 3600L * 1'000'000'000L;

    /// Interval between "waiting for the scheduled start" messages.
    static constexpr int64_t _countdown_interval_ns = 60L * 1'000'000'000L;

    /**
     * @brief Work out, and sanity check, the seq number at which capture starts.
     *
     * Called once, with the seq number of a packet that is arriving now, so that the
     * configured UTC start time can be compared against the F-engine's GPS time.
     *
     * @param current_seq The FPGA seq number of the packet being handled.
     * @return False if the configured start time is unusable, in which case kotekan has
     *         already been asked to shut down and the caller should stop the lcore.
     */
    bool compute_capture_start_seq(uint64_t current_seq);

    /// Instrument (GPS) time in ns of an FPGA seq number, using the cached Telescope values.
    int64_t seq_to_time_ns(uint64_t seq) const {
        return time0_ns + (int64_t)seq * seq_length_ns;
    }

    /// Sample-based FPGA seq monotonicity check. At each sample the seq must
    /// be strictly greater than the previous sample's, otherwise we treat it
    /// as an FPGA reset and shut down. Sampled every @c _seq_check_interval
    /// packets, which at typical CRS packet rates is roughly once per second.
    static constexpr uint64_t _seq_check_interval = 500000;
    uint64_t _last_check_seq = 0;
    uint64_t _seq_check_packet_count = 0;

    /// A packet at least this many frames past the start of the next active frame is not caught
    /// up to (see handle_packet). About 0.34 s of 8192-sample frames.
    ///
    /// Why there is a window at all: a packet lands ahead of the active frames when this worker
    /// went without frames to fill. Downstream stalled until the output buffer was full, the
    /// packets that arrived meanwhile were dropped, and the first one after the stall belongs to
    /// a later frame. Skipping forward to it loses only those packets, and the receipt bitmap
    /// already marks them missing. Following a packet further ahead would advance the frames this
    /// worker shares with the port's other worker past packets that worker is still writing, and
    /// then send empty frames downstream.
    ///
    /// Why 8: the stalls seen once the output buffer was full have been at most 4 frames, and a
    /// stall that outlasts the buffer by a third of a second is a downstream fault in itself.
    static constexpr uint64_t _max_frames_ahead = 8;

    /// Packets outside the active frames and the window above. A one-off among good packets is
    /// dropped, logged with its sequence number and counted
    /// (kotekan_dpdk_crs16_out_of_window_packets_total): it may come from a fault in how the
    /// F-engine assembles packets. This many within _out_of_window_burst_gap packets of each
    /// other mean the stream is out of step -- an FPGA reset puts every packet behind the active
    /// frames, a stall longer than the window puts every packet ahead of them, and a board in a
    /// bad state sends bad sequence numbers continuously -- and kotekan stops.
    static constexpr uint64_t _max_out_of_window_burst = 100;
    /// Packets (about 2 s on one worker) after which an out-of-window packet starts a new burst.
    static constexpr uint64_t _out_of_window_burst_gap = 1000000;
    /// Out-of-window packets logged per burst; the rest are only counted.
    static constexpr uint64_t _out_of_window_logged = 10;
    uint64_t _out_of_window_burst = 0;
    uint64_t _packets_since_out_of_window = 0;

    /**
     * @brief Copies one packet's payload into the output frame using non-temporal stores.
     *
     * @param mbuf              The packet to copy.
     * @param frame_ptr         The output frame to copy into.
     * @param relative_seq_num  Sequence number of the packet relative to the frame start.
     * @param stream_id         Raw stream ID of the packet (frequency axis position).
     * @param dest_slot         Output slot on the element axis (the board's raw
     *                          source_id passed through @c dest_slot_for_source_id).
     */
    inline void packet_copy_to_frame(struct rte_mbuf* mbuf, uint8_t* frame_ptr,
                                     uint64_t relative_seq_num, uint16_t stream_id,
                                     uint16_t dest_slot);

    /// kotekan_dpdk_crs16_out_of_window_packets_total for this worker's port.
    kotekan::prometheus::Counter& out_of_window_packets_metric;
};

inline crs16BoardCaptureWorker::crs16BoardCaptureWorker(
    kotekan::Config& config, const std::string& unique_name,
    kotekan::bufferContainer& buffer_container, int worker_id,
    const std::vector<rte_ring*>& worker_rings) :
    dpdkRXhandler(config, unique_name, buffer_container, config.get<int>(unique_name, "port")),
    worker_rings(worker_rings),
    out_buf(buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buf"))),
    receipt_bitmap_buf(
        buffer_container.get_buffer(config.get<std::string>(unique_name, "receipt_bitmap_buf"))),
    worker_id(worker_id), packet_size(config.get<uint32_t>(unique_name, "packet_size")),
    payload_size(config.get<uint32_t>(unique_name, "payload_size")),
    num_expected_stream_ids(config.get<uint32_t>(unique_name, "num_expected_stream_ids")),
    out_of_window_packets_metric(
        kotekan::prometheus::Metrics::instance()
            .add_counter("kotekan_dpdk_crs16_out_of_window_packets_total", unique_name, {"port"})
            .labels({std::to_string(port)})) {

    out_buf->register_producer(unique_name);
    receipt_bitmap_buf->register_producer(unique_name);

    if ((out_buf->frame_size % (payload_size * num_source_ids * num_stream_ids)) != 0) {
        throw std::runtime_error("The buffer frame size must be a multiple of the combined payload "
                                 "size of all source and stream IDs for a given time sample.");
    }

    if ((payload_size % 32) != 0) {
        throw std::runtime_error("The packet_size must be a multiple of 32 bytes");
    }

    if (payload_size + header_size != packet_size) {
        throw std::runtime_error("The packet_size must be payload_size + header_size bytes");
    }

    packets_per_frame = out_buf->frame_size / payload_size;
    time_samples_per_frame =
        packets_per_frame * time_samples_per_packet / (num_source_ids * num_stream_ids);
    INFO("crs16BoardCaptureWorker {}: time_samples_per_frame = {}, packets_per_frame = {}",
         unique_name, time_samples_per_frame, packets_per_frame);

    // Number of frames to capture before stopping, 0 = unlimited
    capture_n_frames = config.get_default<uint64_t>(unique_name, "capture_n_frames", 0);
    if (capture_n_frames > 0)
        INFO("crs16BoardCaptureWorker {}: capturing {} frames, after which the DPDK threads "
             "stop but kotekan keeps running",
             unique_name, capture_n_frames);

    // Optional scheduled start time. An empty string means start as soon as packets flow.
    // Parsing failures are raised here so a typo is caught at startup rather than minutes
    // into a run. The time itself is checked against the F-engine's GPS time once packets
    // arrive, see compute_capture_start_seq().
    const std::string capture_start_time_utc =
        config.get_default<std::string>(unique_name, "capture_start_time_utc", "");
    if (!capture_start_time_utc.empty()) {
        capture_start_time_ns = utc_string_to_nanosec_i64(capture_start_time_utc);
        INFO("crs16BoardCaptureWorker {}: capture scheduled to start at {}", unique_name,
             nanosec_i64_to_utc_string(capture_start_time_ns));
    }

    int prefetch_depth = config.get_default<int>(unique_name, "prefetch_depth", 4);
    std::vector<int> cpu_affinity =
        config.get<std::vector<int>>(unique_name, "frame_service_cpu_affinity");
    bool allocate_metadata = config.get_default<bool>(unique_name, "allocate_metadata", false);

    // Optional reordering of CRS boards in the output frame. The list is the
    // desired board order by output slot: output slot i receives the board whose
    // raw source_id is crs_board_remap[i]. Must be a permutation of
    // [0, num_source_ids). Absent or empty = identity (no reordering).
    std::vector<uint32_t> crs_board_remap =
        config.get_default<std::vector<uint32_t>>(unique_name, "crs_board_remap", {});

    for (uint32_t i = 0; i < num_source_ids; ++i)
        dest_slot_for_source_id[i] = (uint16_t)i;

    if (!crs_board_remap.empty()) {
        if (crs_board_remap.size() != num_source_ids) {
            throw std::runtime_error(
                fmt::format(fmt("crs16BoardCaptureWorker: crs_board_remap must have exactly {:d} "
                                "entries, got {:d}"),
                            num_source_ids, crs_board_remap.size()));
        }
        std::array<bool, num_source_ids> seen{};
        for (uint32_t i = 0; i < num_source_ids; ++i) {
            const uint32_t src = crs_board_remap[i];
            if (src >= num_source_ids) {
                throw std::runtime_error(
                    fmt::format(fmt("crs16BoardCaptureWorker: crs_board_remap entry {:d} is out of "
                                    "range [0, {:d})"),
                                src, num_source_ids));
            }
            if (seen[src]) {
                throw std::runtime_error(fmt::format(
                    fmt("crs16BoardCaptureWorker: crs_board_remap has duplicate entry {:d}"), src));
            }
            seen[src] = true;
            dest_slot_for_source_id[src] = (uint16_t)i; // board `src` lands in output slot i
        }
    }

    prefetch_service = std::make_unique<kotekan::FramePrefetchService>(
        out_buf, receipt_bitmap_buf, unique_name, port, time_samples_per_frame, prefetch_depth,
        cpu_affinity, capture_n_frames, allocate_metadata);
    prefetch_service->set_log_level(get_log_level());
    prefetch_service->set_log_prefix(unique_name + "_prefetch");
}


inline bool crs16BoardCaptureWorker::compute_capture_start_seq(uint64_t current_seq) {

    const Telescope& tel = Telescope::instance();
    time0_ns = tel.to_time_ns(0);
    seq_length_ns = (int64_t)tel.seq_length_nsec();
    capture_start_seq_known = true;

    prefetch_lead_samples = (uint64_t)(_prefetch_lead_s * 1e9 / (double)seq_length_ns);

    if (capture_start_time_ns == 0) {
        // No scheduled start, so begin at the first frame edge past the prefetch lead time.
        const uint64_t future_seq = current_seq + prefetch_lead_samples;
        capture_start_seq = future_seq - (future_seq % time_samples_per_frame);
        return true;
    }

    if (!tel.gps_time_enabled())
        WARN("Port: {:d}, Worker: {:d}; capture_start_time_utc was set, but the telescope has no "
             "GPS time, so FPGA seq numbers are timed against the system clock at kotekan "
             "startup. The capture will not start when requested.",
             port, worker_id);

    // "Now" is the GPS time of the F-engine seq number of the packet we are handling, never
    // the system clock.
    const int64_t now_ns = tel.to_time_ns(current_seq);
    const int64_t wait_ns = capture_start_time_ns - now_ns;

    if (wait_ns < 0) {
        FATAL_ERROR("Port: {:d}, Worker: {:d}; The requested capture_start_time_utc ({:s}) is "
                    "{:s} in the past; the current F-engine time is {:s}. It must be at least "
                    "one minute in the future.",
                    port, worker_id, nanosec_i64_to_utc_string(capture_start_time_ns),
                    format_duration_ns(-wait_ns), nanosec_i64_to_utc_string(now_ns));
        return false;
    }

    if (wait_ns < _min_start_lead_ns) {
        FATAL_ERROR("Port: {:d}, Worker: {:d}; The requested capture_start_time_utc ({:s}) is "
                    "only {:s} away; the current F-engine time is {:s}. It must be at least one "
                    "minute in the future to leave time to set the capture up.",
                    port, worker_id, nanosec_i64_to_utc_string(capture_start_time_ns),
                    format_duration_ns(wait_ns), nanosec_i64_to_utc_string(now_ns));
        return false;
    }

    if (wait_ns > _max_start_lead_ns)
        WARN("Port: {:d}, Worker: {:d}; The requested capture_start_time_utc ({:s}) is {:s} away, "
             "more than 12 hours in the future. Kotekan will sit idle until then.",
             port, worker_id, nanosec_i64_to_utc_string(capture_start_time_ns),
             format_duration_ns(wait_ns));

    // Round the start down to the nearest frame edge. Every worker uses the same frame
    // length, so they all land on the same seq number and produce aligned frames.
    capture_start_seq = tel.to_seq(capture_start_time_ns);
    capture_start_seq -= capture_start_seq % time_samples_per_frame;

    next_countdown_seq = current_seq + (uint64_t)(_countdown_interval_ns / seq_length_ns);

    INFO("Port: {:d}, Worker: {:d}; Capture starts at {:s} (fpga seq {:d}), which is {:s} from "
         "now, rounded down one frame edge from the requested {:s}. Current F-engine time is "
         "{:s}.",
         port, worker_id, nanosec_i64_to_utc_string(seq_to_time_ns(capture_start_seq)),
         capture_start_seq, format_duration_ns(seq_to_time_ns(capture_start_seq) - now_ns),
         nanosec_i64_to_utc_string(capture_start_time_ns), nanosec_i64_to_utc_string(now_ns));

    return true;
}


inline int crs16BoardCaptureWorker::handle_packet(struct rte_mbuf* mbuf) {

    // Packet validation (size, IP checksum, cookie, stream_id range) is done
    // upstream in crs16BoardDistributor before packets reach this worker.

    // Print the worker ID and stream ID
    uint16_t stream_id = get_crs_packet_stream_id(mbuf);
    uint16_t source_id =
        get_crs_packet_source_id(mbuf).slot_id + 16 * get_crs_packet_source_id(mbuf).crate_id;
    uint64_t seq_num = get_crs_packet_seq_num(mbuf);

    // Sample-based FPGA seq monotonicity check: at each sample boundary the
    // current seq must be strictly greater than the previous sample's,
    // otherwise the FPGA likely reset.
    if (unlikely(++_seq_check_packet_count >= _seq_check_interval)) {
        if (unlikely(_last_check_seq != 0 && seq_num < _last_check_seq)) {
            FATAL_ERROR("Port: {:d}, Worker: {:d}; CRS FPGA seq went backwards ({:d} -> {:d}), "
                        "controller likely reset, kotekan stopping...",
                        port, worker_id, _last_check_seq, seq_num);
            return -1;
        }
        _last_check_seq = seq_num;
        _seq_check_packet_count = 0;
    }

    if (unlikely(first_run)) {

        // Establish the list of expected stream IDs for this port.
        // Note that in most cases we expect all source ID at every port.
        // In the 16 board configuration, each stream ID should be spaced by 16.
        // Note that this is slightly slow, but only happens once at startup.
        // Once the list is full we stop looking, both to keep this out of the hot path and
        // so that a capture waiting on a scheduled start does not keep adding to the list.
        if (stream_ids_expected.size() < num_expected_stream_ids) {

            // Check if the stream ID is already in the list, and if not add it.
            if (std::find(stream_ids_expected.begin(), stream_ids_expected.end(), stream_id)
                == stream_ids_expected.end()) {
                stream_ids_expected.push_back(stream_id);
            }

            if (stream_ids_expected.size() < num_expected_stream_ids) {
                return 0; // Wait for more packets to establish the full list
            }
        }

        // Work out where the capture begins, either a couple of seconds from now or at the
        // scheduled UTC start time.
        if (unlikely(!capture_start_seq_known)) {
            if (!compute_capture_start_seq(seq_num))
                return -1;
        }

        // Hold off arming the prefetch service until the start is close. A capture scheduled
        // hours ahead would otherwise sit on empty buffer frames for the whole wait.
        if (seq_num + prefetch_lead_samples < capture_start_seq) {
            if (worker_id == 0 && seq_num >= next_countdown_seq) {
                INFO("Waiting for the scheduled capture start at {:s}; {:s} remaining",
                     nanosec_i64_to_utc_string(seq_to_time_ns(capture_start_seq)),
                     format_duration_ns(seq_to_time_ns(capture_start_seq)
                                        - seq_to_time_ns(seq_num)));
                next_countdown_seq = seq_num + (uint64_t)(_countdown_interval_ns / seq_length_ns);
            }
            return 0;
        }

        prefetch_service->start(capture_start_seq, stream_ids_expected);
        first_run = false;
        INFO("Port: {:d}, Worker: {:d}; Starting prefetch service at sequence number {:d} ({:s})",
             port, worker_id, capture_start_seq,
             nanosec_i64_to_utc_string(seq_to_time_ns(capture_start_seq)));
        return 0;
    }

    // This should only ever happen if the switch or F-engine is misconfigured.
    // Could possibly be removed at some point if we have other checks in place.
    if (std::find(stream_ids_expected.begin(), stream_ids_expected.end(), stream_id)
        == stream_ids_expected.end()) {
        WARN("Port: {:d}, Worker: {:d}; Got packet with unexpected Stream ID {:d}", port, worker_id,
             stream_id);
        return 0;
    }

    // A raw source_id can exceed num_source_ids if a board's crate_id is
    // misconfigured (crate_id is a 4-bit field). Drop such packets: they would
    // index past the remap table, receipt bitmap, and frame board axis.
    if (unlikely(source_id >= num_source_ids)) {
        WARN("Port: {:d}, Worker: {:d}; Got packet with out-of-range Source ID {:d}", port,
             worker_id, source_id);
        return 0;
    }

    // Print packet details for every 100,000 sequence numbers.
#ifdef DEBUGGING
    if ((seq_num / 16) % 100000 == 0) {
        DEBUG("Port: {:d}, Worker: {:d}; Got packet with Stream ID {:d}, Source ID {:d}, Seq Num "
              "{:d}",
              port, worker_id, stream_id, source_id, seq_num);
    }
#endif

    if (unlikely(!prefetch_service->is_ready())) {
        if (prefetch_service->has_error() || prefetch_service->is_complete())
            return -1;

        if (seq_num >= prefetch_service->get_start_seq()) {
            WARN("Port: {:d}, Worker: {:d}; Dropping packet with sequence number {:d} because "
                 "prefetch service is not ready (start_seq: {:d})",
                 port, worker_id, seq_num, prefetch_service->get_start_seq());
        }
        return 0;
    }

    if (unlikely(active_f0 == nullptr)) {
        active_f0 = prefetch_service->get_frame(0);
        active_f1 = prefetch_service->get_frame(1);

        if (unlikely(active_f0 == nullptr || active_f1 == nullptr)) {
            if (prefetch_service->has_error() || prefetch_service->is_complete())
                return -1;
            // TODO add a metric for out of prefetched frame packet drops.
            active_f0 = nullptr;
            active_f1 = nullptr;
            return 0;
        }
    }

    if (seq_num < active_f0->start_seq
        || seq_num >= active_f1->start_seq + _max_frames_ahead * time_samples_per_frame) {

        // Don't warn if we are just waiting for the first frame
        if (seq_num < active_f0->start_seq
            && active_f0->start_seq == prefetch_service->get_start_seq()) {
            return 0;
        }

        // Behind the active frames (late, or after an FPGA reset), or too far ahead to catch up
        // to. A one-off is dropped, logged and counted; a burst of them stops kotekan (see
        // _max_out_of_window_burst), since ending only this worker leaves the frames it shares
        // unfinished.
        out_of_window_packets_metric.inc();
        if (_packets_since_out_of_window > _out_of_window_burst_gap)
            _out_of_window_burst = 0;
        _packets_since_out_of_window = 0;
        if (++_out_of_window_burst < _max_out_of_window_burst) {
            if (_out_of_window_burst > _out_of_window_logged) {
                return 0;
            }
            if (seq_num < active_f0->start_seq) {
                WARN("Port: {:d}, Worker: {:d}; Dropped packet with sequence number {:d} (source "
                     "ID {:d}, stream ID {:d}): behind the active frames [{:d}, {:d})",
                     port, worker_id, seq_num, source_id, stream_id, active_f0->start_seq,
                     active_f1->start_seq + time_samples_per_frame);
            } else {
                WARN("Port: {:d}, Worker: {:d}; Dropped packet with sequence number {:d} (source "
                     "ID {:d}, stream ID {:d}): {:d} frames past the start of the next active "
                     "frame, the limit is {:d}",
                     port, worker_id, seq_num, source_id, stream_id,
                     (seq_num - active_f1->start_seq) / time_samples_per_frame, _max_frames_ahead);
            }
            return 0;
        }
        FATAL_ERROR("Port: {:d}, Worker: {:d}; {:d} packets outside the active frames [{:d}, {:d}) "
                    "and the {:d}-frame window, each within {:d} packets of the last: the stream "
                    "is out of step (an FPGA reset, a stall, or boards sending bad sequence "
                    "numbers). The last had sequence number {:d} (source ID {:d}, stream ID {:d}). "
                    "kotekan stopping...",
                    port, worker_id, _out_of_window_burst, active_f0->start_seq,
                    active_f1->start_seq + time_samples_per_frame, _max_frames_ahead,
                    _out_of_window_burst_gap, seq_num, source_id, stream_id);
        return -1;
    }
    ++_packets_since_out_of_window;

    // Advance the frames while we are at least 160 time samples past the start of the next
    // frame. The 160 time sample margin ensures that we do not miss any packets that are
    // slightly out of order.
    if (unlikely(seq_num >= active_f1->start_seq + time_samples_per_frame)) {
        // Past the next frame: the packets in between were dropped while this worker had no
        // frames (a downstream stall) or never arrived (a gap in the input). The frames passed
        // over go downstream with them missing from the receipt bitmap.
        WARN("Port: {:d}, Worker: {:d}; Packet with sequence number {:d} is {:d} frame(s) past "
             "the start of the next frame; advancing the frames toward it",
             port, worker_id, seq_num, (seq_num - active_f1->start_seq) / time_samples_per_frame);
    }
    while (seq_num >= active_f1->start_seq + 160) {
        _mm_sfence(); // Ensure all NT stores are complete before advancing
        prefetch_service->advance();

        // Advancing hands the frame we just filled to the prefetch service, which marks it
        // full for the downstream stages. Once we have handed over the requested number of
        // frames we stop this lcore, but leave the rest of kotekan running so the file
        // writers can flush what they have.
        frames_captured++;
        if (unlikely(capture_n_frames > 0 && frames_captured >= capture_n_frames)) {
            INFO("Port: {:d}, Worker: {:d}; Captured the requested {:d} frames (seq {:d} to "
                 "{:d}), stopping packet capture.",
                 port, worker_id, capture_n_frames, capture_start_seq,
                 capture_start_seq + capture_n_frames * time_samples_per_frame);
            return -1;
        }

        active_f0 = prefetch_service->get_frame(0);
        active_f1 = prefetch_service->get_frame(1);

        if (unlikely(active_f0 == nullptr || active_f1 == nullptr)) {
            if (prefetch_service->has_error() || prefetch_service->is_complete())
                return -1;
            // TODO add a metric for dropped packets here.
            active_f0 = nullptr;
            active_f1 = nullptr;
            return 0;
        }
    }


    // The packet belongs to the current frame or the next one.
    const kotekan::FrameInfo* frame = (seq_num < active_f1->start_seq) ? active_f0 : active_f1;
    const uint64_t relative_seq_num = seq_num - frame->start_seq;

    // Map the raw board id to its output slot (identity unless crs_board_remap is set).
    const uint16_t dest_slot = dest_slot_for_source_id[source_id];

    packet_copy_to_frame(mbuf, frame->frame_ptr, relative_seq_num, stream_id, dest_slot);

    // Record which packets were received.
    // The layout of the packet receipt bitmap is:
    // packet_receipt[time_long][source_id][stream_id]
    // with the size (in bits) of:
    // time_long = packets_per_frame / (num_source_ids * num_stream_ids)
    // source_id = num_source_ids
    // stream_id = num_stream_ids
    // For the pathfinder this is [512][16][8] = 65536 bits = 8192 bytes
    // The source_id axis is in output-slot order (after crs_board_remap, if configured).
    frame->receipt_bitmap_ptr[(relative_seq_num / time_samples_per_packet) * num_source_ids
                              + dest_slot] |= (1 << (stream_id / 16));

    return 0;
}


inline void crs16BoardCaptureWorker::packet_copy_to_frame(struct rte_mbuf* mbuf, uint8_t* frame_ptr,
                                                          uint64_t relative_seq_num,
                                                          uint16_t stream_id, uint16_t dest_slot) {
    const uint64_t time_long = relative_seq_num / 16;
    const uint64_t element_long = dest_slot;

    // Base offset calculation
    // stride_time_long = (num_stream_ids * 48) * 2048
    // stride_element_long = 128
    uint64_t base_offset = time_long * (uint64_t)(num_stream_ids * 48) * 2048 + element_long * 128;

    uint8_t* dst_base = frame_ptr + base_offset;

    // Source setup
    struct rte_mbuf* m = mbuf;
    uint8_t* src = rte_pktmbuf_mtod_offset(m, uint8_t*, header_size);
    int rem = rte_pktmbuf_data_len(m) - header_size;

    for (int bin = 0; bin < 48; ++bin) {
        uint64_t freq_bin_idx = stream_id / 16 + 8 * bin;
        uint8_t* dst = dst_base + freq_bin_idx * 2048;

        // Copy 128 bytes
        if (likely(rem >= 128)) {
            // Fast path: AVX non-temporal store with latency hiding
            __m256i r0, r1, r2, r3;

            // Check for alignment
            if (((uintptr_t)src & 0x1F) == 0) {
                r0 = _mm256_load_si256((__m256i*)(src + 0));
                r1 = _mm256_load_si256((__m256i*)(src + 32));
                r2 = _mm256_load_si256((__m256i*)(src + 64));
                r3 = _mm256_load_si256((__m256i*)(src + 96));
            } else {
                r0 = _mm256_loadu_si256((__m256i*)(src + 0));
                r1 = _mm256_loadu_si256((__m256i*)(src + 32));
                r2 = _mm256_loadu_si256((__m256i*)(src + 64));
                r3 = _mm256_loadu_si256((__m256i*)(src + 96));
            }

            _mm256_stream_si256((__m256i*)(dst + 0), r0);
            _mm256_stream_si256((__m256i*)(dst + 32), r1);
            _mm256_stream_si256((__m256i*)(dst + 64), r2);
            _mm256_stream_si256((__m256i*)(dst + 96), r3);

            src += 128;
            rem -= 128;
        } else {
            // Slow path: Handle split across mbuf segments
            int copied = 0;
            while (copied < 128) {
                if (rem == 0) {
                    m = m->next;
                    if (unlikely(m == nullptr))
                        return;
                    src = rte_pktmbuf_mtod(m, uint8_t*);
                    rem = rte_pktmbuf_data_len(m);
                }
                int n = std::min(128 - copied, rem);

                uint8_t* d = dst + copied;
                uint8_t* s = src;
                int left = n;

                while (left > 0) {
                    // Try 32-byte aligned streaming store
                    if (left >= 32 && ((uintptr_t)d & 0x1F) == 0) {
                        _mm256_stream_si256((__m256i*)d, _mm256_loadu_si256((__m256i*)s));
                        d += 32;
                        s += 32;
                        left -= 32;
                        continue;
                    }
                    // Try 16-byte aligned streaming store
                    if (left >= 16 && ((uintptr_t)d & 0xF) == 0) {
                        _mm_stream_si128((__m128i*)d, _mm_loadu_si128((__m128i*)s));
                        d += 16;
                        s += 16;
                        left -= 16;
                        continue;
                    }
                    // Try 8-byte aligned streaming store
                    if (left >= 8 && ((uintptr_t)d & 0x7) == 0) {
                        int64_t val;
                        std::memcpy(&val, s, sizeof(val));
                        _mm_stream_si64((long long*)d, (long long)val);
                        d += 8;
                        s += 8;
                        left -= 8;
                        continue;
                    }
                    // Try 4-byte aligned streaming store
                    if (left >= 4 && ((uintptr_t)d & 0x3) == 0) {
                        int32_t val;
                        std::memcpy(&val, s, sizeof(val));
                        _mm_stream_si32((int*)d, val);
                        d += 4;
                        s += 4;
                        left -= 4;
                        continue;
                    }

                    // Fallback: byte copy (standard store)
                    *d++ = *s++;
                    left--;
                }

                src += n;
                rem -= n;
                copied += n;
            }
        }
    }
}


#endif
