#include "crsWindowAlign.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "crsPacket.hpp"         // for crs_header_of, pack12, CRS_PACKET_COOKIE
#include "kotekanLogging.hpp"    // for FATAL_ERROR, WARN, INFO, DEBUG
#include "prometheusMetrics.hpp" // for Metrics
#include "udpPacket.hpp"         // for udp_slot_tail_of, UDP_SLOT_CSUM_OK
#include "visUtil.hpp"           // for frameID, current_time, double_to_ts

#include "fmt.hpp" // for format

#include <algorithm>  // for clamp, max, min
#include <functional> // for bind
#include <math.h>     // for lround
#include <random>     // for mt19937, normal_distribution
#include <stdexcept>  // for invalid_argument

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(crsWindowAlign);

crsWindowAlign::crsWindowAlign(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&crsWindowAlign::main_thread, this)),
    header_buf(get_buffer("header_buf")), payload_buf(get_buffer("payload_buf")),
    job_buf(get_buffer("job_buf")), noise_buf(get_buffer("noise_buf")),
    geometry(udp_slot_geometry::from_config(config, unique_name, header_buf, payload_buf)),
    packets_counter(
        Metrics::instance().add_counter("kotekan_crs_window_align_packets_total", unique_name)),
    bad_counter(
        Metrics::instance().add_counter("kotekan_crs_window_align_bad_packets_total", unique_name)),
    discarded_counter(Metrics::instance().add_counter(
        "kotekan_crs_window_align_discarded_packets_total", unique_name)),
    reordered_counter(Metrics::instance().add_counter(
        "kotekan_crs_window_align_reordered_packets_total", unique_name)),
    lost_counter(Metrics::instance().add_counter("kotekan_crs_window_align_lost_packets_total",
                                                 unique_name)),
    windows_counter(
        Metrics::instance().add_counter("kotekan_crs_window_align_windows_total", unique_name)),
    realigns_counter(
        Metrics::instance().add_counter("kotekan_crs_window_align_realigns_total", unique_name)),
    current_seq_gauge(
        Metrics::instance().add_gauge("kotekan_crs_window_align_current_seq", unique_name)) {

    fft_length = config.get<uint64_t>(unique_name, "fft_length");
    samples_per_packet = config.get_default<uint32_t>(unique_name, "samples_per_packet", 4096);
    adc_id = config.get_default<uint32_t>(unique_name, "adc_id", 0);
    noise_slots = config.get_default<uint32_t>(unique_name, "noise_slots", 1024);
    noise_mean = config.get_default<float>(unique_name, "noise_mean", 0.0f);
    noise_rms = config.get_default<float>(unique_name, "noise_rms", 100.0f);
    seed = config.get_default<uint32_t>(unique_name, "seed", 42);
    max_gap_windows = config.get_default<uint32_t>(unique_name, "max_gap_windows", 1);
    stall_warn_s = config.get_default<float>(unique_name, "stall_warn_s", 5.0f);
    stall_fatal_s = config.get_default<float>(unique_name, "stall_fatal_s", 0.0f);

    if (samples_per_packet % 2 != 0 || geometry.payload_size != (size_t)samples_per_packet * 3 / 2)
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: payload_size {:d} does not hold samples_per_packet {:d} 12-bit samples"),
            unique_name, geometry.payload_size, samples_per_packet));
    if (fft_length == 0 || fft_length % samples_per_packet != 0)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: fft_length {:d} is not a multiple of samples_per_packet {:d}"),
                        unique_name, fft_length, samples_per_packet));
    num_packets = fft_length / samples_per_packet;
    if (job_buf->frame_size < fft_window_job_size(num_packets))
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: job_buf frames hold {:d} bytes but a window of {:d} packets needs {:d}"),
            unique_name, job_buf->frame_size, num_packets, fft_window_job_size(num_packets)));
    if (job_buf->num_frames < 2)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: job_buf needs at least 2 frames"), unique_name));
    if (noise_slots == 0 || noise_buf->frame_size < noise_slots * geometry.payload_slot)
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: noise_buf frames hold {:d} bytes but noise_slots {:d} need {:d}"),
            unique_name, noise_buf->frame_size, noise_slots, noise_slots * geometry.payload_slot));

    // The FFT stage holds every frame of a window while the receive stage posts ahead; the
    // payload ring must hold both or the pipeline wedges.
    const uint32_t queue_depth = config.get_default<uint32_t>(unique_name, "queue_depth", 8192);
    const int window_frames =
        (num_packets + geometry.packets_per_frame - 1) / geometry.packets_per_frame;
    const int posted =
        std::min<int>(queue_depth / geometry.packets_per_frame, payload_buf->num_frames);
    if (payload_buf->num_frames < window_frames + posted + 2)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: payload_buf needs at least {:d} frames: {:d} for a window, "
                            "{:d} posted by the receive stage, 2 slack (it has {:d})"),
                        unique_name, window_frames + posted + 2, window_frames, posted,
                        payload_buf->num_frames));

    header_buf->register_consumer(unique_name);
    job_buf->register_producer(unique_name);
    noise_buf->register_producer(unique_name);
    // payload_buf is deliberately not consumed here: only its frame addresses are used, and
    // the FFT stage holds the frames (see fftWindowJob.hpp).
}

bool crsWindowAlign::fill_noise() {
    uint8_t* frame = noise_buf->wait_for_empty_frame(unique_name, 0);
    if (frame == nullptr)
        return false;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(noise_mean, noise_rms);
    for (size_t s = 0; s < noise_slots; s++) {
        uint8_t* slot = frame + s * geometry.payload_slot;
        for (uint32_t k = 0; k < samples_per_packet; k++)
            pack12(slot, k, (int16_t)std::clamp(lround(dist(rng)), -2048l, 2047l));
    }
    noise_base = frame;
    noise_buf->mark_frame_full(unique_name, 0);
    return true;
}

bool crsWindowAlign::ensure_job() {
    uint8_t* frame = job_buf->wait_for_empty_frame(unique_name, job_frame_id);
    if (frame == nullptr)
        return false;
    job = reinterpret_cast<fft_window_job*>(frame);
    lut = fft_window_lut(job);
    return true;
}

bool crsWindowAlign::emit_window(uint64_t closer_frame, size_t closer_slot, bool closed_by_packet) {
    job->start_seq = window_start;
    job->window_index = window_start / fft_length;
    job->num_packets = num_packets;
    job->lost_packets = lost_in_window;
    uint64_t release = closed_by_packet
                           ? closer_frame + (closer_slot + 1 == geometry.packets_per_frame)
                           : closer_frame;
    release = std::max(release, prev_release);
    uint64_t acquire = have_ref ? last_ref_frame + 1 : prev_acquire;
    acquire = std::max({acquire, prev_acquire, release});
    job->acquire_up_to = acquire;
    job->release_up_to = release;
    job->flags = realign_pending ? FFT_WINDOW_REALIGNED : 0;
    job->reserved = 0;
    realign_pending = false;
    prev_acquire = acquire;
    prev_release = release;

    job_buf->mark_frame_full(unique_name, job_frame_id);
    job_frame_id = (job_frame_id + 1) % job_buf->num_frames;
    windows_counter.inc();
    noise_window_run = lost_in_window == num_packets ? noise_window_run + 1 : 0;

    window_start += fft_length;
    cursor = 0;
    lost_in_window = 0;
    have_ref = false;
    return ensure_job();
}

void crsWindowAlign::realign(uint64_t seq) {
    realigns_counter.inc();
    window_start = (seq + fft_length - 1) / fft_length * fft_length;
    expected_seq = window_start;
    cursor = 0;
    lost_in_window = 0;
    have_ref = false;
    noise_window_run = 0;
    realign_pending = true;
    INFO("Re-aligned at seq {:d}: next window starts at {:d}", seq, window_start);
}

bool crsWindowAlign::process_packet(uint64_t seq, uint64_t frame, size_t slot) {
    if (!aligned) {
        window_start = (seq + fft_length - 1) / fft_length * fft_length;
        expected_seq = window_start;
        aligned = true;
        realign_pending = true;
        INFO("First packet at seq {:d}; discarding up to the window boundary at {:d}", seq,
             window_start);
    }
    if (window_start > seq) {
        if (window_start - seq > fft_length)
            FATAL_ERROR("seq_number went back {:d} samples (from window start {:d} to {:d}): "
                        "the D-engine restarted",
                        window_start - seq, window_start, seq);
        discarded_counter.inc();
        return true;
    }
    if (seq < expected_seq) {
        reordered_counter.inc();
        return true;
    }
    while (seq > expected_seq) {
        if (cursor == 0 && seq >= window_start + fft_length
            && noise_window_run >= max_gap_windows) {
            realign(seq);
            if (window_start > seq) {
                discarded_counter.inc();
                return true;
            }
            break;
        }
        prng_state ^= prng_state << 13;
        prng_state ^= prng_state >> 7;
        prng_state ^= prng_state << 17;
        lut[cursor++] =
            (uint64_t)(uintptr_t)(noise_base + (prng_state % noise_slots) * geometry.payload_slot);
        lost_in_window++;
        lost_counter.inc();
        expected_seq += samples_per_packet;
        if (cursor == num_packets && !emit_window(frame, slot, false))
            return false;
    }
    if (seq == expected_seq) {
        lut[cursor++] = (uint64_t)(uintptr_t)(payload_buf->frames[frame % payload_buf->num_frames]
                                              + slot * geometry.payload_slot);
        last_ref_frame = frame;
        have_ref = true;
        expected_seq += samples_per_packet;
        if (cursor == num_packets)
            return emit_window(frame, slot, true);
    }
    return true;
}

void crsWindowAlign::main_thread() {
    if (!fill_noise())
        return;
    if (!ensure_job())
        return;
    prng_state = seed | 1;

    frameID frame_id(header_buf);
    float stalled_s = 0;
    bool warned = false;
    uint64_t last_seq = 0;
    bool seen_good = false;

    while (!stop_thread) {
        const auto deadline = double_to_ts(current_time() + 1.0);
        const int status = header_buf->wait_for_full_frame_timeout(unique_name, frame_id, deadline);
        if (status == -1)
            break;
        if (status == 1) {
            stalled_s += 1;
            if (stall_fatal_s > 0 && stalled_s >= stall_fatal_s)
                FATAL_ERROR("No packets for {:.0f} s", stalled_s);
            if (!warned && stalled_s >= stall_warn_s) {
                WARN("No packets for {:.0f} s", stalled_s);
                warned = true;
            }
            continue;
        }
        stalled_s = 0;
        warned = false;

        const uint8_t* headers = header_buf->frames[frame_id];
        for (size_t j = 0; j < geometry.packets_per_frame; j++) {
            const uint8_t* slot = headers + j * geometry.header_slot;
            const udp_slot_tail* tail =
                udp_slot_tail_of(const_cast<uint8_t*>(slot), geometry.header_slot);
            const crs_packet_header* crs = crs_header_of(slot);
            if (tail->wire_len != geometry.wire_len() || !(tail->flags & UDP_SLOT_CSUM_OK)
                || crs->cookie != CRS_PACKET_COOKIE || crs->stream_id != adc_id
                || crs->seq_number % samples_per_packet != 0) {
                bad_counter.inc();
                continue;
            }
            last_seq = crs->seq_number;
            seen_good = true;
            if (!process_packet(last_seq, abs_frame, j))
                return;
        }
        packets_counter.inc(geometry.packets_per_frame);
        if (seen_good)
            current_seq_gauge.set(last_seq);

        header_buf->mark_frame_empty(unique_name, frame_id);
        frame_id++;
        abs_frame++;
    }
}
