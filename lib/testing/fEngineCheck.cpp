#include "fEngineCheck.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "crsPacket.hpp"         // for crs_header_of, CRS_PACKET_COOKIE
#include "errors.h"              // for exit_kotekan, TEST_PASSED, TEST_FAILED
#include "kotekanLogging.hpp"    // for ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "udpPacket.hpp"         // for udp_slot_tail_of, UDP_SLOT_CSUM_OK
#include "visUtil.hpp"           // for frameID

#include "fmt.hpp" // for format

#include <functional> // for bind
#include <stdexcept>  // for invalid_argument
#include <stdlib.h>   // for abs

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(fEngineCheck);

fEngineCheck::fEngineCheck(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&fEngineCheck::main_thread, this)),
    header_buf(get_buffer("header_buf")), payload_buf(get_buffer("payload_buf")),
    geometry(udp_slot_geometry::from_config(config, unique_name, header_buf, payload_buf)),
    windows_counter(
        Metrics::instance().add_counter("kotekan_fengine_check_windows_total", unique_name)),
    bad_windows_counter(
        Metrics::instance().add_counter("kotekan_fengine_check_bad_windows_total", unique_name)) {

    adc_id = config.get_default<uint32_t>(unique_name, "adc_id", 0);
    tone_bin = config.get<uint32_t>(unique_name, "tone_bin");
    expected_real = config.get<int>(unique_name, "expected_real");
    tolerance = config.get_default<int>(unique_name, "tolerance", 1);
    num_windows = config.get_default<int>(unique_name, "num_windows", -1);
    max_bad_windows = config.get_default<int64_t>(unique_name, "max_bad_windows", 0);

    if (tone_bin >= geometry.packets_per_frame * geometry.payload_size)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: tone_bin {:d} is outside the {:d} bins of a window"),
                        unique_name, tone_bin, geometry.packets_per_frame * geometry.payload_size));

    header_buf->register_consumer(unique_name);
    payload_buf->register_consumer(unique_name);
}

bool fEngineCheck::check_window(const uint8_t* headers, const uint8_t* payload) {
    bool good = true;
    int logged = 0;
    const uint64_t seq = crs_header_of(headers)->seq_number;

    for (size_t p = 0; p < geometry.packets_per_frame; p++) {
        const uint8_t* slot = headers + p * geometry.header_slot;
        const udp_slot_tail* tail =
            udp_slot_tail_of(const_cast<uint8_t*>(slot), geometry.header_slot);
        const crs_packet_header* crs = crs_header_of(slot);
        if (tail->wire_len != geometry.wire_len() || crs->cookie != CRS_PACKET_COOKIE
            || crs->stream_id != p || crs->source_id.link_id != adc_id || crs->seq_number != seq) {
            ERROR("Packet {:d}: wire_len {:d}, cookie {:#04x}, stream_id {:d}, link_id {:d}, "
                  "seq {:d} (window seq {:d})",
                  p, tail->wire_len, crs->cookie, crs->stream_id, (int)crs->source_id.link_id,
                  crs->seq_number, seq);
            good = false;
        }
    }
    if (have_seq && seq != expected_seq) {
        ERROR("Window seq {:d}, expected {:d}", seq, expected_seq);
        good = false;
    }
    have_seq = true;
    expected_seq = seq + 1;

    const size_t num_bins = geometry.packets_per_frame * geometry.payload_size;
    for (size_t b = 0; b < num_bins; b++) {
        const uint8_t byte =
            payload[b / geometry.payload_size * geometry.payload_slot + b % geometry.payload_size];
        const int real = (byte >> 4) - 8;
        const int imag = (byte & 0xf) - 8;
        bool ok;
        if (b == tone_bin) {
            ok = abs(real - expected_real) <= tolerance && abs(imag) <= tolerance;
            if (real == expected_real)
                saw_exact = true;
        } else {
            ok = abs(real) <= tolerance && abs(imag) <= tolerance;
        }
        if (!ok) {
            good = false;
            if (logged++ < 4)
                ERROR("Window seq {:d} bin {:d}: {:d}{:+d}i ({:#04x})", seq, b, real, imag, byte);
        }
    }
    return good;
}

void fEngineCheck::main_thread() {
    frameID frame_id(header_buf);
    int checked = 0;

    while (!stop_thread && (num_windows < 0 || checked < num_windows)) {
        const uint8_t* headers = header_buf->wait_for_full_frame(unique_name, frame_id);
        if (headers == nullptr)
            break;
        const uint8_t* payload = payload_buf->wait_for_full_frame(unique_name, frame_id);
        if (payload == nullptr)
            break;

        if (!check_window(headers, payload)) {
            bad_windows++;
            bad_windows_counter.inc();
        }
        windows_counter.inc();
        checked++;

        header_buf->mark_frame_empty(unique_name, frame_id);
        payload_buf->mark_frame_empty(unique_name, frame_id);
        frame_id++;
    }

    if (num_windows >= 0 && checked >= num_windows) {
        INFO("Checked {:d} windows: {:d} bad (max {:d}), exact tone value {:s}", checked,
             bad_windows, max_bad_windows, saw_exact ? "seen" : "never seen");
        if (bad_windows > max_bad_windows || !saw_exact)
            TEST_FAILED();
        else
            TEST_PASSED();
    }
}
