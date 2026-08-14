#include "GnssTelemPack.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <cstring>
#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssTelemPack);

GnssTelemPack::GnssTelemPack(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssTelemPack::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _chain = config.get<std::string>(unique_name, "chain");
    _inst = config.get<std::string>(unique_name, "inst");
    _n_prn = config.get<int>(unique_name, "n_prn");
    _n_elem = config.get_default<int>(unique_name, "n_elements", 0);
    _max_prn = config.get<int>(unique_name, "max_prn");
    _rec_per_frame = config.get_default<int>(unique_name, "records_per_frame", 4);
    _hops_per_record = config.get<int>(unique_name, "hops_per_record");
    _fft_len = config.get<int>(unique_name, "fft_len");
    _n_chan = config.get_default<int>(unique_name, "n_chan", 0);

    // Every one of these fails SILENTLY if it is wrong -- a truncated tag attributes a chain's
    // data to another chain, an over-long PRN list writes past the row block, a records_per_frame
    // past the ceiling overflows `present`. So they are all fatal at construction, where the
    // config that caused them is still on screen.
    if (_chain.size() >= gnss::TELEM_NAME || _inst.size() >= gnss::TELEM_NAME) {
        FATAL_ERROR("GnssTelemPack[{:s}]: chain '{:s}' / inst '{:s}' must be < {:d} chars -- these "
                    "tags are how the broker tells senders apart, and a truncated one silently "
                    "merges two streams",
                    unique_name, _chain, _inst, gnss::TELEM_NAME);
        return;
    }
    if (_max_prn < _n_prn) {
        FATAL_ERROR("GnssTelemPack[{:s}]: max_prn {:d} < n_prn {:d}", unique_name, _max_prn,
                    _n_prn);
        return;
    }
    if (_rec_per_frame < 1 || _rec_per_frame > gnss::TELEM_MAX_REC) {
        FATAL_ERROR("GnssTelemPack[{:s}]: records_per_frame {:d} outside [1, {:d}]", unique_name,
                    _rec_per_frame, gnss::TELEM_MAX_REC);
        return;
    }
    if (_hops_per_record <= 0 || _fft_len <= 0) {
        FATAL_ERROR("GnssTelemPack[{:s}]: hops_per_record {:d} and fft_len {:d} must both be > 0 "
                    "-- they define the window index every instance collates on",
                    unique_name, _hops_per_record, _fft_len);
        return;
    }

    _rec_samples = _hops_per_record * _fft_len;
    _win_samples = (int64_t)_rec_per_frame * _rec_samples;

    const size_t need = gnss::telem_frame_bytes(_rec_per_frame, _max_prn);
    if ((size_t)out_buf->frame_size != need) {
        // EXACT, not ">=": bufferRecv compares frame_size on the wire against its own buffer and
        // closes the connection on a mismatch, so a receiver sized from a different max_prn
        // would simply never receive. Catch it here, on the sender, where the message can name
        // the two numbers.
        FATAL_ERROR("GnssTelemPack[{:s}]: out_buf frame is {:d} B but the wire format needs "
                    "exactly {:d} B ({:d} records x {:d} PRN x {:d} floats + {:d} B header). The "
                    "gather's receive buffer must be sized from the SAME two numbers.",
                    unique_name, (size_t)out_buf->frame_size, need, _rec_per_frame, _max_prn,
                    gnss::RECORD_FLOATS, sizeof(gnss::TelemHeader));
        return;
    }
    const size_t in_need = (size_t)_n_prn * gnss::record_stride(_n_elem) * sizeof(float);
    if ((size_t)in_buf->frame_size < in_need) {
        FATAL_ERROR("GnssTelemPack[{:s}]: in_buf frame {:d} B < {:d} B ({:d} PRN x stride {:d})",
                    unique_name, (size_t)in_buf->frame_size, in_need, _n_prn,
                    gnss::record_stride(_n_elem));
        return;
    }

    _rows.assign((size_t)_rec_per_frame * _max_prn * gnss::RECORD_FLOATS, 0.0f);

    INFO("GnssTelemPack[{:s}]: {:s}/{:s} -- {:d} records/frame x {:d} PRN rows x {:d} floats = "
         "{:d} B/frame; window = {:d} hops = {:d} samples on the F-engine clock",
         unique_name, _chain, _inst, _rec_per_frame, _max_prn, gnss::RECORD_FLOATS, need,
         _win_samples / _fft_len, _win_samples);
}

bool GnssTelemPack::flush(frameID& out_id) {
    if (_cur_win < 0)
        return true;

    uint8_t* frame = out_buf->wait_for_empty_frame(unique_name, out_id);
    if (frame == nullptr)
        return false;

    gnss::TelemHeader h;
    std::memset(&h, 0, sizeof(h));
    h.magic = gnss::TELEM_MAGIC;
    h.version = gnss::TELEM_VERSION;
    h.n_rec = (uint16_t)_rec_per_frame;
    h.n_prn = (uint16_t)_max_prn;
    h.n_row = (uint16_t)gnss::RECORD_FLOATS;
    h.n_chan = (uint16_t)_n_chan;
    h.n_elem = (uint16_t)_n_elem;
    h.hops_per_record = (uint32_t)_hops_per_record;
    h.fft_len = (uint32_t)_fft_len;
    h.win = (uint64_t)_cur_win;
    h.seq = _seq++;
    h.wstart0 = _cur_win * _win_samples;
    h.utc0 = _cur_utc0;
    h.present = _cur_present;
    gnss::telem_set_name(h.chain, _chain);
    gnss::telem_set_name(h.inst, _inst);

    std::memcpy(frame, &h, sizeof(h));
    std::memcpy(frame + sizeof(h), _rows.data(), _rows.size() * sizeof(float));

    // The metadata carries the window's first sample, so a kotekan consumer downstream of
    // bufferRecv can address the frame without parsing the payload. It is NOT the key the
    // broker uses -- that is the header's `win` -- but keeping the two consistent means a
    // frame is self-describing from either side.
    if (out_buf->metadata_pool) {
        out_buf->allocate_new_metadata_object(out_id);
        get_gnss_chan_metadata(out_buf, out_id)->sample_seq = h.wstart0;
    }
    out_buf->mark_frame_full(unique_name, out_id++);

    _cur_win = -1;
    _cur_present = 0;
    _cur_utc0 = 0.0;
    std::fill(_rows.begin(), _rows.end(), 0.0f);
    return true;
}

void GnssTelemPack::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf);
    const int in_stride = gnss::record_stride(_n_elem);

    while (!stop_thread) {
        float* in = (float*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;

        int64_t wstart = -1;
        if (metadata_is_gnss_chan(in_buf)) {
            const GnssChanMetadata* m = get_gnss_chan_metadata(in_buf, in_id);
            if (m)
                wstart = m->sample_seq;
        }

        if (wstart < 0 || (wstart % _rec_samples) != 0) {
            // No absolute address, or one that is not on the record grid. There is nothing
            // honest to do with such a record: placing it anywhere would invent an alignment.
            // Rate-limited, because if it ever fires it fires on every record.
            if ((_dropped++ % 1000) == 0)
                WARN("GnssTelemPack[{:s}]: dropping record with wstart {:d} (metadata missing, or "
                     "not a multiple of the {:d}-sample record) -- {:d} so far",
                     unique_name, wstart, _rec_samples, _dropped);
            in_buf->mark_frame_empty(unique_name, in_id++);
            continue;
        }

        // wstart >= 0 and on the record grid by the check above, so this division is a floor.
        const int64_t win = wstart / _win_samples;

        if (_cur_win >= 0 && win != _cur_win) {
            // ANY change closes the open frame, including a jump backwards. Backwards means the
            // F-engine epoch moved or this stage was fed a replay; either way the old window is
            // finished and holding it open would merge two different skies.
            if (!flush(out_id))
                break;
        }
        if (_cur_win < 0)
            _cur_win = win;

        const int slot = (int)((wstart - win * _win_samples) / _rec_samples);
        if (slot < 0 || slot >= _rec_per_frame) {
            // Unreachable given the divisibility check above; kept because the alternative to a
            // guard here is a write past the row block.
            ERROR("GnssTelemPack[{:s}]: record at wstart {:d} maps to slot {:d} of {:d}",
                  unique_name, wstart, slot, _rec_per_frame);
            in_buf->mark_frame_empty(unique_name, in_id++);
            continue;
        }

        for (int p = 0; p < _n_prn; ++p)
            std::memcpy(&_rows[gnss::telem_row_offset(slot, p, _max_prn)],
                        in + (size_t)p * in_stride, gnss::RECORD_FLOATS * sizeof(float));
        _cur_present |= (1u << slot);
        if (slot == 0 && _n_prn > 0)
            _cur_utc0 = *reinterpret_cast<const double*>(in + gnss::RECORD_UTC_SLOT);
        _n_records++;

        in_buf->mark_frame_empty(unique_name, in_id++);

        // A full window is emitted immediately rather than waiting for the next record to
        // notice the boundary: at 4 records/frame that is 42 ms of latency saved, and it means a
        // chain whose records stop arriving leaves no complete window sitting unsent.
        if (_cur_present == ((1u << _rec_per_frame) - 1u))
            if (!flush(out_id))
                break;
    }
    flush(out_id);
}
