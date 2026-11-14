#include "gnssTelemChain.hpp"

#include "gnssTelem.hpp"

#include <algorithm>
#include <cstring>

using kotekan::Config;

namespace gnss {

TelemChain::TelemChain(Config& config, const std::string& cfg_path, const std::string& name) :
    _name(name) {
    set_log_level(config.get<std::string>(cfg_path, "log_level"));
    set_log_prefix(name);
    _chain = config.get<std::string>(cfg_path, "chain");
    _inst = config.get<std::string>(cfg_path, "inst");
    _n_prn = config.get<int>(cfg_path, "n_prn");
    _n_elem = config.get_default<int>(cfg_path, "n_elements", 0);
    _max_prn = config.get<int>(cfg_path, "max_prn");
    _rec_per_frame = config.get_default<int>(cfg_path, "records_per_frame", 4);
    _hops_per_record = config.get<int>(cfg_path, "hops_per_record");
    _fft_len = config.get<int>(cfg_path, "fft_len");
    _n_chan = config.get_default<int>(cfg_path, "n_chan", 0);
    // The comb's column labels, sent in every frame header.
    _chan_ids = config.get_default<std::vector<int>>(cfg_path, "channel_ids", {});
    if (!config.get_default<bool>(cfg_path, "chan_export", true)) {
        FATAL_ERROR("GnssTelemPack[{:s}]: chan_export must be true -- every frame ships the comb",
                    _name);
        return;
    }

    // Fatal here, where the config is to blame: a truncated tag would merge two senders'
    // streams, and the gather drops every frame with more than TELEM_MAX_REC records.
    if (_chain.size() >= gnss::TELEM_NAME || _inst.size() >= gnss::TELEM_NAME) {
        FATAL_ERROR("GnssTelemPack[{:s}]: chain '{:s}' / inst '{:s}' must be < {:d} chars -- these "
                    "tags are how the broker tells senders apart, and a truncated one silently "
                    "merges two streams",
                    _name, _chain, _inst, gnss::TELEM_NAME);
        return;
    }
    // max_prn is a wire capacity, not n_prn: rows are compacted to the PRNs despread in each
    // window. Overflow is checked per window in add_record and only WARNs; telemetry is not
    // worth stopping the tracker for.
    if (_max_prn < 1) {
        FATAL_ERROR("GnssTelemPack[{:s}]: max_prn {:d} must be >= 1", _name, _max_prn);
        return;
    }
    if (_max_prn < _n_prn)
        INFO("GnssTelemPack[{:s}]: wire carries {:d} PRN rows for a {:d}-slot record buffer -- "
             "rows are compacted to the PRNs with signal, and a window with more than {:d} live "
             "will WARN and ship the lowest-numbered ones",
             _name, _max_prn, _n_prn, _max_prn);
    if (_rec_per_frame < 1 || _rec_per_frame > gnss::TELEM_MAX_REC) {
        FATAL_ERROR("GnssTelemPack[{:s}]: records_per_frame {:d} outside [1, {:d}]", _name,
                    _rec_per_frame, gnss::TELEM_MAX_REC);
        return;
    }
    if (_hops_per_record <= 0 || _fft_len <= 0) {
        FATAL_ERROR("GnssTelemPack[{:s}]: hops_per_record {:d} and fft_len {:d} must both be > 0 "
                    "-- they define the window index every instance collates on",
                    _name, _hops_per_record, _fft_len);
        return;
    }

    if ((int)_chan_ids.size() != _n_chan || _n_chan <= 0) {
        FATAL_ERROR("GnssTelemPack[{:s}]: the comb needs n_chan >= 1 and channel_ids matching it "
                    "(got {:d} ids for n_chan {:d}). The broker labels the comb from these; a "
                    "mismatch mislabels the frequency axis of every delay fit.",
                    _name, (int)_chan_ids.size(), _n_chan);
        return;
    }
    if (_n_chan > gnss::TELEM_MAX_CHAN) {
        FATAL_ERROR("GnssTelemPack[{:s}]: {:d} channels exceeds the wire's {:d} columns. "
                    "TELEM_MAX_CHAN is a WIRE constant -- raising it means rebuilding and "
                    "restarting BOTH ends and regenerating the gather config.",
                    _name, _n_chan, gnss::TELEM_MAX_CHAN);
        return;
    }

    _rec_samples = _hops_per_record * _fft_len;
    _win_samples = (int64_t)_rec_per_frame * _rec_samples;

    _rows.assign((size_t)_rec_per_frame * _max_prn * gnss::telem_row_floats(_n_chan), 0.0f);
    _row_of.assign((size_t)_n_prn, -1);
}

void TelemChain::check_bufs(const Buffer* in_buf, const Buffer* out_buf) {
    // Each sender ships only its own comb columns; TELEM_MAX_CHAN is the format's ceiling.
    const size_t need = gnss::telem_frame_bytes(_rec_per_frame, _max_prn, _n_chan);
    if ((size_t)out_buf->frame_size != need) {
        // Exact, not ">=": bufferSend ships the whole buffer frame, so padding would cross the
        // link for nothing, and bufferRecv drops the connection when a frame exceeds the
        // gather's receive buffer. The config generator sizes out_buf from the same three
        // numbers.
        FATAL_ERROR("GnssTelemPack[{:s}]: out_buf frame is {:d} B but the wire format needs "
                    "exactly {:d} B ({:d} records x {:d} PRN x {:d} floats + {:d} B header). The "
                    "gather's receive buffer must be sized from the SAME two numbers.",
                    _name, (size_t)out_buf->frame_size, need, _rec_per_frame, _max_prn,
                    gnss::telem_row_floats(_n_chan), sizeof(gnss::TelemHeader));
        return;
    }
    const size_t in_need = gnss::frame_floats(_n_prn, _n_elem, _n_chan) * sizeof(float);
    if ((size_t)in_buf->frame_size < in_need) {
        FATAL_ERROR("GnssTelemPack[{:s}]: in_buf frame {:d} B < {:d} B ({:d} PRN x stride {:d})",
                    _name, (size_t)in_buf->frame_size, in_need, _n_prn,
                    gnss::record_stride(_n_elem));
        return;
    }

    INFO("GnssTelemPack[{:s}]: {:s}/{:s} -- {:d} records/frame x {:d} PRN rows x {:d} floats = "
         "{:d} B/frame; window = {:d} hops = {:d} samples on the F-engine clock",
         _name, _chain, _inst, _rec_per_frame, _max_prn, gnss::RECORD_FLOATS, need,
         _win_samples / _fft_len, _win_samples);
}

bool TelemChain::on_grid(int64_t wstart) {
    if (wstart < 0 || (wstart % _rec_samples) != 0) {
        // No absolute address, or one off the record grid: the record has no slot. Logged
        // every 1000th, because if it fires it fires on every record.
        if ((_dropped++ % 1000) == 0)
            WARN("GnssTelemPack[{:s}]: dropping record with wstart {:d} (metadata missing, or "
                 "not a multiple of the {:d}-sample record) -- {:d} so far",
                 _name, wstart, _rec_samples, _dropped);
        return false;
    }
    return true;
}

bool TelemChain::window_changes(int64_t wstart) const {
    // Any change of window closes the open frame, including a backwards jump (an
    // F-engine re-base or a replay).
    return _cur_win >= 0 && wstart / _win_samples != _cur_win;
}

void TelemChain::add_record(const float* in, int64_t wstart) {
    const int in_stride = gnss::record_stride(_n_elem);
    // wstart >= 0, so this division is a floor.
    const int64_t win = wstart / _win_samples;
    if (_cur_win < 0)
        _cur_win = win;

    // In [0, _rec_per_frame): wstart >= 0 is on the record grid and win is its floor.
    const int slot = (int)((wstart - win * _win_samples) / _rec_samples);

    // Row compaction, built once per window from its first record. A PRN gets a wire row
    // if it has replica energy, i.e. was despread; the record buffer has a slot for every
    // configured PRN whether or not it ran. The map is fixed for the window because readers
    // take each row's PRN from the first filled record slot and apply it to every slot, so
    // a PRN that starts being despread mid-window gets a row from the next window.
    if (!_map_built) {
        int next = 0;
        for (int p = 0; p < _n_prn; ++p) {
            if (in[(size_t)p * in_stride + gnss::REC_P_ENERGY] <= 0.0f)
                continue; // not despread this window: silence, not a row of zeros
            if (next >= _max_prn) {
                // More live PRNs than wire rows: ship the lowest slots and name the first
                // one dropped, since downstream a missing row looks like a lost satellite.
                // Logged every 200th window.
                if ((_prn_overflow++ % 200) == 0)
                    WARN("GnssTelemPack[{:s}]: more live PRNs than the {:d} wire rows -- "
                         "shipping the lowest-numbered and DROPPING the rest (PRN {:.0f} "
                         "onwards); raise telem_max_prn. {:d} windows so far",
                         _name, _max_prn, in[(size_t)p * in_stride + gnss::REC_PRN], _prn_overflow);
                break;
            }
            _row_of[(size_t)p] = next++;
        }
        _map_built = true;
    }

    for (int p = 0; p < _n_prn; ++p) {
        const int wrow = _row_of[(size_t)p];
        if (wrow < 0)
            continue;
        float* row =
            &_rows[gnss::telem_row_offset(slot, wrow, _max_prn, gnss::telem_row_floats(_n_chan))];
        std::memcpy(row, in + (size_t)p * in_stride, gnss::RECORD_FLOATS * sizeof(float));
        // Copied per column: the input and wire column offsets come from different layouts
        // (chan_offset, telem_chan_offset) that only happen to share a stride.
        for (int ch = 0; ch < _n_chan; ++ch)
            std::memcpy(row + gnss::telem_chan_offset(ch),
                        in + gnss::chan_offset(p, ch, _n_prn, _n_chan, _n_elem),
                        gnss::CHAN_FLOATS * sizeof(float));
    }
    _cur_present |= (1u << slot);
    if (slot == 0 && _n_prn > 0)
        _cur_utc0 = *reinterpret_cast<const double*>(in + gnss::RECORD_UTC_SLOT);
}

int64_t TelemChain::pack(uint8_t* frame) {
    gnss::TelemHeader h;
    std::memset(&h, 0, sizeof(h));
    h.magic = gnss::TELEM_MAGIC;
    h.version = gnss::TELEM_VERSION;
    h.n_rec = (uint16_t)_rec_per_frame;
    h.n_prn = (uint16_t)_max_prn;
    h.n_row = (uint16_t)gnss::RECORD_FLOATS;
    h.n_chan = (uint16_t)_n_chan;
    h.max_chan = (uint16_t)_n_chan;
    h.n_row_total = (uint16_t)gnss::telem_row_floats(_n_chan);
    for (int ch = 0; ch < _n_chan; ++ch)
        h.chan_id[ch] = (uint16_t)_chan_ids[(size_t)ch];
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

    _cur_win = -1;
    _cur_present = 0;
    _cur_utc0 = 0.0;
    std::fill(_rows.begin(), _rows.end(), 0.0f);
    // The row map belongs to the window that just went out; the next window builds its own.
    std::fill(_row_of.begin(), _row_of.end(), -1);
    _map_built = false;
    return h.wstart0;
}

} // namespace gnss
