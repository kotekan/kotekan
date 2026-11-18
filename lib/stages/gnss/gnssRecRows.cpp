#include "gnssRecRows.hpp"

#include "gnssRecord.hpp"     // for frame_floats
#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO

#include <algorithm> // for sort, clamp
#include <cstring>   // for memcpy, memcmp
#include <set>       // for set
#include <utility>   // for move, pair

namespace gnss {

namespace {

constexpr size_t HEAD_BYTES = 2 * sizeof(int64_t);
static_assert(sizeof(RecRow) == 5 * sizeof(int64_t), "a directory entry is five int64s");

const std::pair<const char*, int64_t RecRow::*> FIELDS[] = {{"chain_id", &RecRow::chain_id},
                                                            {"n_prn", &RecRow::n_prn},
                                                            {"n_chan", &RecRow::n_chan},
                                                            {"offset", &RecRow::offset},
                                                            {"bytes", &RecRow::bytes}};

size_t align_up(size_t b) {
    return (b + REC_ROW_ALIGN - 1) / REC_ROW_ALIGN * REC_ROW_ALIGN;
}

std::string chain_ids(const std::vector<RecRow>& rows) {
    std::string s;
    for (const RecRow& r : rows)
        s += (s.empty() ? "" : ", ") + std::to_string(r.chain_id);
    return s;
}

} // namespace

RecRow rec_row(const nlohmann::json& j) {
    RecRow r;
    r.chain_id = j.at("chain_id").get<int64_t>();
    r.n_prn = j.at("n_prn").get<int64_t>();
    r.n_chan = j.at("n_chan").get<int64_t>();
    r.offset = j.at("offset").get<int64_t>();
    r.bytes = j.at("bytes").get<int64_t>();
    return r;
}

void check_rec_row(const std::string& who, const RecRow& r, int n_prn, int n_elem, int n_chan) {
    const size_t bytes = frame_floats(n_prn, n_elem, n_chan) * sizeof(float);
    if (r.n_prn != n_prn || r.n_chan != n_chan || r.bytes != (int64_t)bytes)
        FATAL_ERROR_NON_OO("{:s}: rec_row holds {:d} PRN x {:d} comb columns in {:d} B, this row's "
                           "record is {:d} PRN of {:d} elements x {:d} comb columns in {:d} B",
                           who, r.n_prn, r.n_chan, r.bytes, n_prn, n_elem, n_chan, bytes);
}

size_t RecDir::bytes(size_t n) {
    return HEAD_BYTES + n * (sizeof(RecRow) + sizeof(int64_t));
}

RecDir::RecDir(const std::string& who, std::vector<RecRow> rows, const Buffer* buf) :
    _buf(buf->buffer_name), _frame_size(buf->frame_size), _rows(std::move(rows)) {
    if (_rows.empty())
        FATAL_ERROR_NON_OO("{:s}: no rows in {:s}", who, _buf);
    std::sort(_rows.begin(), _rows.end(),
              [](const RecRow& a, const RecRow& b) { return a.offset < b.offset; });
    std::set<int64_t> ids;
    size_t at = align_up(bytes(_rows.size()));
    for (const RecRow& r : _rows) {
        if (r.offset != (int64_t)at || r.bytes < 1)
            FATAL_ERROR_NON_OO("{:s}: the rec_row of chain_id {:d} holds bytes [{:d}, {:d}) of "
                               "{:s}; the directory and the rows before it put it at byte {:d}",
                               who, r.chain_id, r.offset, r.offset + r.bytes, _buf, at);
        if (!ids.insert(r.chain_id).second)
            FATAL_ERROR_NON_OO("{:s}: chain_id {:d} is in two rec_rows of {:s}", who, r.chain_id,
                               _buf);
        at = align_up((size_t)(r.offset + r.bytes));
    }
    const RecRow& last = _rows.back();
    if ((size_t)(last.offset + last.bytes) > _frame_size)
        FATAL_ERROR_NON_OO("{:s}: the rec_rows end at byte {:d}, past the {:d} B frame of {:s}",
                           who, last.offset + last.bytes, _frame_size, _buf);
    const int64_t head[2] = {REC_DIR_MAGIC, (int64_t)_rows.size()};
    _image.resize(HEAD_BYTES + _rows.size() * sizeof(RecRow));
    std::memcpy(_image.data(), head, HEAD_BYTES);
    std::memcpy(_image.data() + HEAD_BYTES, _rows.data(), _rows.size() * sizeof(RecRow));
}

size_t RecDir::find(int64_t offset) const {
    for (size_t i = 0; i < _rows.size(); ++i)
        if (_rows[i].offset == offset)
            return i;
    FATAL_ERROR_NON_OO("no rec_row of {:s} starts at byte {:d}", _buf, offset);
}

void RecDir::write(uint8_t* frame, size_t i, int64_t wstart) const {
    if (i == 0)
        std::memcpy(frame, _image.data(), HEAD_BYTES);
    std::memcpy(frame + HEAD_BYTES + i * sizeof(RecRow), &_rows[i], sizeof(RecRow));
    std::memcpy(frame + _image.size() + i * sizeof(int64_t), &wstart, sizeof(wstart));
}

int64_t RecDir::wstart(const uint8_t* frame, size_t i) const {
    int64_t w;
    std::memcpy(&w, frame + _image.size() + i * sizeof(int64_t), sizeof(w));
    return w;
}

void RecDir::check(const std::string& who, int id, const uint8_t* frame) const {
    if (std::memcmp(frame, _image.data(), _image.size()) == 0)
        return;
    int64_t head[2];
    std::memcpy(head, frame, HEAD_BYTES);
    if (head[0] != REC_DIR_MAGIC)
        FATAL_ERROR_NON_OO("{:s}: frame {:d} of {:s} starts with {:#x}, not a row directory", who,
                           id, _buf, head[0]);
    if (head[1] != (int64_t)_rows.size()) {
        // The rows it lists, as many as the frame can hold.
        const int64_t fit = (int64_t)((_frame_size - HEAD_BYTES) / sizeof(RecRow));
        std::vector<RecRow> listed((size_t)std::clamp<int64_t>(head[1], 0, fit));
        std::memcpy(listed.data(), frame + HEAD_BYTES, listed.size() * sizeof(RecRow));
        FATAL_ERROR_NON_OO("{:s}: frame {:d} of {:s} lists {:d} rows (chain_id {:s}), this host "
                           "reads {:d} (chain_id {:s})",
                           who, id, _buf, head[1], chain_ids(listed), _rows.size(),
                           chain_ids(_rows));
    }
    for (size_t i = 0; i < _rows.size(); ++i) {
        RecRow got;
        std::memcpy(&got, frame + HEAD_BYTES + i * sizeof(RecRow), sizeof(RecRow));
        for (const auto& [field, member] : FIELDS)
            if (got.*member != _rows[i].*member)
                FATAL_ERROR_NON_OO("{:s}: frame {:d} of {:s}: directory row {:d} has {:s} {:d}, "
                                   "this host's row {:d} (chain_id {:d}) has {:d}",
                                   who, id, _buf, i, field, got.*member, i, _rows[i].chain_id,
                                   _rows[i].*member);
    }
}

} // namespace gnss
