/**
 * @file
 * @brief The record buffer of a GPU half: each frame one record of every row, after a directory
 *        of the rows.
 */
#ifndef GNSS_REC_ROWS_HPP
#define GNSS_REC_ROWS_HPP

#include "buffer.hpp" // for Buffer

#include "json.hpp" // for json

#include <cstddef> // for size_t
#include <cstdint> // for int64_t, uint8_t
#include <string>  // for string
#include <vector>  // for vector

namespace gnss {

/// A row's record block in each frame of a GPU half's record buffer, from its rec_row entry.
struct RecRow {
    int64_t chain_id = 0;
    int64_t n_prn = 0;
    /// Comb columns per PRN in the block (gnssRecord.hpp), 0 for a record without them.
    int64_t n_chan = 0;
    int64_t offset = 0; ///< in bytes from the frame's start
    int64_t bytes = 0;
};

/// A row from its rec_row entry; every key is required.
RecRow rec_row(const nlohmann::json& j);

/// FATAL, naming @p who, unless rec_row @p r holds a record of @p n_prn PRNs of @p n_elem
/// elements with @p n_chan comb columns, and nothing more.
void check_rec_row(const std::string& who, const RecRow& r, int n_prn, int n_elem, int n_chan);

/// The first word of each frame of a record buffer.
constexpr int64_t REC_DIR_MAGIC = 0x52454344; // "RECD"
/// Each row's block starts on a multiple of this, so no two rows' writers share a cache line.
constexpr size_t REC_ROW_ALIGN = 64;

/**
 * @brief The directory at the head of each frame of a GPU half's record buffer.
 *
 *     int64  REC_DIR_MAGIC, n
 *     RecRow rows[n]      in offset order
 *     int64  wstart[n]    the first sample of each row's record in this frame
 *
 * Each row's record block follows at its offset. The rows tile the frame in offset order: the
 * first starts at the end of the directory, every other at the end of the row before it, each
 * rounded up to REC_ROW_ALIGN. A row's writer writes its entry, wstart and block, and the
 * writer of row 0 also the head, so the rows of a frame may be written by different threads.
 */
class RecDir {
public:
    /// The directory of rows @p rows of record buffer @p buf, in offset order. FATAL, naming
    /// @p who, unless they tile its frame and their chain ids are unique.
    RecDir(const std::string& who, std::vector<RecRow> rows, const Buffer* buf);

    /// Bytes of the directory of @p n rows.
    static size_t bytes(size_t n);

    size_t size() const {
        return _rows.size();
    }
    const RecRow& row(size_t i) const {
        return _rows[i];
    }
    /// The position of the row at byte @p offset, which is one of them.
    size_t find(int64_t offset) const;

    /// Write row @p i's entry and @p wstart into @p frame, and for row 0 the head.
    void write(uint8_t* frame, size_t i, int64_t wstart) const;
    /// FATAL, naming @p who and the first field that differs, unless @p frame, frame @p id of
    /// the buffer, carries this directory.
    void check(const std::string& who, int id, const uint8_t* frame) const;
    /// Row @p i's wstart in @p frame.
    int64_t wstart(const uint8_t* frame, size_t i) const;

private:
    std::string _buf;
    size_t _frame_size;
    std::vector<RecRow> _rows;
    /// The head and the rows' entries, as each frame carries them.
    std::vector<uint8_t> _image;
};

} // namespace gnss

#endif // GNSS_REC_ROWS_HPP
