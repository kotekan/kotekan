#ifndef GNSS_CUBE_WRITE_HPP
#define GNSS_CUBE_WRITE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssRecord.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gnss {

/// Version field of a packed beam-cube frame: the archive form of a CUBE_VERSION 3 frame. A wire
/// layout change means a new packed version too.
constexpr int64_t CUBE_PACKED_VERSION = 4;
static_assert(CUBE_VERSION == 3,
              "GnssCubeWrite packs CUBE_VERSION 3 frames; update it with the layout");

/// Bytes in a packed frame holding @c n_live PRN slots, @c n_bin bins and @c n_elem antennas.
constexpr size_t cube_packed_bytes(int n_live, int n_bin, int n_elem) {
    return CUBE_HEADER_BYTES + (size_t)n_bin * 2 * sizeof(int32_t) // freq_id lo/hi
           + (size_t)n_live * 4 * sizeof(int32_t)                  // prn, n_rec, n_reanchor, slot
           + (size_t)n_live * sizeof(double)                       // phi0
           + (size_t)n_live * n_bin * 2 * sizeof(float)            // w, energy
           + (size_t)n_live * n_bin * n_elem * 3 * sizeof(float);  // coh_re/im, incoh
}

/**
 * Pack one beam-cube frame (gnssRecord.hpp layout, CUBE_VERSION) into @c out: only the PRN slots
 * with records this window (n_rec > 0) and the frame's n_bin bins. Returns an empty string, or
 * why the frame was refused (an unknown version, or maxima that disagree with @c frame_size).
 */
std::string pack_cube_frame(const uint8_t* frame, size_t frame_size, std::vector<uint8_t>& out);

} // namespace gnss

/**
 * @class GnssCubeWrite
 * @brief Writes the beam-cube archive with the empty PRN slots and bins left out.
 *
 * Every sender pads its frames to the fleet's maxima (gnssRecord.hpp), because one
 * @c bufferRecv serves the whole fleet; on 2026-11-03 a frame used ~25 % of its bytes. This stage
 * takes the archiver's place of @c rawFileWrite and writes each frame packed:
 *
 *   the v3 header with version = CUBE_PACKED_VERSION, max_prn = the live slots, max_bins = n_bin
 *   (n_prn keeps the chain's slot count), then in order
 *     double  phi0[max_prn]
 *     int32   freq_id_lo[max_bins], freq_id_hi[max_bins]
 *     int32   prn[max_prn], n_rec[max_prn], n_reanchor[max_prn], slot[max_prn]
 *     float   w[max_prn][max_bins], energy[max_prn][max_bins]
 *     float   coh_re[max_prn][max_bins][n_elem], coh_im[...], incoh[...]
 *
 * where slot is each live slot's index in the sender's frame. A reader puts every slot back at
 * its index, zero elsewhere, and gets the v3 frame's values
 * (python/scripts/gnss/gnss_cube_read.py).
 *
 * Files are <base_dir>/<file_name>_NNNNNNN.raw, each frame stored as
 * `uint32 metadata_size | metadata | packed frame` as rawFileWrite stores it. Frames differ in
 * length, so a reader walks the headers. Numbering resumes past the highest existing number, so a
 * restart never overwrites the archive.
 *
 * @par buffers
 * @buffer in_buf  beam-cube frames as received (producer: bufferRecv)
 *
 * @conf base_dir             String. Directory the archive is written to.
 * @conf file_name            String. File name stem; must be one no other archive directory has
 *                            used (the compactor keys its manifest by basename).
 * @conf num_frames_per_file  Int, default 3600.
 */
class GnssCubeWrite : public kotekan::Stage {
public:
    GnssCubeWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    void main_thread() override;

private:
    Buffer* in_buf;
    std::string _base_dir;
    std::string _file_name;
    uint32_t _num_frames_per_file;
};

#endif // GNSS_CUBE_WRITE_HPP
