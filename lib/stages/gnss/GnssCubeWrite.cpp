#include "GnssCubeWrite.hpp"

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "metadata.hpp"
#include "prometheusMetrics.hpp"
#include "visUtil.hpp" // for current_time

#include "fmt.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <unistd.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(GnssCubeWrite);

namespace gnss {

template<typename T>
static T peek(const uint8_t* p, size_t off) {
    T v;
    std::memcpy(&v, p + off, sizeof(T));
    return v;
}

// One item per live slot from a [max_prn] array.
static uint8_t* copy_slots(uint8_t* o, const uint8_t* src, const std::vector<int32_t>& live,
                           size_t item) {
    for (int32_t s : live) {
        std::memcpy(o, src + s * item, item);
        o += item;
    }
    return o;
}

// The first n_bin rows of each live slot's [max_bins][inner] float block.
static uint8_t* copy_slot_bins(uint8_t* o, const uint8_t* src, const std::vector<int32_t>& live,
                               size_t max_bins, size_t n_bin, size_t inner) {
    for (int32_t s : live) {
        std::memcpy(o, src + (size_t)s * max_bins * inner * sizeof(float),
                    n_bin * inner * sizeof(float));
        o += n_bin * inner * sizeof(float);
    }
    return o;
}

std::string pack_cube_frame(const uint8_t* frame, size_t frame_size, std::vector<uint8_t>& out) {
    if (frame_size < CUBE_HEADER_BYTES)
        return fmt::format("{:d} B is shorter than the cube header", frame_size);
    const int64_t version = peek<int64_t>(frame, 0);
    if (version != CUBE_VERSION)
        return fmt::format("version {:d}, this stage packs version {:d}", version, CUBE_VERSION);
    const int64_t n_prn = peek<int64_t>(frame, 40);
    const int64_t n_bin = peek<int64_t>(frame, 48);
    const int64_t n_elem = peek<int64_t>(frame, 56);
    const int32_t max_prn = peek<int32_t>(frame, 88);
    const int32_t max_bins = peek<int32_t>(frame, 92);
    if (max_prn <= 0 || max_bins <= 0 || n_elem <= 0 || n_prn < 0 || n_prn > max_prn || n_bin < 0
        || n_bin > max_bins || cube_frame_bytes(max_prn, max_bins, (int)n_elem) != frame_size)
        return fmt::format("n_prn {:d}/{:d}, n_bin {:d}/{:d}, n_elem {:d} do not describe a {:d} B "
                           "frame",
                           n_prn, max_prn, n_bin, max_bins, n_elem, frame_size);
    const size_t mp = max_prn, mb = max_bins, ne = n_elem, nb = n_bin;

    // The frame's arrays, in gnssRecord.hpp order.
    const uint8_t* p = frame + CUBE_HEADER_BYTES;
    const uint8_t* phi0 = p;
    const uint8_t* fid_lo = phi0 + mp * sizeof(double);
    const uint8_t* fid_hi = fid_lo + mb * sizeof(int32_t);
    const uint8_t* prn = fid_hi + mb * sizeof(int32_t);
    const uint8_t* n_rec = prn + mp * sizeof(int32_t);
    const uint8_t* n_re = n_rec + mp * sizeof(int32_t);
    const uint8_t* w = n_re + mp * sizeof(int32_t);
    const uint8_t* energy = w + mp * mb * sizeof(float);
    const uint8_t* coh_re = energy + mp * mb * sizeof(float);
    const uint8_t* coh_im = coh_re + mp * mb * ne * sizeof(float);
    const uint8_t* incoh = coh_im + mp * mb * ne * sizeof(float);

    std::vector<int32_t> live;
    for (int32_t s = 0; s < (int32_t)n_prn; s++)
        if (peek<int32_t>(n_rec, s * sizeof(int32_t)) > 0)
            live.push_back(s);
    const size_t nl = live.size();

    out.resize(cube_packed_bytes((int)nl, (int)nb, (int)ne));
    std::memcpy(out.data(), frame, CUBE_HEADER_BYTES);
    const int32_t nl32 = (int32_t)nl, nb32 = (int32_t)nb;
    std::memcpy(out.data(), &CUBE_PACKED_VERSION, sizeof(int64_t));
    std::memcpy(out.data() + 88, &nl32, sizeof(int32_t));
    std::memcpy(out.data() + 92, &nb32, sizeof(int32_t));
    uint8_t* o = out.data() + CUBE_HEADER_BYTES;

    o = copy_slots(o, phi0, live, sizeof(double));
    std::memcpy(o, fid_lo, nb * sizeof(int32_t));
    o += nb * sizeof(int32_t);
    std::memcpy(o, fid_hi, nb * sizeof(int32_t));
    o += nb * sizeof(int32_t);
    o = copy_slots(o, prn, live, sizeof(int32_t));
    o = copy_slots(o, n_rec, live, sizeof(int32_t));
    o = copy_slots(o, n_re, live, sizeof(int32_t));
    std::memcpy(o, live.data(), nl * sizeof(int32_t));
    o += nl * sizeof(int32_t);
    o = copy_slot_bins(o, w, live, mb, nb, 1);
    o = copy_slot_bins(o, energy, live, mb, nb, 1);
    o = copy_slot_bins(o, coh_re, live, mb, nb, ne);
    o = copy_slot_bins(o, coh_im, live, mb, nb, ne);
    copy_slot_bins(o, incoh, live, mb, nb, ne);
    return "";
}

} // namespace gnss

GnssCubeWrite::GnssCubeWrite(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssCubeWrite::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    _base_dir = config.get<std::string>(unique_name, "base_dir");
    _file_name = config.get<std::string>(unique_name, "file_name");
    _num_frames_per_file = config.get_default<uint32_t>(unique_name, "num_frames_per_file", 3600);
    if (_num_frames_per_file == 0)
        FATAL_ERROR("GnssCubeWrite[{:s}]: num_frames_per_file must be > 0", unique_name);
}

void GnssCubeWrite::main_thread() {
    // Resume past the highest existing number: a restart that numbered from 0 would overwrite the
    // archive's oldest files.
    const std::string prefix = _file_name + "_", suffix = ".raw";
    uint32_t file_num = 0;
    std::error_code ec;
    for (auto& ent : std::filesystem::directory_iterator(_base_dir, ec)) {
        std::string n = ent.path().filename().string();
        if (n.size() != prefix.size() + 7 + suffix.size() || n.compare(0, prefix.size(), prefix)
            || n.compare(n.size() - suffix.size(), suffix.size(), suffix))
            continue;
        std::string digits = n.substr(prefix.size(), 7);
        if (digits.find_first_not_of("0123456789") == std::string::npos)
            file_num = std::max(file_num, (uint32_t)std::stoul(digits) + 1);
    }
    if (ec)
        FATAL_ERROR("GnssCubeWrite[{:s}]: cannot list {:s}: {:s}", unique_name, _base_dir,
                    ec.message());
    INFO("GnssCubeWrite[{:s}]: writing {:s}/{:s}NNNNNNN{:s} from {:07d}", unique_name, _base_dir,
         prefix, suffix, file_num);

    auto& write_time =
        Metrics::instance().add_gauge("kotekan_gnss_cube_write_time_seconds", unique_name);
    auto& bytes_in =
        Metrics::instance().add_counter("kotekan_gnss_cube_write_frame_bytes_total", unique_name);
    auto& bytes_out =
        Metrics::instance().add_counter("kotekan_gnss_cube_write_packed_bytes_total", unique_name);

    int fd = -1;
    uint32_t frames_in_file = 0;
    std::string path;
    std::vector<uint8_t> packed, rec;
    frameID frame_id(in_buf);
    while (!stop_thread) {
        uint8_t* frame = in_buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;
        double t0 = current_time();

        std::string err = gnss::pack_cube_frame(frame, in_buf->frame_size, packed);
        if (!err.empty())
            FATAL_ERROR("GnssCubeWrite[{:s}]: frame {:d} refused: {:s}", unique_name, (int)frame_id,
                        err);

        uint32_t metadata_size = 0;
        std::shared_ptr<metadataObject> mc = in_buf->get_metadata(frame_id);
        if (mc)
            metadata_size = mc->get_serialized_size();
        rec.resize(sizeof(metadata_size) + metadata_size + packed.size());
        std::memcpy(rec.data(), &metadata_size, sizeof(metadata_size));
        if (mc)
            mc->serialize((char*)rec.data() + sizeof(metadata_size));
        std::memcpy(rec.data() + sizeof(metadata_size) + metadata_size, packed.data(),
                    packed.size());

        if (fd == -1) {
            path = fmt::format("{:s}/{:s}{:07d}{:s}", _base_dir, prefix, file_num, suffix);
            fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
            if (fd == -1)
                FATAL_ERROR("GnssCubeWrite[{:s}]: cannot open {:s}: {:s}", unique_name, path,
                            std::strerror(errno));
        }
        if (write(fd, rec.data(), rec.size()) != (ssize_t)rec.size())
            FATAL_ERROR("GnssCubeWrite[{:s}]: short write to {:s}: {:s}", unique_name, path,
                        std::strerror(errno));
        bytes_in.inc(in_buf->frame_size);
        bytes_out.inc(packed.size());

        if (++frames_in_file == _num_frames_per_file) {
            if (close(fd) == -1)
                ERROR("GnssCubeWrite[{:s}]: cannot close {:s}: {:s}", unique_name, path,
                      std::strerror(errno));
            DEBUG("GnssCubeWrite[{:s}]: closed {:s}", unique_name, path);
            fd = -1;
            frames_in_file = 0;
            file_num++;
        }
        write_time.set(current_time() - t0);
        in_buf->mark_frame_empty(unique_name, frame_id++);
    }
    if (fd != -1)
        close(fd);
}
