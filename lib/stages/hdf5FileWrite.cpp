#include "CHORDTelescope.hpp"      // for CHORDTelescope
#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for DataType, type_to_string
#include "ICETelescope.hpp"        // for ICETelescope
#include "N2FrameView.hpp"         // for N2FrameView
#include "N2Metadata.hpp"          // for metadata_is_N2
#include "NDArray.hpp"             // for GenericNDArray
#include "Stage.hpp"               // for Stage
#include "StageFactory.hpp"        // for REGISTER_KOTEKAN_STAGE
#include "Symbol.hpp"              // for Symbol
#include "Telescope.hpp"           // for Telescope
#include "basebandGains.hpp"       // for BasebandGains
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata, metadata_is_chord, get_c...
#include "errors.h"                // for exit_kotekan, ReturnCode
#include "hdf5Files.hpp"           // for chord2hdf5, BITSHUFFLE_BLOCKSIZE_AUTO
#include "kotekanLogging.hpp"      // for DEBUG, FATAL_ERROR, WARN, INFO
#include "metadata.hpp"            // for metadataObject
#include "prometheusMetrics.hpp"   // for Metrics, Gauge
#include "timeUtil.hpp"            // for EOP
#include "visUtil.hpp"             // for current_time
#include "waitingForMaxFrames.hpp" // for waiting_for_max_frames

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>                             // for max, min
#include <array>                                 // for array
#include <atomic>                                // for __atomic_base, atomic
#include <cassert>                               // for assert
#include <cctype>                                // for isspace
#include <cstddef>                               // for size_t
#include <cstdint>                               // for int64_t, uint8_t, uint32_t
#include <errno.h>                               // for errno, EEXIST, EISDIR
#include <functional>                            // for function
#include <gsl-lite.hpp>                          // for span
#include <highfive/H5DataSet.hpp>                // for DataSet, AnnotateTraits::createAttribute
#include <highfive/H5DataSpace.hpp>              // for DataSpace, DataSpace::DataSpace, DataSp...
#include <highfive/H5DataType.hpp>               // for DataType
#include <highfive/H5File.hpp>                   // for File, NodeTraits::createDataSet, File::...
#include <highfive/H5Object.hpp>                 // for hsize_t, H5Z_FLAG_MANDATORY
#include <highfive/H5PropertyList.hpp>           // for PropertyType, RawPropertyList, Chunking
#include <highfive/H5Selection.hpp>              // for SliceTraits::write_raw, Selection, Slic...
#include <highfive/bits/H5PropertyList_misc.hpp> // for PropertyList::_initializeIfNeeded, Prop...
#include <highfive/bits/H5Selection_misc.hpp>    // for Selection::getSpace
#include <iomanip>                               // for operator<<, setfill, setw
#include <memory>                                // for allocator, shared_ptr, __shared_ptr_access
#include <sstream>                               // for basic_ostream, operator<<, basic_ostrin...
#include <stdexcept>                             // for invalid_argument
#include <string.h>                              // for strerror
#include <string>                                // for basic_string, char_traits, string, oper...
#include <sys/stat.h>                            // for mkdir
#include <vector>                                // for vector


using namespace hdf5;
using namespace HighFive;

/**
 * @class hdf5FileWrite
 * @brief Stream a buffer to disk.
 *
 * @par Buffers:
 * @buffer in_buf Buffer to write to disk.
 *     @buffer_format Any
 *     @buffer_metadata chord or N2
 *
 * @conf base_dir  String. Directory to write into.
 * @conf timestamp_subdir  Bool. Default false. Write into a
 *       `base_dir/<capture start time>` sub-directory, created if it does not exist.
 *       The name is the UTC capture start time, taken from the FPGA sequence number in
 *       the first frame's metadata, so it needs no configuring and matches the data.
 *       It is resolved once, on the first frame, so a run's files all land together.
 *       The time is rounded to the nearest second, because a capture scheduled on a
 *       whole second starts at the frame edge just before it.
 * @conf timestamp_subdir_format  String. Default "%Y%m%dT%H%M%SZ" (e.g.
 *       `20260131T184500Z`). strftime(3) format for the sub-directory name. Must not
 *       contain a path separator, and any whitespace it produces is stripped.
 * @conf file_name String. Base filename to write.
 * @conf prefix_hostname  Bool. Prepend hostname to output file names. Default:
 *       true.
 * @conf prefix_host_rank  Bool. Prepend rank to output file names. Default: false.
 * @conf frequency_pool_rank    Int. This stage's rank in the frequency pool.
 * @conf frequency_pool_size    Int. Number of stages in the frequency pool.
 * @conf max_frames  Int. Default: -1. Values <= 0 mean unlimited; N > 0 stops after N
 *       frames and shuts kotekan down.
 * @conf skip_writing  Bool. Do not actually write anything. Default:
 *       false.
 * @conf use_compression  Bool. Default: true. bitshuffle+zstd (filter 32008).
 * @conf create_single_file Bool. Default: false. Write all data to one single file,
 *       concatenating the frames along axis 0. Only supported for chord metadata. The
 *       metadata attributes are stored only once (from the first frame), so neither the
 *       per-frame metadata nor the frame boundary is on disk; see hdf5FileRead for how
 *       they are reconstructed when replaying. Because of that a single file must hold a
 *       contiguous, uniformly sampled stream, which is enforced here (fatal otherwise):
 *       - dimension 0 must be a time axis (its name starts with "T");
 *       - `dim_scaling[0]` must equal `time_downsampling_fpga` (1 if that is not set);
 *       - consecutive frames must advance `fpga_seq_num` by exactly
 *         `dim[0] * time_downsampling_fpga`, and must not change the frame shape;
 *       - frame decimation (@c write_x_frames / @c per_y_frames) must not be used.
 *       A writer whose buffer violates one of these has to use per-frame files
 *       (`create_single_file: false`), which have no such constraints.
 * @conf frames_per_file  Int. Append this many frames to each file before rolling
 *       over to a new one (CHORD metadata only). Default: 1 (one file per frame).
 *       Mutually exclusive with create_single_file. A file with more than one frame is
 *       concatenated like a single file, with the metadata of its first frame only, so each
 *       such file must meet the same constraints as create_single_file (enforced per file).
 *       A new file starts a new stream, so consecutive files need not be contiguous.
 * @conf write_x_frames  Int. Write the first X out of every Y frames (see per_y_frames).
 *       Default: -1 (disabled). Cannot be combined with @c create_single_file or
 *       @c frames_per_file > 1.
 * @conf per_y_frames  Int. Period Y for frame decimation (see write_x_frames).
 *       Default: -1 (disabled). Cannot be combined with @c create_single_file or
 *       @c frames_per_file > 1.
 * @conf baseband_gain_file, baseband_gain_host_info, baseband_gain_update_idx,
 *       crs_board_remap  The FPGA digital gains archive copied into every file, see
 *       kotekan::BasebandGains.
 *
 * Frames whose chordMetadata carries `dish_index` (as SubsetBaseband's do) are described
 * like hdf5N2Write's files: root attributes `input_list` and `input_order`, the /index_map
 * datasets (see hdf5::write_index_map), and the gains of just those inputs in
 * /digital_gains (see kotekan::BasebandGains::write_by_dish).
 *
 * @par Metrics
 * @metric kotekan_hdf5filewrite_write_time_seconds
 *         The write time to write out the last frame.
 *
 * @note The on-disk file format is documented in
 *       docs/sphinx/user/file_formats/hdf5_frames.rst. Any change to the
 *       datasets or attributes written by this stage should be reflected there.
 *
 * @author Erik Schnetter
 **/
class hdf5FileWrite : public kotekan::Stage {

    const std::string base_dir = config.get<std::string>(unique_name, "base_dir");
    const bool timestamp_subdir = config.get_default<bool>(unique_name, "timestamp_subdir", false);
    const std::string timestamp_subdir_format =
        config.get_default<std::string>(unique_name, "timestamp_subdir_format", "%Y%m%dT%H%M%SZ");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    const bool prefix_host_rank = config.get_default<bool>(unique_name, "prefix_host_rank", false);
    const int host_pool_rank = config.get_default<int>(unique_name, "frequency_pool_rank", 0);
    const int host_pool_size = config.get_default<int>(unique_name, "frequency_pool_size", 1);

    const int max_frames = config.get_default<int>(unique_name, "max_frames", -1);
    const bool use_compression = config.get_default<bool>(unique_name, "use_compression", true);
    const bool skip_writing = config.get_default<bool>(unique_name, "skip_writing", false);
    const bool create_single_file =
        config.get_default<bool>(unique_name, "create_single_file", false);
    const std::int64_t frames_per_file =
        config.get_default<std::int64_t>(unique_name, "frames_per_file", 1);
    /// Whether a file holds several frames concatenated along axis 0, with the metadata of its
    /// first frame only. Such a file must hold a contiguous stream; see the checks in write_chord.
    const bool concatenates_frames = create_single_file || frames_per_file > 1;
    /// The option that makes files hold several frames, and its per-frame-file setting, for
    /// the error messages.
    const char* const concat_option = create_single_file ? "create_single_file" : "frames_per_file";
    const char* const concat_off =
        create_single_file ? "create_single_file: false" : "frames_per_file: 1";
    const int64_t write_x_frames = config.get_default<int64_t>(unique_name, "write_x_frames", -1);
    const int64_t per_y_frames = config.get_default<int64_t>(unique_name, "per_y_frames", -1);

    const uint64_t num_polarizations = config.get<uint64_t>(unique_name, "num_polarizations");
    const uint64_t num_dishes = config.get<uint64_t>(unique_name, "num_dishes");

    Buffer* const buffer;

    /// The FPGA digital gains archive, copied into every output file as /digital_gains so
    /// that the recorded voltages can be interpreted without a separate lookup.
    kotekan::BasebandGains gains{config, unique_name, "hdf5FileWrite"};
    /// The archive is fetched at most once, when the first output file is created.
    bool gains_resolved = false;

    // The directory the files are actually written into. Equal to base_dir unless
    // timestamp_subdir is set, in which case resolve_output_dir() appends the capture
    // start time on the first frame.
    std::string output_dir = base_dir;
    bool output_dir_resolved = false;

    // The currently open output file (the single file, or a multi-frame file still
    // being filled) and the number of frames written to it so far.
    std::shared_ptr<File> current_file;
    std::int64_t frames_in_current_file = 0;

    // The state of the stream in the current multi-frame file: the shape and time downsampling
    // of its first frame, and the fpga_seq_num the next frame has to start at. Reset by every
    // new file, so consecutive files need not be contiguous. See the checks in write_chord.
    std::size_t stream_dim0 = 0;
    std::int64_t stream_tds = 0;
    bool stream_has_seq = false;
    std::int64_t stream_next_seq = 0;

    kotekan::prometheus::Gauge& write_time_metric =
        kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_hdf5filewrite_write_time_seconds", unique_name);

public:
    hdf5FileWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("in_buf")) {

        if (frames_per_file < 1)
            throw std::invalid_argument(fmt::format(
                fmt("hdf5FileWrite: frames_per_file ({:d}) must be >= 1"), frames_per_file));
        if (create_single_file && frames_per_file != 1)
            throw std::invalid_argument(
                "hdf5FileWrite: create_single_file and frames_per_file are mutually exclusive");
        if (concatenates_frames && write_x_frames >= 0 && per_y_frames > 0)
            FATAL_ERROR("hdf5FileWrite {:s}: write_x_frames/per_y_frames (frame decimation) cannot "
                        "be combined with create_single_file or frames_per_file > 1; a file "
                        "holding several frames must hold a contiguous stream",
                        unique_name);

        if (max_frames > 0)
            ++waiting_for_max_frames;

        buffer->register_consumer(unique_name);
    }

    virtual ~hdf5FileWrite() {}

    /**
     * @brief Work out which directory this run's files go in, from the first frame.
     *
     * With timestamp_subdir set the output goes in a per-capture sub-directory of
     * base_dir named after the UTC time of the first frame. Called for every frame but
     * only does anything on the first one, so that a run does not straddle directories
     * if it happens to cross a second boundary.
     *
     * @param fpga_seq_num The FPGA sequence number from the first frame's metadata.
     * @param seq_valid False if the frame carries no sequence number, in which case there
     *        is no capture time to name a directory after and base_dir is used as-is.
     */
    void resolve_output_dir(const std::int64_t fpga_seq_num, const bool seq_valid) {
        // Resolve on the very first frame, whatever it holds, so that a run cannot end up
        // split across two directories.
        if (output_dir_resolved)
            return;
        output_dir_resolved = true;

        if (!timestamp_subdir)
            return;

        if (!seq_valid) {
            WARN("timestamp_subdir is set, but the first frame of buffer \"{:s}\" carries no FPGA "
                 "sequence number, so there is no capture time to name a directory after. "
                 "Writing directly into \"{:s}\".",
                 buffer->buffer_name, base_dir);
            return;
        }

        // Round to the nearest second: a capture scheduled for a whole second starts at
        // the frame edge just before it, which would otherwise name the directory for the
        // previous second.
        const std::int64_t t_ns = Telescope::instance().to_time_ns((std::uint64_t)fpga_seq_num);
        const std::int64_t t_rounded_ns = ((t_ns + 500'000'000L) / 1'000'000'000L) * 1'000'000'000L;

        std::string dir_name = nanosec_i64_to_utc_string(t_rounded_ns, timestamp_subdir_format);

        // The name is a single path component, so drop anything that would break that.
        // Whitespace goes too, both because timestamps like "2026-01-31 18:45:00" are
        // awkward in paths and because the format is meant to avoid it in the first place.
        dir_name.erase(std::remove_if(dir_name.begin(), dir_name.end(),
                                      [](unsigned char c) { return std::isspace(c) || c == '/'; }),
                       dir_name.end());

        if (dir_name.empty())
            FATAL_ERROR("timestamp_subdir_format \"{:s}\" produced an empty directory name for "
                        "fpga seq {:d}",
                        timestamp_subdir_format, fpga_seq_num);

        output_dir = base_dir + "/" + dir_name;
        INFO("Writing this capture into \"{:s}\" (fpga seq {:d}, {:s})", output_dir, fpga_seq_num,
             nanosec_i64_to_utc_string(t_ns));
    }

    /// Create a directory, tolerating one that already exists.
    void make_directory(const std::string& path) const {
        if (mkdir(path.c_str(), 0777) && errno != EEXIST && errno != EISDIR)
            FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", path, strerror(errno));
    }

    // Create the HDF5 file.
    // Determine the file name, and create the containing directory if necessary.
    std::shared_ptr<File> create_file(const std::int64_t frame_counter) {
        // Define file name
        std::ostringstream buf;
        buf << hdf5::file_path_prefix(output_dir, file_name, prefix_hostname, prefix_host_rank,
                                      host_pool_rank);
        if (create_single_file) {
            // Do not include the frame counter when all output is written to a single file
            if (frame_counter >= 0)
                FATAL_ERROR("Internal error -- cannot handle a non-negative frame_counter when "
                            "creating a single_file HDF5 file with base name \"{:s}\"",
                            file_name);
        } else {
            if (frame_counter < 0)
                FATAL_ERROR("Internal error -- cannot handle a negative frame_counter when a "
                            "regular (non single_file) HDF5 file with base name \"{:s}\"",
                            file_name);
            buf << "." << std::setw(8) << std::setfill('0') << frame_counter;
        }
        buf << ".h5";
        const std::string full_path = buf.str();

        // Create the directories if necessary, base_dir first so that the per-capture
        // sub-directory can be made inside a base_dir that does not exist yet.
        make_directory(base_dir);
        if (output_dir != base_dir)
            make_directory(output_dir);

        // Create HDF5 file
        FileAccessProps fapl;
        // We need at least version 1.10 version for SWMR.
        // (That's the file format version, not the library version.)
        fapl.add(FileVersionBounds(H5F_LIBVER_V110, H5F_LIBVER_V110));
        HighFive::FileCreateProps fprops;
        // Use SWMR. This keeps HDF5 files readable even if they were not properly closed.
        // You may have to use `h5clear -s FILENAME.h5` to "tell" the file that the writer does not
        // exist any more.
        auto file =
            std::make_shared<File>(full_path, File::Truncate | File::WriteSWMR, fprops, fapl);

        // Fetch the gains archive once the capture directory exists, then copy it into this
        // file. The cache name carries the stage's unique name so that two writers sharing a
        // base_dir (one per NUMA domain, say) do not fetch onto each other.
        if (gains.configured()) {
            if (!gains_resolved) {
                gains_resolved = true;
                const std::string partial_dir = output_dir + "/.partial";
                make_directory(partial_dir);
                std::string stem = "baseband_gains" + unique_name;
                std::replace(stem.begin(), stem.end(), '/', '_');
                gains.resolve_local_copy(partial_dir, stem);
            }
            gains.copy_into(*file);
        }

        return file;
    }

    /**
     * @brief The array element held by each (polarization, dish) position of a frame.
     *
     * For a frame that names its dishes with `dish_index`, position (p, k) of its P and D axes
     * holds array element `p * num_dishes + dish_index[k]` in the telescope's fiducial order.
     * The positions are listed P-major, which is the element axis of /index_map. Empty unless
     * the frame carries `dish_index` and the telescope is a CHORDTelescope.
     */
    std::vector<std::uint64_t> element_rows(const chordMetadata& meta) const {
        const auto* chord = dynamic_cast<const CHORDTelescope*>(&Telescope::instance());
        if (!chord || !meta.has_dish_index())
            return {};

        int p_axis = -1, d_axis = -1;
        for (int i = 0; i < meta.dims; i++) {
            if (meta.get_dimension_name(i) == "P")
                p_axis = i;
            else if (meta.get_dimension_name(i) == "D")
                d_axis = i;
        }
        const std::vector<int> dish_index = meta.get_dish_index();
        if (p_axis < 0 || d_axis < p_axis || (std::size_t)meta.dim[d_axis] != dish_index.size()
            || (std::uint64_t)meta.dim[p_axis] > chord->get_num_polarizations())
            FATAL_ERROR("Buffer \"{:s}\": dish_index ({:d} dishes) does not describe a frame with "
                        "a P axis before its D axis and at most {:d} polarizations",
                        buffer->buffer_name, dish_index.size(), chord->get_num_polarizations());

        const std::uint64_t n_dish = chord->get_num_dishes();
        std::vector<std::uint64_t> rows;
        for (int p = 0; p < meta.dim[p_axis]; p++) {
            for (const int dish : dish_index) {
                if (dish < 0 || (std::uint64_t)dish >= n_dish)
                    FATAL_ERROR("Buffer \"{:s}\": dish_index entry {:d} is outside the "
                                "telescope's {:d} dishes",
                                buffer->buffer_name, dish, n_dish);
                rows.push_back(p * n_dish + dish);
            }
        }
        return rows;
    }

    /**
     * @brief Write the telescope metadata
     *
     * @param node   The HDF5 object (file, group, dataset, etc.) where the metadata should be
     *               attached
     */
    template<typename Node>
    void write_telescope_metadata(Node& node) {
        const auto& telescope = Telescope::instance();

        node.createAttribute("telescope_name", telescope.get_name());
        node.createAttribute("seq_length_nsec", telescope.seq_length_nsec());
        node.createAttribute("gps_time_enabled", telescope.gps_time_enabled());
        node.createAttribute("num_polarizations", num_polarizations);
        node.createAttribute("num_dishes", num_dishes);
        node.createAttribute("itrs_lat_deg", telescope.get_itrs_lat_deg());
        node.createAttribute("itrs_lon_deg", telescope.get_itrs_lon_deg());
        node.createAttribute("grid_orientation", telescope.get_grid_orientation());
        node.createAttribute("grid_size_x", telescope.get_grid_size_x());
        node.createAttribute("grid_size_y", telescope.get_grid_size_y());
        node.createAttribute("feed_separation_x_m", telescope.get_feed_separation_x_m());
        node.createAttribute("feed_separation_y_m", telescope.get_feed_separation_y_m());

        // Flatten to proper HDF5 multi-dimensional arrays
        const auto& dish_grid_indices =
            telescope.get_main_array_grid_indices(num_dishes, telescope.fiducial_element_order());
        node.createAttribute("dish_grid_indices", DataSpace({dish_grid_indices.size(), 2}),
                             create_datatype<std::int64_t>())
            .write_raw(reinterpret_cast<const std::int64_t*>(dish_grid_indices.data()));
        const auto& feed_positions_m =
            telescope.get_feed_positions_m(num_dishes, telescope.fiducial_element_order());
        node.createAttribute("feed_positions_m", DataSpace({feed_positions_m.size(), 3}),
                             create_datatype<double>())
            .write_raw(reinterpret_cast<const double*>(feed_positions_m.data()));
    }

    /**
     * @brief Create and write a file for a frame with chordMetadata
     *
     * @param full_path The full path, including file name, for the file.
     * @param frame uint8_t pointer to the frame data
     * @param meta shared pointer to the chordMetadata object for this frame.
     * @param frame_desc shared pointer to the NDarray description for this frame.
     */
    void write_chord(const std::uint8_t* const frame,
                     const std::shared_ptr<const chordMetadata>& meta,
                     const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc,
                     const std::int64_t frame_counter) {
        const auto& telescope = Telescope::instance();

        resolve_output_dir(meta->has_fpga_seq_num() ? meta->get_fpga_seq_num() : 0,
                           meta->has_fpga_seq_num());

        std::shared_ptr<File> fileptr;
        bool do_create_dataset = false; // Whether to create the dataset and write the attributes
        if (create_single_file) {
            // Create only a single file
            if (!current_file) {
                current_file = create_file(-1);
                do_create_dataset = true;
            }
        } else {
            // Start a new file when none is open. It is named after the first frame
            // it holds and closes once frames_per_file frames have been written.
            if (!current_file) {
                current_file = create_file(frame_counter);
                frames_in_current_file = 0;
                do_create_dataset = true;
            }
        }
        fileptr = current_file;
        auto& file = *fileptr;

        if (concatenates_frames) {
            // A multi-frame file concatenates frames along axis 0 and stores the metadata only
            // once, from its first frame, so a reader (hdf5FileRead, merge_baseband_freq.py) can
            // only reconstruct the per-frame fpga_seq_num if axis 0 is a contiguous, uniformly
            // sampled time axis. The stream state is reset by each new file.
            const std::string dim0_name = meta->get_dimension_name(0);
            if (!is_time_axis(dim0_name))
                FATAL_ERROR("Buffer \"{:s}\": {:s} requires a time axis as dimension 0, but "
                            "dimension 0 is \"{:s}\"; set {:s} for this writer",
                            buffer->buffer_name, concat_option, dim0_name, concat_off);
            const std::int64_t tds =
                meta->has_time_downsampling_fpga() ? meta->get_time_downsampling_fpga() : 1;
            if (meta->dim_scaling[0] != tds)
                FATAL_ERROR("Buffer \"{:s}\": {:s} requires dim_scaling[0] ({:d}) == "
                            "time_downsampling_fpga ({:d}); set {:s} for this writer",
                            buffer->buffer_name, concat_option, meta->dim_scaling[0], tds,
                            concat_off);
            if (do_create_dataset) {
                stream_dim0 = meta->dim[0];
                stream_tds = tds;
            } else {
                if (std::size_t(meta->dim[0]) != stream_dim0 || tds != stream_tds)
                    FATAL_ERROR("Buffer \"{:s}\": frame {:d} changes the frame shape/downsampling "
                                "(dim[0] {:d}->{:d}, time_downsampling_fpga {:d}->{:d}) within a "
                                "file",
                                buffer->buffer_name, frame_counter, stream_dim0, meta->dim[0],
                                stream_tds, tds);
                if (meta->has_fpga_seq_num() != stream_has_seq)
                    FATAL_ERROR("Buffer \"{:s}\": fpga_seq_num is present in some frames but not "
                                "in others; cannot write them to one file",
                                buffer->buffer_name);
                if (stream_has_seq && meta->get_fpga_seq_num() != stream_next_seq)
                    FATAL_ERROR("Buffer \"{:s}\": {:s} requires a contiguous stream, but frame "
                                "{:d} starts at fpga_seq_num {:d}, expected {:d}",
                                buffer->buffer_name, concat_option, frame_counter,
                                meta->get_fpga_seq_num(), stream_next_seq);
            }
            stream_has_seq = meta->has_fpga_seq_num();
            if (stream_has_seq)
                stream_next_seq = meta->get_fpga_seq_num() + std::int64_t(meta->dim[0]) * tds;
        }

        if (do_create_dataset) {

            // Choose dataspace
            const auto extents = frame_desc->get_extents();
            std::vector<std::size_t> initial_shape(extents.begin(), extents.end());
            std::vector<std::size_t> maximum_shape(extents.begin(), extents.end());
            initial_shape.at(0) = 0;
            maximum_shape.at(0) = DataSpace::UNLIMITED;
            const DataSpace space(initial_shape, maximum_shape);
            assert(meta->offset == 0);
            assert(space.getNumberDimensions() == frame_desc->get_rank());

            // Choose datatype
            const DataType type = chord2hdf5(frame_desc->get_value_datatype());

            RawPropertyList<PropertyType::DATASET_CREATE> props;

            // Enable chunking
            std::vector<hsize_t> chunk_dims(extents.begin(), extents.end());
            if (!chunk_dims.empty()) {
                // Choose chunk size
                std::size_t npoints_lo = 1;
                for (std::size_t d = 1; d < chunk_dims.size(); ++d)
                    npoints_lo *= chunk_dims.at(d);
                std::size_t npoints_hi = 8 * 1024 * 1024 / npoints_lo;
                using std::max, std::min;
                npoints_hi = min(std::size_t(chunk_dims.at(0)), max(std::size_t(1), npoints_hi));
                chunk_dims.at(0) = npoints_hi;
            }
            (*(DataSetCreateProps*)&props).add(Chunking(chunk_dims));

            if (use_compression) {
                // // Enable compression
                // constexpr int blosc_compression_level = 9;
                // const std::vector<unsigned int> blosc_flags{
                //     blosc_compression_level,
                //     BLOSC_SHUFFLE_BIT,
                //     BLOSC_COMPRESS_ZSTD,
                // };
                // props.add(H5Pset_filter, H5Z_BLOSC, H5Z_FLAG_MANDATORY, blosc_flags.size(),
                //           blosc_flags.data());
                constexpr int bitshuffle_compression_level = 9;
                const std::vector<unsigned int> bitshuffle_flags{
                    BITSHUFFLE_BLOCKSIZE_AUTO,
                    BITSHUFFLE_COMPRESS_ZSTD,
                    bitshuffle_compression_level,
                };
                props.add(H5Pset_filter, H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY,
                          bitshuffle_flags.size(), bitshuffle_flags.data());
            }

            // Create dataset
            auto dataset = file.createDataSet(file_name, space, type, props);

            // Write metadata (attributes)

            dataset.createAttribute("chord_metadata_version", chord_metadata_version);
            dataset.createAttribute("name", meta->get_name());
            dataset.createAttribute("type",
                                    kotekan::type_to_string(frame_desc->get_value_datatype()));
            const auto dimnames = frame_desc->get_dimnames();
            std::vector<std::string> dim_names(dimnames.begin(), dimnames.end());
            dataset.createAttribute("dim_names", dim_names);
            const auto dimscalings = frame_desc->get_dimscalings();
            std::vector<std::ptrdiff_t> dim_scalings(dimscalings.begin(), dimscalings.end());
            dataset.createAttribute("dim_scalings", dim_scalings);

            if (meta->has_fpga_seq_num()) {
                dataset.createAttribute("fpga_seq_num", meta->get_fpga_seq_num());
                dataset.createAttribute("fpga_seq_time_nsec",
                                        telescope.to_time_ns(meta->get_fpga_seq_num()));
            }

            if (meta->has_time_downsampling_fpga())
                dataset.createAttribute("time_downsampling_fpga",
                                        meta->get_time_downsampling_fpga());

            if (meta->has_coarse_freq())
                dataset.createAttribute("coarse_freq", meta->get_coarse_freq());

            if (meta->has_freq_upchan_factor())
                dataset.createAttribute("freq_upchan_factor", meta->get_freq_upchan_factor());

            if (meta->has_freq_upchan_index())
                dataset.createAttribute("freq_upchan_index", meta->get_freq_upchan_index());

            if (meta->has_rfi_frame_excision_enabled())
                dataset.createAttribute("rfi_frame_excision_enabled",
                                        meta->get_rfi_frame_excision_enabled());

            if (meta->has_rfi_frame_excision_thresholds())
                dataset.createAttribute("rfi_frame_excision_thresholds",
                                        meta->get_rfi_frame_excision_thresholds());

            write_telescope_metadata(dataset);

            // Alongside the /digital_gains group: a digest of the archive, on the dataset
            // where tools merging several writers' files look for their attributes.
            if (const auto gains_hash = gains.content_hash())
                dataset.createAttribute("digital_gains_hash", *gains_hash);

            // Describe the inputs the frame holds the way hdf5N2Write does, and add the gains of
            // just those inputs. Must happen before SWMR starts, which freezes the layout.
            const std::vector<std::uint64_t> rows = element_rows(*meta);
            if (!rows.empty()) {
                const auto& chord = Telescope::instance().cast<CHORDTelescope>();
                const ElementOrder order = chord.fiducial_element_order();
                file.createAttribute("input_order", ElementOrder_to_string(order));
                file.createAttribute("input_list",
                                     std::vector<std::int32_t>(rows.begin(), rows.end()));
                hdf5::write_index_map(file, chord, rows, order, 0);
                gains.write_by_dish(file, rows, chord.get_num_dishes());
            }

            if (create_single_file || frames_per_file > 1) {
                // Start SWMR mode on files that stay open across frames
                // (Can only do that after creating the dataset and writing all attributes.)
                file.flush(); // for good measure
                file.startSWMRWrite();
            }

        } // if do_create_dataset

        // Get datatype
        const DataType type = chord2hdf5(frame_desc->get_value_datatype());

        // Get the dataset
        auto dataset = file.getDataSet(file_name);

        // Extend the dataset
        const auto dataspace = dataset.getSpace();
        const auto old_dims = dataspace.getDimensions();
        std::vector<std::size_t> new_dims(old_dims);
        assert(meta->dims > 0);
        new_dims.at(0) = old_dims.at(0) + meta->dim[0];
        dataset.resize(new_dims);

        // Choose hpyerslab
        std::vector<std::size_t> offset(new_dims.size(), 0);
        std::vector<std::size_t> count(new_dims);
        offset.at(0) = old_dims.at(0);
        count.at(0) = meta->dim[0];
        auto hyperslab = dataset.select(offset, count);

        // Write data
        hyperslab.write_raw(frame, type);

        // Close a multi-frame file once its quota is reached (closing flushes it);
        // flush files that stay open so SWMR readers see the new frame.
        if (!create_single_file && ++frames_in_current_file >= frames_per_file) {
            current_file.reset();
        } else {
            // dataset.flush();
            file.flush();
        }
    }

    /**
     * @brief Create and write a file for an N2FrameView
     *
     * @param full_path The full path, including file name, for the file.
     * @param frame An N2FrameView object for the frame to output.
     */
    void write_n2(const N2FrameView& frame, const std::int64_t frame_counter) {
        DEBUG("Writing N2 frame seq: {:d}", frame.fpga_start_tick);

        if (frames_per_file != 1)
            FATAL_ERROR("frames_per_file is only supported for CHORD-metadata buffers");

        resolve_output_dir(frame.fpga_start_tick, true);

        std::shared_ptr<File> fileptr = create_file(frame_counter);
        auto& file = *fileptr;

        const std::vector<size_t> vis_dims({frame.num_prod, 2});
        const std::vector<size_t> weight_dims({frame.num_prod});
        const std::vector<size_t> flags_dims({frame.num_elements});
        const std::vector<size_t> eval_dims({frame.num_ev});
        const std::vector<size_t> evec_dims({frame.num_ev, frame.num_elements, 2});
        const std::vector<size_t> emethod_dims({1});
        const std::vector<size_t> erms_dims({1});
        const std::vector<size_t> gain_dims({frame.num_elements, 2});
        const std::vector<size_t> radiometer_chi2_dims({3}); // 3 pol pairs XX, XY, YY

        // Create dataspaces
        const DataSpace vis_space(vis_dims);
        const DataSpace weight_space(weight_dims);
        const DataSpace flags_space(flags_dims);
        const DataSpace eval_space(eval_dims);
        const DataSpace evec_space(evec_dims);
        const DataSpace emethod_space(emethod_dims);
        const DataSpace erms_space(erms_dims);
        const DataSpace gain_space(gain_dims);
        const DataSpace radiometer_chi2_space(radiometer_chi2_dims);

        // Create datatypes
        const DataType float_type = chord2hdf5(kotekan::float32);
        const DataType int_type = chord2hdf5(kotekan::int32);

        // Make props
        auto vis_props = make_chunked_props(vis_dims);
        auto weight_props = make_chunked_props(weight_dims);
        auto flags_props = make_chunked_props(flags_dims);
        auto eval_props = make_chunked_props(eval_dims);
        auto evec_props = make_chunked_props(evec_dims);
        auto emethod_props = make_chunked_props(emethod_dims);
        auto erms_props = make_chunked_props(erms_dims);
        auto gain_props = make_chunked_props(gain_dims);
        auto radiometer_chi2_props = make_chunked_props(radiometer_chi2_dims);

        // Create dataset
        auto vis_dset = file.createDataSet("vis", vis_space, float_type, vis_props);
        auto weight_dset = file.createDataSet("weight", weight_space, float_type, weight_props);
        auto flags_dset = file.createDataSet("flags", flags_space, float_type, flags_props);
        auto eval_dset = file.createDataSet("eval", eval_space, float_type, eval_props);
        auto evec_dset = file.createDataSet("evec", evec_space, float_type, evec_props);
        auto emethod_dset = file.createDataSet("emethod", emethod_space, int_type, emethod_props);
        auto erms_dset = file.createDataSet("erms", erms_space, float_type, erms_props);
        auto gain_dset = file.createDataSet("gain", gain_space, float_type, gain_props);
        auto radiometer_chi2_dset = file.createDataSet("radiometer_chi2", radiometer_chi2_space,
                                                       float_type, radiometer_chi2_props);

        vis_dset.write_raw(frame.vis.data(), float_type);
        weight_dset.write_raw(frame.weight.data(), float_type);
        flags_dset.write_raw(frame.flags.data(), float_type);
        eval_dset.write_raw(frame.eval.data(), float_type);
        evec_dset.write_raw(frame.evec.data(), float_type);
        emethod_dset.write_raw(&frame.emethod, int_type);
        erms_dset.write_raw(&frame.erms, float_type);
        gain_dset.write_raw(frame.gain.data(), float_type);
        radiometer_chi2_dset.write_raw(frame.radiometer_chi2.data(), float_type);

        // Set metadata as file-level attributes
        file.createAttribute("num_prod", frame.num_prod);
        file.createAttribute("num_ev", frame.num_ev);
        file.createAttribute("freq_id", frame.freq_id);
        file.createAttribute("freq_MHz", frame.freq_MHz);
        file.createAttribute("abs_time_idx", frame.abs_time_idx);
        file.createAttribute("time_center_eop.t_inst_ns", frame.time_center_eop.t_inst_ns);
        file.createAttribute("time_center_eop.t_ut1_ns", frame.time_center_eop.t_ut1_ns);
        file.createAttribute("time_center_eop.delta_UT1_inst",
                             frame.time_center_eop.delta_UT1_inst);
        file.createAttribute("time_center_eop.ERA_deg", frame.time_center_eop.ERA_deg);
        file.createAttribute("time_center_eop.xp_as", frame.time_center_eop.xp_as);
        file.createAttribute("time_center_eop.yp_as", frame.time_center_eop.yp_as);
        file.createAttribute("bin_eop.t_inst_ns", frame.bin_eop.t_inst_ns);
        file.createAttribute("bin_eop.t_ut1_ns", frame.bin_eop.t_ut1_ns);
        file.createAttribute("bin_eop.delta_UT1_inst", frame.bin_eop.delta_UT1_inst);
        file.createAttribute("bin_eop.ERA_deg", frame.bin_eop.ERA_deg);
        file.createAttribute("bin_eop.xp_as", frame.bin_eop.xp_as);
        file.createAttribute("bin_eop.yp_as", frame.bin_eop.yp_as);
        file.createAttribute("bin_start_ERA_deg", frame.bin_start_ERA_deg);
        file.createAttribute("bin_end_ERA_deg", frame.bin_end_ERA_deg);
        file.createAttribute("bin_start_ERAL_deg", frame.bin_start_ERAL_deg);
        file.createAttribute("bin_end_ERAL_deg", frame.bin_end_ERAL_deg);
        file.createAttribute("fpga_start_tick", frame.fpga_start_tick);
        file.createAttribute("frame_start_time_ns", frame.frame_start_time_ns);
        file.createAttribute("frame_length_fpga_ticks", frame.frame_length_fpga_ticks);
        file.createAttribute("n_valid_fpga_ticks", frame.n_valid_fpga_ticks);
        file.createAttribute("n_rfi_fpga_ticks", frame.n_rfi_fpga_ticks);
        file.createAttribute("n_rfi_only_fpga_ticks", frame.n_rfi_only_fpga_ticks);
        file.createAttribute("n_pl_fpga_ticks", frame.n_pl_fpga_ticks);
        file.createAttribute("rfi_frame_excision_enabled", frame.rfi_frame_excision_enabled);
        file.createAttribute("rfi_frame_excision_num", frame.rfi_frame_excision_num);
        file.createAttribute("rfi_frame_excision_threshold", frame.rfi_frame_excision_threshold);
        file.createAttribute("rfi_frame_excision_fraction", frame.rfi_frame_excision_fraction);

        write_telescope_metadata(file);
    }

    /**
     * @brief Build a property list for dataset creation that enables compression
     * and chunking.
     *
     * @param dims The dataset dimensions.
     * @return RawPropertyList<PropertyType::DATASET_CREATE> The property list.
     */
    RawPropertyList<PropertyType::DATASET_CREATE>
    make_chunked_props(const std::vector<size_t>& dims) {

        RawPropertyList<PropertyType::DATASET_CREATE> props;

        std::vector<hsize_t> chunk_dims;

        bool dims_nonzero = true;
        for (size_t d = 0; d < dims.size(); d++) {
            chunk_dims.push_back((hsize_t)dims[d]);
            if (dims[d] == 0)
                dims_nonzero = false;
        }

        if (dims_nonzero) {
            // Enable chunking
            if (!chunk_dims.empty()) {
                // Choose chunk size
                std::size_t npoints_lo = 1;
                for (std::size_t d = 1; d < chunk_dims.size(); ++d)
                    npoints_lo *= chunk_dims.at(d);
                std::size_t npoints_hi = 10 * 1000 * 1000 / npoints_lo;
                using std::max, std::min;
                npoints_hi = min(std::size_t(chunk_dims.at(0)), max(std::size_t(1), npoints_hi));
                chunk_dims.at(0) = npoints_hi;
            }
            (*(DataSetCreateProps*)&props).add(Chunking(chunk_dims));

            constexpr int bitshuffle_compression_level = 9;
            const std::vector<unsigned int> bitshuffle_flags{
                BITSHUFFLE_BLOCKSIZE_AUTO,
                BITSHUFFLE_COMPRESS_ZSTD,
                bitshuffle_compression_level,
            };
            props.add(H5Pset_filter, H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY, bitshuffle_flags.size(),
                      bitshuffle_flags.data());
        }

        return props;
    }

    /**
     * @brief The main thread function for hdf5FileWrite.
     *
     * This function is responsible for the main logic of the hdf5FileWrite class.
     */
    void main_thread() override {
        const double start_time = current_time();

        for (std::int64_t frame_counter = 0;; ++frame_counter) {
            const std::uint32_t frame_id = frame_counter % buffer->num_frames;

            if (stop_thread)
                break;

            // Wait for the next frame
            DEBUG("wait_for_full_frame: frame_id={}", frame_id);
            const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, frame_id);
            if (!frame)
                break;
            DEBUG("got frame: frame_id={}", frame_id);

            // Start timer
            const double t0 = current_time();

            const double this_time = current_time();
            const double elapsed_time = this_time - start_time;

            INFO("Received buffer {} frame {} (duration {} sec)", unique_name, frame_counter,
                 elapsed_time);

            // Optionally, only write every X out of Y frames.
            bool do_write = true;
            if (write_x_frames >= 0 && per_y_frames > 0) {
                if (frame_counter % per_y_frames > write_x_frames) {
                    do_write = false;
                }
            }

            if (!skip_writing && do_write) {
                // Fetch metadata
                const std::shared_ptr<const metadataObject> mc = buffer->get_metadata(frame_id);
                if (!mc)
                    FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                                buffer->buffer_name, frame_id);
                assert(mc);

                // Call one of the writers, depending if metadata is chord or N2
                if (metadata_is_chord(mc)) {
                    assert(metadata_is_chord(mc));
                    const std::shared_ptr<const chordMetadata> meta = get_chord_metadata(mc);
                    const std::shared_ptr<const kotekan::GenericNDArray> frame_desc =
                        buffer->get_frame_desc<kotekan::GenericNDArray>();
                    if (!frame_desc)
                        FATAL_ERROR(
                            "Buffer \"{:s}\" has no NDArray frame descriptor; hdf5FileWrite "
                            "needs one to write CHORD-metadata frames",
                            buffer->buffer_name);
                    if (frame_desc->get_byte_size() != buffer->frame_size)
                        FATAL_ERROR(
                            "Buffer \"{:s}\" has inconsistent size in NDArray "
                            "frame descriptor ({:d}) and buffer ({:d}); hdf5FileWrite needs "
                            "consistent frame sizes",
                            buffer->buffer_name, frame_desc->get_byte_size(), buffer->frame_size);
                    write_chord(frame, meta, frame_desc, frame_counter);
                } else if (metadata_is_N2(mc)) {
                    assert(metadata_is_N2(mc));
                    N2FrameView frame(buffer, frame_id);
                    write_n2(frame, frame_counter);
                } else {
                    FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD or N2",
                                buffer->buffer_name, frame_id);
                }
            } // if !skip_writing

            // Stop timer
            const double t1 = current_time();
            const double elapsed = t1 - t0;
            write_time_metric.set(elapsed);

            // Mark frame as done
            DEBUG("mark_frame_empty: frame_id={}", frame_id);
            buffer->mark_frame_empty(unique_name, frame_id);

            if (max_frames > 0 && frame_counter + 1 >= max_frames) {
                WARN("Processed {} frames", frame_counter + 1);
                break;
            }
        } // for

        // Close file if necessary
        current_file.reset();

        if (max_frames > 0) {
            // Unregister to allow the pipeline to continue, unless I'm the last
            // consumer on this buffer.
            buffer->unregister_consumer(unique_name, true);

            if (--waiting_for_max_frames == 0) {
                WARN("Shutting down Kotekan");
                exit_kotekan(CLEAN_EXIT);
            }
        }

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileWrite);
