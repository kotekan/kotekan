#include "Config.hpp"          // for Config
#include "NDArray.hpp"         // for GenericNDArray
#include "asdfFiles.hpp"       // for beautify_buffer_name, chord2asdf, chord_metadata_version
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG, FATAL_ERROR, ERROR, INFO, WARN
#include "metadata.hpp"        // for metadataObject

#include "fmt.hpp"      // for compile_string_to_view, join
#include "gsl-lite.hpp" // for span

#include <DataType.hpp>            // for type_to_string, type_total_bytes
#include <N2FrameView.hpp>         // for N2FrameView
#include <N2Metadata.hpp>          // for N2Metadata, metadata_is_N2
#include <Stage.hpp>               // for Stage
#include <StageFactory.hpp>        // for REGISTER_KOTEKAN_STAGE
#include <algorithm>               // for max
#include <array>                   // for array
#include <asdf/asdf.hxx>           // for asdf
#include <asdf/byteorder.hxx>      // for host_byteorder
#include <asdf/config.hxx>         // for ASDF_CHECK_VERSION
#include <asdf/datatype.hxx>       // for datatype_t, complex64_t, float32_t, scalar_type_id_t
#include <asdf/entry.hxx>          // for int_entry, group, sequence, ndarray_entry, string_entry
#include <asdf/io.hxx>             // for block_format_t, compression_t
#include <asdf/memoized.hxx>       // for make_constant_memoized
#include <asdf/ndarray.hxx>        // for ndarray, ptr_block_t, block_info_t, block_t
#include <atomic>                  // for __atomic_base, atomic
#include <cassert>                 // for assert
#include <chordMetadata.hpp>       // for chordMetadata, metadata_is_chord, get_chord_metadata
#include <cstddef>                 // for ptrdiff_t, size_t
#include <cstdint>                 // for int64_t, uint8_t, uint32_t
#include <cstring>                 // for memcpy, strerror
#include <errno.h>                 // for errno, EEXIST, EISDIR
#include <errors.h>                // for exit_kotekan, ReturnCode
#include <functional>              // for function
#include <iomanip>                 // for operator<<, setfill, setw
#include <memory>                  // for shared_ptr, __shared_ptr_access, make_shared, allocator
#include <optional>                // for optional
#include <prometheusMetrics.hpp>   // for Metrics, Gauge
#include <sstream>                 // for basic_ostream, operator<<, basic_ostringstream, ostri...
#include <string>                  // for basic_string, char_traits, string, operator<<
#include <sys/stat.h>              // for mkdir
#include <unistd.h>                // for ssize_t, gethostname
#include <vector>                  // for vector
#include <visUtil.hpp>             // for current_time
#include <waitingForMaxFrames.hpp> // for waiting_for_max_frames

using namespace asdf;

/**
 * @class asdfFileWrite
 * @brief Stream a buffer to disk.
 *
 * @par Buffers:
 * @buffer in_buf Buffer to write to disk.
 *     @buffer_format Any
 *     @buffer_metadata Any
 *
 * @conf base_dir  String. Directory to write into.
 * @conf file_name String. Base filename to write.
 * @conf exit_after_n_frames  Int. Stop writing after this many frames, Default 0 = unlimited
 *       frames.
 * @conf exit_with_n_writers  Int. Exit after this many ASDF writers finished writing, Default 0 =
 *       unlimited writers.
 *
 * @par Metrics
 * @metric kotekan_asdffilewrite_write_time_seconds
 *         The write time to write out the last frame.
 *
 * @author Erik Schnetter
 **/
class asdfFileWrite : public kotekan::Stage {
    const std::string base_dir = config.get<std::string>(unique_name, "base_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);

    const int max_frames = config.get_default<int>(unique_name, "max_frames", -1);
    const bool skip_writing = config.get_default<bool>(unique_name, "skip_writing", false);

    Buffer* const buffer;

public:
    asdfFileWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("in_buf")) {
        ASDF_CHECK_VERSION();

        if (max_frames >= 0)
            ++waiting_for_max_frames;

        buffer->register_consumer(unique_name);
    }

    virtual ~asdfFileWrite() {}

    void main_thread() override {
        auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_asdffilewrite_write_time_seconds", unique_name);

        const double start_time = current_time();

        for (std::int64_t frame_counter = 0;; ++frame_counter) {
            const std::uint32_t frame_id = frame_counter % buffer->num_frames;

            if (stop_thread)
                break;

            DEBUG("Writing frame {:d} of {:d}.", frame_counter, max_frames);

            // Wait for the next frame
            DEBUG("wait_for_full_frame: frame_id={}", frame_id);
            const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, frame_id);
            if (!frame)
                break;
            DEBUG("got frame: frame_id={}", frame_id);

            // Start timer
            const double t0 = current_time();


            // Fetch metadata
            const std::shared_ptr<const metadataObject> mc = buffer->get_metadata(frame_id);
            if (!mc)
                FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                            buffer->buffer_name, frame_id);
            assert(mc);
            if (!metadata_is_chord(mc) && !metadata_is_N2(mc))
                FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD or N2",
                            buffer->buffer_name, frame_id);

            assert(metadata_is_chord(mc) || metadata_is_N2(mc));
            const std::shared_ptr<const chordMetadata> meta = get_chord_metadata(mc);
            const std::shared_ptr<const kotekan::GenericNDArray> frame_desc =
                buffer->get_frame_desc();
            assert(frame_desc);

            const double this_time = current_time();
            const double elapsed_time = this_time - start_time;

            INFO("Received buffer {} frame {} time sample {} (duration {} sec)", unique_name,
                 frame_counter, meta->get_sample0_offset(), elapsed_time);

            if (!skip_writing) {

                // Create ASDF project
                auto group = std::make_shared<ASDF::group>();

                // Create ASDF ndarray
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
                const ASDF::scalar_type_id_t type = chord2asdf(frame_desc->get_value_datatype());
                const std::size_t typesize = type_total_bytes(frame_desc->get_value_datatype());
#pragma GCC diagnostic pop

                const ssize_t ndims = ssize_t(frame_desc->get_rank());
                const auto extents = frame_desc->get_extents();
                std::vector<std::int64_t> dims(extents.begin(), extents.end());
                std::ptrdiff_t size = frame_desc->get_size();

                // "simple" strides have a layout that does not require copying the array
                bool strides_are_simple;
                {
                    strides_are_simple = true;
                    std::ptrdiff_t str = 1;
                    for (ssize_t d = ndims - 1; d >= 0; --d) {
                        if (str != frame_desc->get_extent(d))
                            ERROR("buffer {} d {} dim {} stride (found) {} str (expected) {}",
                                  unique_name, d, frame_desc->get_extent(d),
                                  frame_desc->get_stride(d), str);
                        strides_are_simple &= str == frame_desc->get_stride(d);
                        str *= frame_desc->get_extent(d);
                    }
                    strides_are_simple &= str == size;
                }

                std::vector<std::uint8_t> frame_copy;
                std::shared_ptr<ASDF::block_t> block;
                if (strides_are_simple) {
                    // Create a block that just points to the frame
                    block = std::make_shared<ASDF::ptr_block_t>(const_cast<std::uint8_t*>(frame),
                                                                size * typesize);
                } else {
                    // We need to copy the frame
                    frame_copy.resize(size * typesize);
                    if (!(frame_desc->get_stride(frame_desc->get_rank() - 1) == 1)) {
                        const auto extents = frame_desc->get_extents(); // cannot be formatted
                        ERROR("name={} type={} dims={} laststride={}", meta->name,
                              type_to_string(frame_desc->get_value_datatype()),
                              fmt::join(extents, ", "),
                              frame_desc->get_stride(frame_desc->get_rank() - 1));
                    }
                    if (!(frame_desc->get_stride(frame_desc->get_rank() - 1) == 1)
                        && frame_desc->get_rank() == 4)
                        for (int d = 0; d < 4; ++d)
                            ERROR("dim[{}]={} stride[{}]={}", d, frame_desc->get_extent(d), d,
                                  frame_desc->get_stride(d));
                    assert(frame_desc->get_stride(frame_desc->get_rank() - 1) == 1);
                    const std::uint8_t* __restrict__ const input_ptr = frame;
                    std::uint8_t* __restrict__ const output_ptr = frame_copy.data();
                    const std::ptrdiff_t lastdim =
                        frame_desc->get_extent(frame_desc->get_rank() - 1);
                    assert(frame_desc->get_rank() < 20);
                    std::ptrdiff_t output_stride[20];
                    for (ssize_t d = ssize_t(frame_desc->get_rank()) - 1; d >= 0; --d)
                        if (d == ssize_t(frame_desc->get_rank()) - 1)
                            output_stride[d] = 1;
                        else
                            output_stride[d] = output_stride[d + 1] * frame_desc->get_extent(d + 1);
                    assert(frame_desc->get_rank() <= 4);
                    switch (frame_desc->get_rank()) {
                        case 3:
                            for (std::ptrdiff_t j = 0; j < frame_desc->get_extent(0); ++j) {
                                for (std::ptrdiff_t k = 0; k < frame_desc->get_extent(1); ++k) {
                                    std::ptrdiff_t input_offset = j * frame_desc->get_stride(0)
                                                                  + k * frame_desc->get_stride(1);
                                    std::ptrdiff_t output_offset =
                                        j * output_stride[0] + k * output_stride[1];
                                    std::memcpy(output_ptr + typesize * output_offset,
                                                input_ptr + typesize * input_offset,
                                                typesize * lastdim);
                                }
                            }
                            break;
                        case 4:
                            for (std::ptrdiff_t j = 0; j < frame_desc->get_extent(0); ++j) {
                                for (std::ptrdiff_t k = 0; k < frame_desc->get_extent(1); ++k) {
                                    for (std::ptrdiff_t l = 0; l < frame_desc->get_extent(2); ++l) {
                                        std::ptrdiff_t input_offset =
                                            j * frame_desc->get_stride(0)
                                            + k * frame_desc->get_stride(1)
                                            + l * frame_desc->get_stride(2);
                                        std::ptrdiff_t output_offset = j * output_stride[0]
                                                                       + k * output_stride[1]
                                                                       + l * output_stride[2];
                                        std::memcpy(output_ptr + typesize * output_offset,
                                                    input_ptr + typesize * input_offset,
                                                    typesize * lastdim);
                                    }
                                }
                            }
                            break;
                        default:
                            FATAL_ERROR("frame_desc->get_rank()={}", frame_desc->get_rank());
                            assert(0);
                            break;
                    }
                    block = std::make_shared<ASDF::ptr_block_t>(frame_copy.data(), size * typesize);
                }

                // read segfault      const auto compression = ASDF::compression_t::blosc;
                // write error        const auto compression = ASDF::compression_t::blosc2;
                // too slow writing   const auto compression = ASDF::compression_t::bzip2;
                const auto compression = ASDF::compression_t::zlib;
                // working            const auto compression = ASDF::compression_t::none;
                const int compression_level = 9;

                if (metadata_is_chord(mc)) {
                    const auto ndarray = std::make_shared<ASDF::ndarray>(
                        ASDF::make_constant_memoized(block), std::optional<ASDF::block_info_t>(),
                        ASDF::block_format_t::block, compression, compression_level,
                        std::vector<bool>(), std::make_shared<ASDF::datatype_t>(type),
                        ASDF::host_byteorder(), dims);
                    const std::string ndarray_name = beautify_buffer_name(buffer->buffer_name);
                    group->emplace(ndarray_name, std::make_shared<ASDF::ndarray_entry>(ndarray));

                    // Describe metadata

                    if (meta->has_coarse_freq()) {
                        auto coarse_freq = std::make_shared<ASDF::sequence>();
                        const std::vector<int> I_coarse_freq = meta->get_coarse_freq();
                        for (int freq = 0; freq < meta->get_nfreq(); ++freq)
                            coarse_freq->push_back(
                                std::make_shared<ASDF::int_entry>(I_coarse_freq[freq]));
                        group->emplace("coarse_freq", coarse_freq);
                    }

                    if (meta->has_freq_upchan_factor()) {
                        auto freq_upchan_factor = std::make_shared<ASDF::sequence>();
                        const std::vector<int> I_freq_upchan_factor =
                            meta->get_freq_upchan_factor();
                        for (int freq = 0; freq < meta->get_nfreq(); ++freq)
                            freq_upchan_factor->push_back(
                                std::make_shared<ASDF::int_entry>(I_freq_upchan_factor[freq]));
                        group->emplace("freq_upchan_factor", freq_upchan_factor);
                    }

                    if (meta->has_fpga_seq_num())
                        group->emplace("fpga_seq_num",
                                       std::make_shared<ASDF::int_entry>(meta->get_fpga_seq_num()));

                    if (meta->has_sample0_offset())
                        group->emplace("sample0_offset", std::make_shared<ASDF::int_entry>(
                                                             meta->get_sample0_offset()));

                    if (meta->has_offset_downsampling())
                        group->emplace("offset_downsampling", std::make_shared<ASDF::int_entry>(
                                                                  meta->get_offset_downsampling()));

                    if (meta->has_half_fpga_sample0()) {
                        auto half_fpga_sample0 = std::make_shared<ASDF::sequence>();
                        const std::vector<int64_t> I_half_fpga_sample0 =
                            meta->get_half_fpga_sample0();
                        for (int freq = 0; freq < meta->get_nfreq(); ++freq)
                            half_fpga_sample0->push_back(
                                std::make_shared<ASDF::int_entry>(I_half_fpga_sample0[freq]));
                        group->emplace("half_fpga_sample0", half_fpga_sample0);
                    }

                    if (meta->has_time_downsampling_fpga())
                        group->emplace(
                            "time_downsampling_fpga",
                            std::make_shared<ASDF::int_entry>(meta->get_time_downsampling_fpga()));

                    auto chord_metadata_version_attribute = std::make_shared<ASDF::sequence>();
                    for (std::size_t n = 0; n < chord_metadata_version.size(); ++n)
                        chord_metadata_version_attribute->push_back(
                            std::make_shared<ASDF::int_entry>(chord_metadata_version.at(n)));
                    group->emplace("chord_metadata_version", chord_metadata_version_attribute);

                    auto name = std::make_shared<ASDF::string_entry>(meta->get_name());
                    group->emplace("name", name);

                    auto type_attribute =
                        std::make_shared<ASDF::string_entry>(type_to_string(meta->type));
                    group->emplace("type", type_attribute);

                    auto dim_names = std::make_shared<ASDF::sequence>();
                    for (ssize_t d = 0; d < ndims; ++d)
                        dim_names->push_back(
                            std::make_shared<ASDF::string_entry>(meta->get_dimension_name(d)));
                    group->emplace("dim_names", dim_names);

                    if (meta->ndishes >= 0)
                        group->emplace("ndishes", std::make_shared<ASDF::int_entry>(meta->ndishes));

                    if (meta->dish_index) {
                        auto dish_index = std::make_shared<ASDF::ndarray>(
                            std::vector<int>(meta->dish_index,
                                             meta->dish_index
                                                 + meta->n_dish_locations_ew
                                                       * meta->n_dish_locations_ns),
                            ASDF::block_format_t::inline_array, ASDF::compression_t::none, -1,
                            std::vector<bool>(),
                            std::vector<int64_t>{meta->n_dish_locations_ns,
                                                 meta->n_dish_locations_ew});
                        auto dish_index_entry = std::make_shared<ASDF::ndarray_entry>(dish_index);
                        group->emplace("dish_index", dish_index_entry);
                    }

                } else if (metadata_is_N2(mc)) {

                    const std::shared_ptr<const N2Metadata> meta =
                        std::static_pointer_cast<const N2Metadata>(mc);
                    N2FrameView frame_view(buffer, frame_id);

                    std::vector<ASDF::complex64_t> vis_view(frame_view.vis.begin(),
                                                            frame_view.vis.end());
                    auto vis_array = std::make_shared<ASDF::ndarray>(
                        vis_view, ASDF::block_format_t::inline_array, compression,
                        compression_level, std::vector<bool>(),
                        std::vector<int64_t>{meta->num_prod});
                    group->emplace("vis", vis_array);

                    std::vector<ASDF::float32_t> weights_view(frame_view.weight.begin(),
                                                              frame_view.weight.end());
                    auto weights_array = std::make_shared<ASDF::ndarray>(
                        weights_view, ASDF::block_format_t::inline_array, compression,
                        compression_level, std::vector<bool>(),
                        std::vector<int64_t>{meta->num_prod});
                    group->emplace("weights", weights_array);

                    std::vector<ASDF::float32_t> flags_view(frame_view.flags.begin(),
                                                            frame_view.flags.end());
                    auto flags_array = std::make_shared<ASDF::ndarray>(
                        flags_view, ASDF::block_format_t::inline_array, compression,
                        compression_level, std::vector<bool>(),
                        std::vector<int64_t>{meta->num_elements});
                    group->emplace("flags", flags_array);

                    std::vector<ASDF::float32_t> eval_view(frame_view.eval.begin(),
                                                           frame_view.eval.end());
                    auto eval_array = std::make_shared<ASDF::ndarray>(
                        eval_view, ASDF::block_format_t::inline_array, compression,
                        compression_level, std::vector<bool>(), std::vector<int64_t>{meta->num_ev});
                    group->emplace("eval", eval_array);

                    std::vector<ASDF::complex64_t> evec_view(frame_view.evec.begin(),
                                                             frame_view.evec.end());
                    auto evec_array = std::make_shared<ASDF::ndarray>(
                        evec_view, ASDF::block_format_t::inline_array, compression,
                        compression_level, std::vector<bool>(),
                        std::vector<int64_t>{meta->num_ev * meta->num_elements});
                    group->emplace("evec", evec_array);

                    group->emplace("emethod",
                                   std::make_shared<ASDF::int_entry>((int)frame_view.emethod));
                    group->emplace("erms", std::make_shared<ASDF::int_entry>((int)frame_view.erms));

                    std::vector<ASDF::complex64_t> gain_view(frame_view.gain.begin(),
                                                             frame_view.gain.end());
                    auto gain_array = std::make_shared<ASDF::ndarray>(
                        gain_view, ASDF::block_format_t::inline_array, compression,
                        compression_level, std::vector<bool>(),
                        std::vector<int64_t>{meta->num_elements});
                    group->emplace("gain", evec_array);

                    group->emplace("n_valid_fpga_ticks",
                                   std::make_shared<ASDF::int_entry>(meta->n_valid_fpga_ticks));
                    group->emplace("num_elements",
                                   std::make_shared<ASDF::int_entry>(meta->num_elements));
                    group->emplace("num_prod", std::make_shared<ASDF::int_entry>(meta->num_prod));
                    group->emplace("freq_id", std::make_shared<ASDF::int_entry>(meta->freq_id));
                }

                // Define file name
                std::ostringstream ibuf;
                ibuf << std::setw(8) << std::setfill('0') << frame_counter;
                const std::string iteration = ibuf.str();
                std::ostringstream buf;
                buf << base_dir << "/";
                if (prefix_hostname) {
                    char hostname[256];
                    gethostname(hostname, sizeof hostname);
                    buf << hostname << "_";
                }
                buf << file_name << "." << std::setw(8) << std::setfill('0') << frame_counter
                    << ".asdf";
                const std::string full_path = buf.str();

                // Write project to file
                const auto group0 = std::make_shared<ASDF::group>();
                group0->emplace(iteration, group);
                const auto project = ASDF::asdf({}, group0);
                int ierr = mkdir(base_dir.c_str(), 0777);
                if (ierr) {
                    if (errno != EEXIST && errno != EISDIR) {
                        const char* const msg = strerror(errno);
                        FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", base_dir.c_str(),
                                    msg);
                    }
                }
                project.write(full_path);
            } // if !skip_writing

            // Stop timer
            const double t1 = current_time();
            const double elapsed = t1 - t0;
            write_time_metric.set(elapsed);

            // Mark frame as done
            DEBUG("mark_frame_empty: frame_id={}", frame_id);
            buffer->mark_frame_empty(unique_name, frame_id);

            if (max_frames >= 0 && frame_counter + 1 >= max_frames) {
                INFO("Processed {} frames, exiting", frame_counter + 1);
                break;
            }
        } // for

        if (--waiting_for_max_frames == 0) {
            WARN("Shutting down Kotekan");
            exit_kotekan(CLEAN_EXIT);
        }

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(asdfFileWrite);
