#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>

#include <chordMetadata.hpp>
#include <N2Metadata.hpp>
#include <N2FrameView.hpp>

#include "json.hpp"
#include "xtensor/xarray.hpp"

// factory functions to create files, groups and datasets
#include "z5/factory.hxx"
// handles for z5 filesystem objects
#include "z5/filesystem/handle.hxx"
// io for xtensor multi-arrays
#include "z5/multiarray/xtensor_access.hxx"
// attribute functionality
#include "z5/attributes.hxx"

#include <cstdint>
#include <errno.h>
#include <errors.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <prometheusMetrics.hpp>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include <visUtil.hpp>  // for current_time

/**
 * @class zarrFileWrite
 * @brief Stream a buffer to disk.
 *
 * @par Buffers:
 * @buffer in_buf Buffer to write to disk.
 *     @buffer_format Any
 *     @buffer_metadata Any
 *
 * @par Metrics
 * @metric 
 **/
class zarrFileWrite : public kotekan::Stage {
    const std::string base_dir = config.get<std::string>(unique_name, "base_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);

    const int max_frames = config.get_default<int>(unique_name, "max_frames", -1);
    const bool skip_writing = config.get_default<bool>(unique_name, "skip_writing", false);

    Buffer* const buffer;

public:
    zarrFileWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("in_buf")) {

        buffer->register_consumer(unique_name);
    }

    virtual ~zarrFileWrite() {}

    void main_thread() override {
        auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_zarrfilewrite_write_time_seconds", unique_name);

        for (std::int64_t frame_counter = 0;; ++frame_counter) {
            const std::uint32_t frame_id = frame_counter % buffer->num_frames;

            if (max_frames >= 0 && frame_counter >= max_frames) {
                INFO("Processed {} frames, shutting down Kotekan", frame_counter);
                exit_kotekan(CLEAN_EXIT);
            }

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
            const std::shared_ptr<metadataObject> mc = buffer->get_metadata(frame_id);
            if (!mc)
                FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                            buffer->buffer_name, frame_id);
            assert(mc);
            if (!metadata_is_chord(mc) && !metadata_is_N2(mc))
                FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD or N2",
                            buffer->buffer_name, frame_id);
            assert(metadata_is_chord(mc) || metadata_is_N2(mc));
            const std::shared_ptr<chordMetadata> meta = get_chord_metadata(mc);

            if (!skip_writing) {

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
                    << ".zarr";
                const std::string full_path = buf.str();

                // get handle to a File on the filesystem
                z5::filesystem::handle::File zfile(full_path);

                // create the file in zarr format
                const bool createAsZarr = true;
                z5::createFile(zfile, createAsZarr);


//-----------
                if(metadata_is_chord(mc)) {
                    // zarr output ... 
                } else if(metadata_is_N2(mc)) {
                    
                    const std::shared_ptr<N2Metadata> meta = std::static_pointer_cast<N2Metadata>(mc);
                    N2FrameView frame_view (buffer, frame_id);

                    // create a new zarr dataset
                    const std::string dsName = "data";
                    std::vector<size_t> shape = { 1000, 1000, 1000 };
                    std::vector<size_t> chunks = { 100, 100, 100 };
                    auto ds = z5::createDataset(zfile, dsName, "float32", shape, chunks);

                    // write array to roi
                    z5::types::ShapeType offset1 = { 50, 100, 150 };
                    xt::xarray<float>::shape_type shape1 = { 150, 200, 100 };
                    xt::xarray<float> array1(shape1, 42.0);
                    z5::multiarray::writeSubarray<float>(ds, array1, offset1.begin());


                    // std::vector<ASDF::complex64_t> vis_view( frame_view.vis.begin(), frame_view.vis.end() );
                    // auto vis_array = std::make_shared<ASDF::ndarray>(
                    //     vis_view, ASDF::block_format_t::inline_array, compression, compression_level,
                    //     std::vector<bool>(), std::vector<int64_t>{meta->num_prod});
                    // group->emplace("vis", vis_array);

                    // std::vector<ASDF::float32_t> weights_view( frame_view.weight.begin(), frame_view.weight.end() );
                    // auto weights_array = std::make_shared<ASDF::ndarray>(
                    //     weights_view, ASDF::block_format_t::inline_array, compression, compression_level,
                    //     std::vector<bool>(), std::vector<int64_t>{meta->num_prod});
                    // group->emplace("weights", weights_array);

                    // std::vector<ASDF::float32_t> flags_view( frame_view.flags.begin(), frame_view.flags.end() );
                    // auto flags_array = std::make_shared<ASDF::ndarray>(
                    //     flags_view, ASDF::block_format_t::inline_array, compression, compression_level,
                    //     std::vector<bool>(), std::vector<int64_t>{meta->num_elements});
                    // group->emplace("flags", flags_array);

                    // std::vector<ASDF::float32_t> eval_view( frame_view.eval.begin(), frame_view.eval.end() );
                    // auto eval_array = std::make_shared<ASDF::ndarray>(
                    //     eval_view, ASDF::block_format_t::inline_array, compression, compression_level,
                    //     std::vector<bool>(), std::vector<int64_t>{meta->num_ev});
                    // group->emplace("eval", eval_array);

                    // std::vector<ASDF::complex64_t> evec_view( frame_view.evec.begin(), frame_view.evec.end() );
                    // auto evec_array = std::make_shared<ASDF::ndarray>(
                    //     evec_view, ASDF::block_format_t::inline_array, compression, compression_level,
                    //     std::vector<bool>(), std::vector<int64_t>{meta->num_ev * meta->num_elements});
                    // group->emplace("evec", evec_array);

                    // group->emplace("emethod", std::make_shared<ASDF::int_entry>((int) frame_view.emethod));
                    // group->emplace("erms", std::make_shared<ASDF::int_entry>((int) frame_view.erms));

                    // std::vector<ASDF::complex64_t> gain_view( frame_view.gain.begin(), frame_view.gain.end() );
                    // auto gain_array = std::make_shared<ASDF::ndarray>(
                    //     gain_view, ASDF::block_format_t::inline_array, compression, compression_level,
                    //     std::vector<bool>(), std::vector<int64_t>{meta->num_elements});
                    // group->emplace("gain", evec_array);

                    // group->emplace("n_valid_fpga_ticks_in_frame", std::make_shared<ASDF::int_entry>(meta->n_valid_fpga_ticks_in_frame));
                    // group->emplace("num_elements", std::make_shared<ASDF::int_entry>(meta->num_elements));
                    // group->emplace("num_prod", std::make_shared<ASDF::int_entry>(meta->num_prod));
                    // group->emplace("freq_id", std::make_shared<ASDF::int_entry>(meta->freq_id));
                }

                // Write project to file
                int ierr = mkdir(base_dir.c_str(), 0777);
                if (ierr) {
                    if (errno != EEXIST && errno != EISDIR) {
                        const char* const msg = strerror(errno);
                        FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", base_dir.c_str(),
                                    msg);
                    }
                }
                // ...
//----------------

            } // if !skip_writing

            // Stop timer
            const double t1 = current_time();
            const double elapsed = t1 - t0;
            write_time_metric.set(elapsed);

            // Mark frame as done
            DEBUG("mark_frame_empty: frame_id={}", frame_id);
            buffer->mark_frame_empty(unique_name, frame_id);

        } // for

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(zarrFileWrite);
