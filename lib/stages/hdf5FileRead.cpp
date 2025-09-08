#include "hdf5Files.hpp"

#include <Stage.hpp>
#include <StageFactory.hpp>
#include <algorithm>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <memory>
#include <mutex>
#include <prometheusMetrics.hpp>
#include <sstream>
#include <string>
#include <unistd.h>
#include <visUtil.hpp>

using namespace hdf5;
using namespace HighFive;

class hdf5FileRead : public kotekan::Stage {
    const std::string input_dir = config.get<std::string>(unique_name, "input_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    const bool do_once = config.get_default<bool>(unique_name, "do_once", false);

    Buffer* const buffer;

public:
    hdf5FileRead(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("out_buf")) {
        assert(buffer);
        buffer->register_producer(unique_name);
    }

    virtual ~hdf5FileRead() {}

    void main_thread() override {
        auto& read_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_hdf5fileread_read_time_seconds", unique_name);

        for (int frame_index = 0;; ++frame_index) {
            const int frame_id = frame_index % buffer->num_frames;

        wait:

            if (stop_thread)
                break;

            if (do_once && frame_index > 0) {
                sleep(1);
                goto wait;
            }

            // Start timer
            const double t0 = current_time();

            // Define file name
            std::ostringstream buf;
            buf << input_dir << "/";
            if (prefix_hostname) {
                char hostname[256];
                gethostname(hostname, sizeof hostname);
                buf << hostname << "_";
            }
            buf << file_name << "." << std::setw(8) << std::setfill('0') << frame_index << ".h5";
            const std::string full_path = buf.str();

            // Open HDF5 file
            try {
                const File file(full_path, File::ReadOnly);

                // Wait for buffer
                DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
                std::uint8_t* const frame = buffer->wait_for_empty_frame(unique_name, frame_id);
                if (!frame)
                    break;

                // Read metadata (attributes)
                buffer->allocate_new_metadata_object(frame_id);
                const std::shared_ptr<metadataObject> metadata = buffer->get_metadata(frame_id);
                if (!metadata)
                    FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                                buffer->buffer_name, frame_id);
                assert(metadata);
                if (!metadata_is_chord(metadata))
                    FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD",
                                buffer->buffer_name, frame_id);
                assert(metadata_is_chord(metadata));
                const std::shared_ptr<chordMetadata> meta = get_chord_metadata(metadata);
                assert(meta);

                // Open dataset
                const auto dataset = file.getDataSet(file_name);
                const auto space = dataset.getSpace();
                const auto type = dataset.getDataType();
                const auto dims = space.getDimensions();

                {
                    const auto metadata_version =
                        dataset.getAttribute("chord_metadata_version").read<std::array<int, 2>>();
                    const int major = metadata_version[0];
                    const int minor = metadata_version[1];
                    assert(major >= 0);
                    assert(minor >= 0);
                    assert(major == chord_metadata_version.at(0));
                    assert(minor <= chord_metadata_version.at(1));
                }

                meta->set_name(dataset.getAttribute("name").read<std::string>());
                meta->type =
                    kotekan::string_to_type(dataset.getAttribute("type").read<std::string>());
                meta->dims = space.getNumberDimensions();
                assert(meta->dims <= CHORD_META_MAX_DIM);
                const auto dim_names =
                    dataset.getAttribute("dim_names").read<std::vector<std::string>>();
                assert(std::ptrdiff_t(dim_names.size()) == meta->dims);
                for (int d = 0; d < meta->dims; ++d)
                    meta->set_array_dimension(d, dims.at(d), dim_names.at(d));
                {
                    std::ptrdiff_t npoints = 1;
                    for (int d = meta->dims - 1; d >= 0; --d) {
                        meta->stride[d] = npoints;
                        npoints *= meta->dim[d];
                    }
                    assert(std::ptrdiff_t(space.getElementCount()) == npoints);
                }
                meta->offset = 0;

                if (dataset.hasAttribute("sample0_offset"))
                    meta->sample0_offset = dataset.getAttribute("sample0_offset").read<int>();
                else
                    meta->sample0_offset = -1;
                if (dataset.hasAttribute("offset_downsampling"))
                    meta->offset_downsampling =
                        dataset.getAttribute("offset_downsampling").read<int>();
                else
                    meta->offset_downsampling = -1;

                if (dataset.hasAttribute("nfreq")) {
                    meta->nfreq = dataset.getAttribute("nfreq").read<int>();
                    assert(meta->nfreq <= CHORD_META_MAX_FREQ);

                    const auto coarse_freq =
                        dataset.getAttribute("coarse_freq").read<std::vector<int>>();
                    assert(std::ptrdiff_t(coarse_freq.size()) == meta->nfreq);
                    std::copy(coarse_freq.begin(), coarse_freq.end(), meta->coarse_freq);

                    const auto freq_upchan_factor =
                        dataset.getAttribute("freq_upchan_factor").read<std::vector<int>>();
                    assert(std::ptrdiff_t(freq_upchan_factor.size()) == meta->nfreq);
                    std::copy(freq_upchan_factor.begin(), freq_upchan_factor.end(),
                              meta->freq_upchan_factor);

                    const auto half_fpga_sample0 =
                        dataset.getAttribute("half_fpga_sample0").read<std::vector<std::int64_t>>();
                    assert(std::ptrdiff_t(half_fpga_sample0.size()) == meta->nfreq);
                    std::copy(half_fpga_sample0.begin(), half_fpga_sample0.end(),
                              meta->half_fpga_sample0);

                    const auto time_downsampling_fpga =
                        dataset.getAttribute("time_downsampling_fpga").read<std::vector<int>>();
                    assert(std::ptrdiff_t(time_downsampling_fpga.size()) == meta->nfreq);
                    std::copy(time_downsampling_fpga.begin(), time_downsampling_fpga.end(),
                              meta->time_downsampling_fpga);

                } else {
                    meta->nfreq = -1;
                }


                if (dataset.hasAttribute("ndishes")) {
                    meta->ndishes = dataset.getAttribute("ndishes").read<int>();

                    meta->n_dish_locations_ns =
                        dataset.getAttribute("n_dish_locations_ns").read<int>();
                    meta->n_dish_locations_ew =
                        dataset.getAttribute("n_dish_locations_ew").read<int>();

                    const auto dish_index =
                        dataset.getAttribute("dish_index").read<std::vector<int>>();
                    assert(std::ptrdiff_t(dish_index.size())
                           == meta->n_dish_locations_ns * meta->n_dish_locations_ew);
                    meta->dish_index =
                        new int[meta->n_dish_locations_ns * meta->n_dish_locations_ew];
                    std::copy(dish_index.begin(), dish_index.end(), meta->dish_index);

                } else {
                    meta->ndishes = -1;
                    meta->dish_index = nullptr;
                }

                // Read buffer
                DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);
                dataset.read_raw(frame, type);

                // Mark buffer as full
                DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
                buffer->mark_frame_full(unique_name, frame_id);

                // Stop timer
                const double t1 = current_time();
                const double elapsed = t1 - t0;
                read_time_metric.set(elapsed);
            } catch (const FileException& ex) {
                INFO("Could not open HDF5 file {:s}, terminating reader", full_path);
                break;
            }

        } // while !stop_thread
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileRead);
