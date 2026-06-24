#include "Config.hpp"            // for Config
#include "DataType.hpp"          // for string_to_type, DataType
#include "NDArray.hpp"           // for GenericNDArray
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Symbol.hpp"            // for Symbol
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "chordMetadata.hpp"     // for chordMetadata, get_chord_metadata
#include "hdf5Files.hpp"         // for hdf5
#include "kotekanLogging.hpp"    // for DEBUG, INFO, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "visUtil.hpp"           // for current_time

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>                          // for minmax_element, copy, find
#include <cassert>                            // for assert
#include <cstddef>                            // for size_t, ptrdiff_t
#include <cstdint>                            // for uint8_t
#include <cstring>                            // for memchr, memcpy, memset
#include <functional>                         // for function
#include <highfive/H5Attribute.hpp>           // for Attribute, Attribute::read
#include <highfive/H5DataSet.hpp>             // for DataSet, AnnotateTraits::getAttribute, Dat...
#include <highfive/H5DataSpace.hpp>           // for DataSpace, DataSpace::getDimensions
#include <highfive/H5Exception.hpp>           // for FileException
#include <highfive/H5File.hpp>                // for File, File::File, NodeTraits::getDataSet
#include <highfive/H5Selection.hpp>           // for Selection, SliceTraits::read_raw, SliceTra...
#include <highfive/bits/H5Selection_misc.hpp> // for Selection::getSpace
#include <memory>                             // for allocator, __shared_ptr_access, shared_ptr
#include <sstream>                            // for basic_ostream, operator<<, basic_ostringst...
#include <string>                             // for basic_string, string, char_traits, operator==
#include <unistd.h>                           // for sleep
#include <utility>                            // for pair
#include <vector>                             // for vector

using namespace hdf5;
using namespace HighFive;

class hdf5FileReadSingleFile : public kotekan::Stage {
    const std::string input_dir = config.get<std::string>(unique_name, "input_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");
    const int num_times = config.get<int>(unique_name, "num_times");

    const bool combine_dishes_and_polarizations =
        config.get_default<bool>(unique_name, "combine_dishes_and_polarizations", false);
    const bool transpose_frequency_and_time =
        config.get_default<bool>(unique_name, "transpose_frequency_and_time", false);

    Buffer* const buffer;

public:
    hdf5FileReadSingleFile(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("out_buf")) {
        assert(buffer);
        buffer->register_producer(unique_name);
    }

    virtual ~hdf5FileReadSingleFile() {}

    void main_thread() override {
        auto& read_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_hdf5filereadsinglefile_read_time_seconds", unique_name);

        // Define file name
        std::ostringstream buf;
        buf << input_dir << "/" << file_name << ".h5";
        const std::string full_path = buf.str();

        try {
            ////////////////////////////////////////////////////////////////////////////////

            // Open HDF5 file
            const File file(full_path, File::ReadOnly);

            // Open dataset
            const auto dataset = file.getDataSet(file_name);
            const auto space = dataset.getSpace();
            const auto hdf5_type = dataset.getDataType();
            const auto dims = space.getDimensions();

            // Read attributes
            const auto name = dataset.getAttribute("name").read<std::string>();
            const auto type =
                kotekan::string_to_type(dataset.getAttribute("type").read<std::string>());
            assert(type == kotekan::int4x2_swapped_withoffset);
            const auto dim_names =
                dataset.getAttribute("dim_names").read<std::vector<std::string>>();
            const auto dim_scalings =
                dataset.getAttribute("dim_scalings").read<std::vector<std::ptrdiff_t>>();

            const auto coarse_freq = dataset.getAttribute("coarse_freq").read<std::vector<int>>();
            const auto freq_upchan_factor =
                dataset.getAttribute("freq_upchan_factor").read<std::vector<int>>();
            const auto freq_upchan_index =
                dataset.getAttribute("freq_upchan_index").read<std::vector<int>>();

            const auto time_downsampling_fpga =
                dataset.getAttribute("time_downsampling_fpga").read<int>();
            const auto fpga_seq_num = dataset.getAttribute("fpga_seq_num").read<int>();
            const auto seq_length_nsec [[maybe_unused]] =
                dataset.getAttribute("seq_length_nsec").read<double>();

            // Convert dimensions
            assert(dims.size() == 4);
            assert(dim_names.size() == 4);
            assert(dim_scalings.size() == 4);
            std::vector<std::ptrdiff_t> new_dims;
            std::vector<kotekan::Symbol> new_dim_names;
            std::vector<std::ptrdiff_t> new_dim_scalings;
            if (!transpose_frequency_and_time) {
                // Standard case
                new_dims.push_back(num_times);
                new_dims.push_back(frequency_channels.size());
                new_dim_names.push_back(dim_names.at(0));
                new_dim_names.push_back(dim_names.at(1));
                new_dim_scalings.push_back(dim_scalings.at(0));
                new_dim_scalings.push_back(dim_scalings.at(1));
            } else {
                // For CHIME, tranpose time and frequency
                new_dims.push_back(frequency_channels.size());
                new_dims.push_back(num_times);
                new_dim_names.push_back(dim_names.at(1));
                new_dim_names.push_back(dim_names.at(0));
                new_dim_scalings.push_back(dim_scalings.at(1));
                new_dim_scalings.push_back(dim_scalings.at(0));
            }
            if (!combine_dishes_and_polarizations) {
                // Standard case
                new_dims.push_back(dims.at(2));
                new_dims.push_back(dims.at(3));
                new_dim_names.push_back(dim_names.at(2));
                new_dim_names.push_back(dim_names.at(3));
                new_dim_scalings.push_back(dim_scalings.at(2));
                new_dim_scalings.push_back(dim_scalings.at(3));
            } else {
                // For CHIME, combine dishes and polarizations into elements
                new_dims.push_back(dims.at(2) * dims.at(3));
                new_dim_names.push_back("E");
                new_dim_scalings.push_back(dim_scalings.at(2) * dim_scalings.at(3));
            }

            // Find which frequencies to read
            assert(std::string(dim_names.at(1)) == "F");
            assert(dim_scalings.at(1) == 1);
            INFO("This file has {} frequencies", dims.at(1));
            INFO("Will read {} frequencies", frequency_channels.size());
            std::vector<int> frequency_indices(frequency_channels.size());
            for (std::size_t n = 0; n < frequency_channels.size(); ++n) {
                const int frequency_channel = frequency_channels.at(n);
                const auto frequency_index_iter =
                    std::find(coarse_freq.begin(), coarse_freq.end(), frequency_channel);
                assert(frequency_index_iter != coarse_freq.end());
                const int frequency_index = frequency_index_iter - coarse_freq.begin();
                frequency_indices.at(n) = frequency_index;
                INFO("   local array index {}, file dataset index {}, channel {}", n,
                     frequency_index, frequency_channel);
            }
            std::vector<int> new_freq_upchan_factor(frequency_indices.size());
            std::vector<int> new_freq_upchan_index(frequency_indices.size());
            for (std::size_t n = 0; n < frequency_indices.size(); ++n) {
                const int frequency_index = frequency_indices.at(n);
                new_freq_upchan_factor.at(n) = freq_upchan_factor.at(frequency_index);
                new_freq_upchan_index.at(n) = freq_upchan_index.at(frequency_index);
            }

            // Calculate available number of frames
            assert(std::string(dim_names.at(0)) == "T");
            assert(dim_scalings.at(0) == 1);
            const std::ptrdiff_t available_num_times = dims.at(0);
            const std::ptrdiff_t available_num_frames = available_num_times / num_times;
            INFO("This file has {} time samples", available_num_times);
            INFO("Will read {} frames with {} time samples each", available_num_frames, num_times);

            ////////////////////////////////////////////////////////////////////////////////

            // Loop over frames
            for (int frame_index = 0; frame_index < available_num_frames; ++frame_index) {
                DEBUG("Reading frame {}", frame_index);
                const int frame_id = frame_index % buffer->num_frames;

                if (stop_thread)
                    break;

                // Start timer
                const double t0 = current_time();

                // Wait for buffer
                DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
                std::uint8_t* const frame = buffer->wait_for_empty_frame(unique_name, frame_id);
                if (!frame)
                    break;

                // Set metadata
                buffer->require_frame_desc(kotekan::GenericNDArray::describe(
                    type, kotekan::Symbol(name), new_dims, new_dim_names, new_dim_scalings));
                buffer->allocate_new_metadata_object(frame_id);
                const auto& meta = get_chord_metadata(buffer->get_metadata(frame_id));
                meta->set_from_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());

                meta->set_coarse_freq(frequency_channels);
                meta->set_freq_upchan_factor(new_freq_upchan_factor);
                meta->set_freq_upchan_index(new_freq_upchan_index);

                meta->set_time_downsampling_fpga(time_downsampling_fpga);
                meta->set_fpga_seq_num(fpga_seq_num + 1LL * frame_index * num_times);

                // Poison buffer
                const std::uint8_t voltage_poison = 0x00;
                std::memset(frame, voltage_poison, buffer->frame_size);

                // Read buffer
                DEBUG("[{:s}/{:d}] Reading buffer from file...", buffer->buffer_name, frame_index);
                const std::ptrdiff_t buf_stride = dims.at(2) * dims.at(3);
                const std::ptrdiff_t frame_stride =
                    dims.at(2) * dims.at(3) * frequency_channels.size();
                std::vector<std::uint8_t> buf(buf_stride * num_times);
                // Loop over all frequencies that should be read
                for (std::size_t freq = 0; freq < frequency_indices.size(); ++freq) {
                    const int frequency_index = frequency_indices.at(freq);
                    assert(dims.size() == 4);
                    // Choose a hyperslab for reading
                    const std::vector<std::size_t> offset{
                        std::size_t(1LL * frame_index * num_times), std::size_t(frequency_index), 0,
                        0};
                    const std::vector<std::size_t> count{std::size_t(num_times), 1, dims.at(2),
                                                         dims.at(3)};
                    const auto selection = dataset.select(offset, count);
                    // Transpose after reading
                    // TODO: Use memory hyperslab
                    // selection.read_raw(frame, hdf5_type);
                    selection.read_raw(buf.data(), hdf5_type);
                    for (int time = 0; time < num_times; ++time) {
                        const std::uint8_t* __restrict__ const src = buf.data() + buf_stride * time;
                        std::uint8_t* __restrict__ const dst =
                            frame + buf_stride * freq + frame_stride * time;
                        std::memcpy(dst, src, buf_stride);
                    }
                }

                // Check for poison
                // (Poison can either originate from a bad input file, or from a logic error in this
                // reader function.)
                if (std::memchr(frame, voltage_poison, buffer->frame_size))
                    FATAL_ERROR("Poison found in voltage buffer after reading");

                // Mark buffer as full
                DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
                buffer->mark_frame_full(unique_name, frame_id);

                // Stop timer
                const double t1 = current_time();
                const double elapsed = t1 - t0;
                read_time_metric.set(elapsed);

            } // for frame_index

            DEBUG("Done reading frames.");

        } catch (const FileException& ex) {
            FATAL_ERROR("Could not open HDF5 file {:s}: {:s}", full_path, ex.what());
        }

        // Wait for shutdown (don't trigger a shutdown)
        while (!stop_thread)
            sleep(1);
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileReadSingleFile);
