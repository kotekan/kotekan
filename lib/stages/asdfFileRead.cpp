#include "asdfFiles.hpp"

#include <DataType.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <asdf/asdf.hxx>
#include <atomic>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstdint>
#include <cstring>
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
#include <visUtil.hpp>

using namespace asdf;

class asdfFileRead : public kotekan::Stage {
    const std::string input_dir = config.get<std::string>(unique_name, "input_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    const bool do_once = config.get_default<bool>(unique_name, "do_once", false);

    Buffer* const buffer;

public:
    asdfFileRead(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("out_buf")) {

        ASDF_CHECK_VERSION();

        assert(buffer);
        buffer->register_producer(unique_name);
    }

    virtual ~asdfFileRead() {}

    void main_thread() override {
        auto& read_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_asdffileread_read_time_seconds", unique_name);

        for (std::int64_t frame_counter = 0;; ++frame_counter) {
            const std::uint32_t frame_id = frame_counter % buffer->num_frames;

        wait:

            if (stop_thread)
                break;

            if (do_once && frame_counter > 0) {
                sleep(1);
                goto wait;
            }

            // Start timer
            const double t0 = current_time();

            // Define file name
            std::ostringstream ibuf;
            ibuf << std::setw(8) << std::setfill('0') << frame_counter;
            const std::string iteration = ibuf.str();
            std::ostringstream buf;
            buf << input_dir << "/";
            if (prefix_hostname) {
                char hostname[256];
                gethostname(hostname, sizeof hostname);
                buf << hostname << "_";
            }
            buf << file_name << "." << std::setw(8) << std::setfill('0') << frame_counter
                << ".asdf";
            const std::string full_path = buf.str();

            // Read project from file
            auto file = std::make_shared<std::ifstream>(full_path);
            if (!file->good()) {
                INFO("Could not open ASDF file {:s}, terminating reader", full_path);
                break;
            }
            DEBUG("[{:s}/{:d}] std::make_unique<ASDF::asdf>(full_path=\"{}\")", buffer->buffer_name,
                  frame_counter, full_path);
            const auto project = std::make_unique<ASDF::asdf>(file);
            assert(project);
            const auto group0 = project->get_group()->get_group();

            DEBUG("[{:s}/{:d}] group0->at(iteration)", buffer->buffer_name, frame_counter);
            const auto group = group0->at(iteration)->get_maybe_group();
            assert(group);

            // Wait for buffer
            DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_counter);
            std::uint8_t* const frame = buffer->wait_for_empty_frame(unique_name, frame_id);
            if (!frame)
                break;

            // Read metadata (attributes), part 1
            DEBUG("[{:s}/{:d}] Setting metadata...", buffer->buffer_name, frame_counter);
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

            {
                DEBUG("[{:s}/{:d}] group0->at(\"chord_metadata_version\")", buffer->buffer_name,
                      frame_counter);
                const auto chord_metadata_version_attribute =
                    group->at("chord_metadata_version")->get_maybe_sequence();
                assert(chord_metadata_version_attribute->size() == 2);
                const int major = chord_metadata_version_attribute->at(0)->get_maybe_int().value();
                const int minor = chord_metadata_version_attribute->at(1)->get_maybe_int().value();
                assert(major >= 0);
                assert(minor >= 0);
                assert(major == chord_metadata_version.at(0));
                assert(minor <= chord_metadata_version.at(1));
            }

            {
                DEBUG("[{:s}/{:d}] group0->at(\"name\")", buffer->buffer_name, frame_counter);
                const std::string name = group->at("name")->get_maybe_string().value();
                meta->set_name(name);
                DEBUG("[{:s}/{:d}] meta->name={}", buffer->buffer_name, frame_counter,
                      meta->get_name());
            }

            {
                DEBUG("[{:s}/{:d}] group0->at(\"type\")", buffer->buffer_name, frame_counter);
                const std::string type = group->at("type")->get_maybe_string().value();
                meta->type = kotekan::string_to_type(type);
                DEBUG("[{:s}/{:d}] meta->type={}", buffer->buffer_name, frame_counter,
                      type_to_string(meta->type));
                assert(meta->type != kotekan::unknown_type);
            }

            // Read buffer
            DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_counter);

            {
                const std::string ndarray_name = beautify_buffer_name(buffer->buffer_name);
                DEBUG("[{:s}/{:d}] group0->at(ndarray_name=\"{:s}\")", buffer->buffer_name,
                      frame_counter, ndarray_name);
                const auto ndarray = group->at(ndarray_name)->get_maybe_ndarray();
                assert(ndarray);

                const auto dimensions = ndarray->get_shape();
                meta->dims = dimensions.size();
                assert(meta->dims <= CHORD_META_MAX_DIM);
                for (int d = 0; d < meta->dims; ++d) {
                    meta->dim[d] = dimensions.at(d);
                    assert(meta->dim[d] >= 0);
                }

                const auto datatype = ndarray->get_datatype();
                // TODO: Check datatype
                const auto datatype_size = datatype->type_size();

                const auto strides = ndarray->get_strides();
                assert(std::ptrdiff_t(strides.size()) == meta->dims);
                for (int d = 0; d < meta->dims; ++d) {
                    assert(strides.at(d) % datatype_size == 0);
                    meta->stride[d] = strides.at(d) / datatype_size;
                }

                const auto offset = ndarray->get_offset();
                assert(offset % datatype_size == 0);
                meta->offset = offset / datatype_size;

                std::ptrdiff_t npoints = 1;
                for (int d = meta->dims - 1; d >= 0; --d) {
                    assert(meta->stride[d] == npoints);
                    npoints *= meta->dim[d];
                }
                const std::ptrdiff_t data_size = npoints * datatype_size;
                DEBUG("[{:s}/{:d}] data_size={} datatype->type_size={} meta->stride[0]={} "
                      "meta->dim[0]={} buffer->frame_size={}",
                      buffer->buffer_name, frame_counter, data_size, datatype->type_size(),
                      meta->stride[0], meta->dim[0], buffer->frame_size);
                assert(data_size == std::ptrdiff_t(buffer->frame_size));

                assert(std::ptrdiff_t(buffer->frame_size) == data_size);
                const auto data = ndarray->get_data();
                assert(std::ptrdiff_t(data->nbytes()) == data_size);
                std::memcpy(frame, data->ptr(), buffer->frame_size);
            }

            {
                DEBUG("[{:s}/{:d}] group0->at(\"dim_names\")", buffer->buffer_name, frame_counter);
                const auto dim_names = group->at("dim_names")->get_maybe_sequence();
                assert(dim_names);
                meta->dims = dim_names->size();
                assert(meta->dims <= CHORD_META_MAX_DIM);
                for (int d = 0; d < meta->dims; ++d) {
                    const std::string dim_name = dim_names->at(d)->get_maybe_string().value();
                    std::strncpy(meta->dim_name[d], dim_name.c_str(), CHORD_META_MAX_DIMNAME);
                }
            }

            // Read metadata (attributes), part 2

            {
                if (group->count("coarse_freq")) {
                    DEBUG("[{:s}/{:d}] group0->at(\"coarse_freq\")", buffer->buffer_name,
                          frame_counter);
                    const auto coarse_freq = group->at("coarse_freq")->get_maybe_sequence();
                    assert(coarse_freq);
                    meta->nfreq = coarse_freq->size();
                    assert(meta->nfreq <= CHORD_META_MAX_FREQ);
                    assert(meta->nfreq >= 0);
                    assert(std::ptrdiff_t(coarse_freq->size()) == meta->nfreq);
                    assert(meta->nfreq <= CHORD_META_MAX_FREQ);
                    for (int n = 0; n < meta->nfreq; ++n)
                        meta->coarse_freq[n] = coarse_freq->at(n)->get_maybe_int().value();
                } else {
                    meta->nfreq = -1;
                    assert(meta->nfreq < 0);
                }
            }

            {
                if (group->count("freq_upchan_factor")) {
                    assert(meta->nfreq >= 0);
                    DEBUG("[{:s}/{:d}] group0->at(\"freq_upchan_factor\")", buffer->buffer_name,
                          frame_counter);
                    const auto freq_upchan_factor =
                        group->at("freq_upchan_factor")->get_maybe_sequence();
                    assert(freq_upchan_factor);
                    assert(std::ptrdiff_t(freq_upchan_factor->size()) == meta->nfreq);
                    assert(meta->nfreq <= CHORD_META_MAX_FREQ);
                    for (int n = 0; n < meta->nfreq; ++n)
                        meta->freq_upchan_factor[n] =
                            freq_upchan_factor->at(n)->get_maybe_int().value();
                } else {
                    assert(meta->nfreq < 0);
                }
            }

            {
                if (group->count("sample0_offset")) {
                    DEBUG("[{:s}/{:d}] group0->at(\"sample0_offset\")", buffer->buffer_name,
                          frame_counter);
                    const auto sample0_offset =
                        group->at("sample0_offset")->get_maybe_int().value();
                    meta->sample0_offset = sample0_offset;
                    assert(meta->sample0_offset >= 0);
                } else {
                    meta->sample0_offset = -1;
                }
            }

            {
                if (group->count("offset_downsampling")) {
                    DEBUG("[{:s}/{:d}] group0->at(\"offset_downsampling\")", buffer->buffer_name,
                          frame_counter);
                    const auto offset_downsampling =
                        group->at("offset_downsampling")->get_maybe_int().value();
                    meta->offset_downsampling = offset_downsampling;
                    assert(meta->offset_downsampling > 0);
                } else {
                    meta->offset_downsampling = -1;
                }
            }

            {
                if (group->count("half_fpga_sample0")) {
                    assert(meta->nfreq >= 0);
                    DEBUG("[{:s}/{:d}] group0->at(\"half_fpga_sample0\")", buffer->buffer_name,
                          frame_counter);
                    const auto half_fpga_sample0 =
                        group->at("half_fpga_sample0")->get_maybe_sequence();
                    assert(half_fpga_sample0);
                    assert(std::ptrdiff_t(half_fpga_sample0->size()) == meta->nfreq);
                    assert(meta->nfreq <= CHORD_META_MAX_FREQ);
                    for (int n = 0; n < meta->nfreq; ++n)
                        meta->half_fpga_sample0[n] =
                            half_fpga_sample0->at(n)->get_maybe_int().value();
                } else {
                    assert(meta->nfreq < 0);
                }
            }

            {
                if (group->count("time_downsampling_fpga")) {
                    assert(meta->nfreq >= 0);
                    DEBUG("[{:s}/{:d}] group0->at(\"time_downsampling_fpga\")", buffer->buffer_name,
                          frame_counter);
                    const auto time_downsampling_fpga =
                        group->at("time_downsampling_fpga")->get_maybe_sequence();
                    assert(time_downsampling_fpga);
                    assert(std::ptrdiff_t(time_downsampling_fpga->size()) == meta->nfreq);
                    assert(meta->nfreq <= CHORD_META_MAX_FREQ);
                    for (int n = 0; n < meta->nfreq; ++n)
                        meta->time_downsampling_fpga[n] =
                            time_downsampling_fpga->at(n)->get_maybe_int().value();
                } else {
                    assert(meta->nfreq < 0);
                }
            }

            {
                if (group->count("ndishes")) {
                    DEBUG("[{:s}/{:d}] group0->at(\"ndishes\")", buffer->buffer_name,
                          frame_counter);
                    const auto ndishes = group->at("ndishes")->get_maybe_int().value();
                    meta->ndishes = ndishes;
                    assert(meta->ndishes >= 0);
                } else {
                    meta->ndishes = -1;
                }
            }

            {
                if (group->count("dish_index")) {
                    DEBUG("[{:s}/{:d}] group0->at(\"dish_index\")", buffer->buffer_name,
                          frame_counter);
                    const auto dish_index = group->at("dish_index")->get_maybe_ndarray();
                    assert(dish_index);
                    const auto dimensions = dish_index->get_shape();
                    assert(dimensions.size() == 2);
                    meta->n_dish_locations_ns = dimensions.at(0);
                    meta->n_dish_locations_ew = dimensions.at(1);
                    assert(meta->n_dish_locations_ns >= 0);
                    assert(meta->n_dish_locations_ew >= 0);
                    meta->dish_index =
                        new int[meta->n_dish_locations_ns * meta->n_dish_locations_ew];
                    const auto data = dish_index->get_data();
                    assert(data->nbytes()
                           == sizeof(int) * meta->n_dish_locations_ns * meta->n_dish_locations_ew);
                    std::memcpy(meta->dish_index, data->ptr(), data->nbytes());
                } else {
                    meta->dish_index = nullptr;
                }
            }

            // Mark buffer as full
            DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_counter);
            buffer->mark_frame_full(unique_name, frame_id);

            // Stop timer
            const double t1 = current_time();
            const double elapsed = t1 - t0;
            read_time_metric.set(elapsed);

        } // while !stop_thread
    }
};

REGISTER_KOTEKAN_STAGE(asdfFileRead);
