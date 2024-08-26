#include "gdalFiles.hpp"

#include <Stage.hpp>
#include <StageFactory.hpp>
#include <algorithm>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstdint>
#include <gdal.h>
#include <gdal_priv.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <prometheusMetrics.hpp>
#include <sstream>
#include <string>
#include <visUtil.hpp>

using namespace gdal;

class gdalFileRead : public kotekan::Stage {
    const std::string input_dir = config.get<std::string>(unique_name, "input_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);

    Buffer* const buffer;

public:
    gdalFileRead(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("out_buf")) {

        static std::once_flag gdal_register;
        std::call_once(gdal_register, GDALAllRegister);

        assert(buffer);
        buffer->register_producer(unique_name);
    }

    virtual ~gdalFileRead() {}

    void main_thread() override {
        auto& read_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_gdalfileread_read_time_seconds", unique_name);

        for (int frame_index = 0;; ++frame_index) {
            const int frame_id = frame_index % buffer->num_frames;

            if (stop_thread)
                break;

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
            buf << file_name << "." << std::setw(8) << std::setfill('0') << frame_index << ".zarr";
            const std::string full_path = buf.str();

            // Open file
            const auto dataset = std::unique_ptr<GDALDataset>(static_cast<GDALDataset*>(GDALOpenEx(
                full_path.c_str(),
                GDAL_OF_MULTIDIM_RASTER | GDAL_OF_READONLY | GDAL_OF_SHARED | GDAL_OF_VERBOSE_ERROR,
                nullptr, nullptr, nullptr)));
            if (!dataset) {
                INFO("Could not open GDAL file {:s}, terminating reader", full_path);
                break;
            }
            const auto group = dataset->GetRootGroup();

            // Wait for buffer
            DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
            std::uint8_t* const frame = buffer->wait_for_empty_frame(unique_name, frame_id);
            if (!frame)
                break;

            // Read metadata (attributes)
            DEBUG("[{:s}/{:d}] Setting metadata...", buffer->buffer_name, frame_index);
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
                const auto name = group->GetAttribute("name");
                assert(name);
                const auto name_shape = name->GetDimensionsSize();
                assert(name_shape.empty());
                const auto name_datatype = name->GetDataType();
                assert(name_datatype.GetClass() == GEDTC_STRING);
                const std::string name_value = std::string(name->ReadAsString());
                meta->set_name(name_value);
                DEBUG("[{:s}/{:d}] meta->name={}", buffer->buffer_name, frame_index,
                      meta->get_name());
            }

            {
                const auto type = group->GetAttribute("type");
                assert(type);
                const auto type_shape = type->GetDimensionsSize();
                assert(type_shape.empty());
                const auto type_datatype = type->GetDataType();
                assert(type_datatype.GetClass() == GEDTC_STRING);
                const std::string type_value = std::string(type->ReadAsString());
                meta->type = chord_datatype_from_string(type_value);
                DEBUG("[{:s}/{:d}] meta->type={}", buffer->buffer_name, frame_index,
                      chord_datatype_string(meta->type));
                assert(meta->type != unknown_type);
            }

            {
                const auto chord_metadata_version_attribute =
                    group->GetAttribute("chord_metadata_version");
                assert(chord_metadata_version_attribute);
                const auto chord_metadata_version_attribute_shape =
                    chord_metadata_version_attribute->GetDimensionsSize();
                assert(chord_metadata_version_attribute_shape.size() == 1);
                assert(chord_metadata_version_attribute_shape.at(0) == 2);
                const auto chord_metadata_version_found =
                    chord_metadata_version_attribute->ReadAsIntArray();
                assert(chord_metadata_version_found.size() == 2);
                const int major = chord_metadata_version_found.at(0);
                const int minor = chord_metadata_version_found.at(1);
                assert(major >= 0);
                assert(minor >= 0);
                assert(major == chord_metadata_version.at(0));
                assert(minor <= chord_metadata_version.at(1));
            }

            {
                const auto nfreq = group->GetAttribute("nfreq");
                if (nfreq) {
                    const auto nfreq_shape = nfreq->GetDimensionsSize();
                    assert(nfreq_shape.empty());
                    meta->nfreq = nfreq->ReadAsInt();
                    DEBUG("[{:s}/{:d}] meta->nfreq={}", buffer->buffer_name, frame_index,
                          meta->nfreq);
                    assert(meta->nfreq >= 0);
                    assert(meta->nfreq <= CHORD_META_MAX_FREQ);
                } else {
                    meta->nfreq = -1;
                    DEBUG("[{:s}/{:d}] meta->nfreq", buffer->buffer_name, frame_index);
                }
            }

            {
                const auto coarse_freq = group->GetAttribute("coarse_freq");
                assert((meta->nfreq >= 0) == bool(coarse_freq));
                if (coarse_freq) {
                    const auto coarse_nfreqs_shape = coarse_freq->GetDimensionsSize();
                    assert(coarse_nfreqs_shape.size() == 1);
                    assert(std::ptrdiff_t(coarse_nfreqs_shape.at(0)) == meta->nfreq);
                    const auto coarse_freq_data = coarse_freq->ReadAsIntArray();
                    assert(std::ptrdiff_t(coarse_freq_data.size()) == meta->nfreq);
                    std::copy(coarse_freq_data.begin(), coarse_freq_data.end(), meta->coarse_freq);
                }
            }

            {
                const auto freq_upchan_factor = group->GetAttribute("freq_upchan_factor");
                assert((meta->nfreq >= 0) == bool(freq_upchan_factor));
                if (freq_upchan_factor) {
                    const auto freq_upchan_factor_shape = freq_upchan_factor->GetDimensionsSize();
                    assert(freq_upchan_factor_shape.size() == 1);
                    assert(std::ptrdiff_t(freq_upchan_factor_shape.at(0)) == meta->nfreq);
                    const auto freq_upchan_factor_data = freq_upchan_factor->ReadAsIntArray();
                    assert(std::ptrdiff_t(freq_upchan_factor_data.size()) == meta->nfreq);
                    std::copy(freq_upchan_factor_data.begin(), freq_upchan_factor_data.end(),
                              meta->freq_upchan_factor);
                }
            }

            {
                const auto sample0_offset = group->GetAttribute("sample0_offset");
                if (sample0_offset) {
                    const auto sample0_offset_shape = sample0_offset->GetDimensionsSize();
                    assert(sample0_offset_shape.empty());
                    // Cannot read int64_t directly yet...
                    meta->sample0_offset = sample0_offset->ReadAsDouble();
                    assert(meta->sample0_offset >= 0);
                } else {
                    meta->sample0_offset = -1;
                }
            }

            {
                const auto half_fpga_sample0 = group->GetAttribute("half_fpga_sample0");
                assert((meta->nfreq >= 0) == bool(half_fpga_sample0));
                if (half_fpga_sample0) {
                    const auto coarse_nfreqs_shape = half_fpga_sample0->GetDimensionsSize();
                    assert(coarse_nfreqs_shape.size() == 1);
                    assert(std::ptrdiff_t(coarse_nfreqs_shape.at(0)) == meta->nfreq);
                    // Cannot read int64_t directly yet...
                    const auto half_fpga_sample0_data = half_fpga_sample0->ReadAsDoubleArray();
                    assert(std::ptrdiff_t(half_fpga_sample0_data.size()) == meta->nfreq);
                    std::copy(half_fpga_sample0_data.begin(), half_fpga_sample0_data.end(),
                              meta->half_fpga_sample0);
                }
            }

            {
                const auto time_downsampling_fpga = group->GetAttribute("time_downsampling_fpga");
                assert((meta->nfreq >= 0) == bool(time_downsampling_fpga));
                if (time_downsampling_fpga) {
                    const auto coarse_nfreqs_shape = time_downsampling_fpga->GetDimensionsSize();
                    assert(coarse_nfreqs_shape.size() == 1);
                    assert(std::ptrdiff_t(coarse_nfreqs_shape.at(0)) == meta->nfreq);
                    const auto time_downsampling_fpga_data =
                        time_downsampling_fpga->ReadAsIntArray();
                    assert(std::ptrdiff_t(time_downsampling_fpga_data.size()) == meta->nfreq);
                    std::copy(time_downsampling_fpga_data.begin(),
                              time_downsampling_fpga_data.end(), meta->time_downsampling_fpga);
                }
            }

            {
                const auto ndishes = group->GetAttribute("ndishes");
                if (ndishes) {
                    const auto ndishes_shape = ndishes->GetDimensionsSize();
                    assert(ndishes_shape.empty());
                    meta->ndishes = ndishes->ReadAsInt();
                    assert(meta->ndishes >= 0);
                    assert(meta->ndishes <= CHORD_META_MAX_FREQ);
                } else {
                    meta->ndishes = -1;
                }
            }

            {
                const auto dish_index = group->OpenMDArray("dish_index");
                if (dish_index) {
                    const auto dimensions = dish_index->GetDimensions();
                    assert(dimensions.size() == 2);
                    assert(dimensions.at(0)->GetName() == "dishM");
                    assert(dimensions.at(1)->GetName() == "dishN");
                    meta->n_dish_locations_ns = dimensions.at(0)->GetSize();
                    meta->n_dish_locations_ew = dimensions.at(1)->GetSize();
                    assert(meta->n_dish_locations_ns >= 0);
                    assert(meta->n_dish_locations_ew >= 0);
                    const std::vector<GUInt64> arrayStartIdx{0, 0};
                    const std::vector<std::size_t> count{std::size_t(meta->n_dish_locations_ns),
                                                         std::size_t(meta->n_dish_locations_ew)};
                    const auto datatype =
                        GDALExtendedDataType::Create(get_gdal_datatype(*meta->dish_index));
                    meta->dish_index =
                        new int[meta->n_dish_locations_ns * meta->n_dish_locations_ew];
                    const auto success =
                        dish_index->Read(arrayStartIdx.data(), count.data(), nullptr, nullptr,
                                         datatype, meta->dish_index, meta->dish_index,
                                         sizeof *meta->dish_index * meta->n_dish_locations_ns
                                             * meta->n_dish_locations_ew);
                    assert(success);
                } else {
                    meta->dish_index = nullptr;
                }
            }

            // Read buffer
            DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);

            {
                const auto mdarray = group->OpenMDArray(meta->get_name());
                assert(mdarray);

                const auto dimensions = mdarray->GetDimensions();
                meta->dims = dimensions.size();
                assert(meta->dims <= CHORD_META_MAX_DIM);
                for (int d = 0; d < meta->dims; ++d) {
                    meta->dim[d] = dimensions.at(d)->GetSize();
                    assert(meta->dim[d] >= 0);
                }
                for (int d = 0; d < meta->dims; ++d)
                    std::strncpy(meta->dim_name[d], dimensions.at(d)->GetName().c_str(),
                                 CHORD_META_MAX_DIMNAME);
                for (int d = meta->dims - 1; d >= 0; --d)
                    meta->stride[d] =
                        d == meta->dims - 1 ? 1 : meta->dim[d + 1] * meta->stride[d + 1];
                meta->offset = 0;

                const std::vector<GUInt64> arrayStartIdx(meta->dims, 0);
                std::vector<std::size_t> count(meta->dims);
                for (int d = 0; d < meta->dims; ++d)
                    count.at(d) = meta->dim[d];

                const auto datatype = mdarray->GetDataType();
                const std::ptrdiff_t data_size =
                    std::int64_t(1) * datatype.GetSize() * meta->stride[0] * meta->dim[0];
                assert(data_size == std::ptrdiff_t(buffer->frame_size));

                const auto success =
                    mdarray->Read(arrayStartIdx.data(), count.data(), nullptr, nullptr, datatype,
                                  frame, frame, buffer->frame_size);
                assert(success);
            }

            // Mark buffer as full
            DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
            buffer->mark_frame_full(unique_name, frame_id);

            // Stop timer
            const double t1 = current_time();
            const double elapsed = t1 - t0;
            read_time_metric.set(elapsed);

        } // while !stop_thread
    }
};

REGISTER_KOTEKAN_STAGE(gdalFileRead);
