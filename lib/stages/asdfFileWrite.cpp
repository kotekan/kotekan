#include <Stage.hpp>
#include <StageFactory.hpp>
#include <asdf/asdf.hxx>
#include <atomic>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstdint>
#include <errno.h>
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

namespace {
ASDF::scalar_type_id_t chord2asdf(const chordDataType type) {
    switch (type) {
        case int4p4:
            return ASDF::id_uint8; // TODO: Define ASDF int4+4 type
        case int8:
            return ASDF::id_int8;
        case int16:
            return ASDF::id_int16;
        case int32:
            return ASDF::id_int32;
        case int64:
            return ASDF::id_int64;
        case float16:
            return ASDF::id_float16;
        case float32:
            return ASDF::id_float32;
        case float64:
            return ASDF::id_float64;
        default:
            assert(0);
    }
}
} // namespace

class asdfFileWrite : public kotekan::Stage {

    const std::string base_dir = config.get<std::string>(unique_name, "base_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);

    const std::uint32_t exit_after_n_frames;
    const std::uint32_t exit_with_n_writers;

    Buffer* const buffer;

    static std::atomic<std::uint32_t> n_finished;

public:
    asdfFileWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        exit_after_n_frames(config.get<std::uint32_t>(unique_name, "exit_after_n_frames")),
        exit_with_n_writers(config.get<std::uint32_t>(unique_name, "exit_with_n_writers")),
        buffer(get_buffer("in_buf")) {
        buffer->register_consumer(unique_name);
    }

    virtual ~asdfFileWrite() {}

    void main_thread() override {
        auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_asdffilewrite_write_time_seconds", unique_name);

        for (std::uint32_t frame_counter = 0;
             exit_after_n_frames == 0 || frame_counter < exit_after_n_frames; ++frame_counter) {
            const std::uint32_t frame_id = frame_counter % buffer->num_frames;

            if (stop_thread)
                break;

            // Wait for the next frame
            const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, frame_id);
            if (frame == nullptr)
                break;

            // Start timer
            const double t0 = current_time();

            // Fetch metadata
            const std::shared_ptr<metadataObject> mc = buffer->get_metadata(frame_id);
            assert(mc);
            assert(metadata_is_chord(mc));
            const std::shared_ptr<chordMetadata> meta = get_chord_metadata(mc);

            // Create ASDF project
            auto group = std::make_shared<ASDF::group>();

            // Create ASDF ndarray
            const ASDF::scalar_type_id_t type = chord2asdf(meta->type);
            const std::size_t typesize = chord_datatype_bytes(meta->type);

            const int ndims = meta->dims;
            std::vector<std::int64_t> dims(ndims);
            for (int d = 0; d < ndims; ++d)
                dims.at(d) = meta->dim[d];
            std::int64_t size = 1;
            for (int d = 0; d < ndims; ++d)
                size *= dims.at(d);

            const std::shared_ptr<ASDF::block_t> block = std::make_shared<ASDF::ptr_block_t>(
                const_cast<std::uint8_t*>(frame), size * typesize);

            const auto compression = ASDF::compression_t::blosc;
            const int compression_level = 9;

            const auto ndarray = std::make_shared<ASDF::ndarray>(
                ASDF::make_constant_memoized(block), std::optional<ASDF::block_info_t>(),
                ASDF::block_format_t::block, compression, compression_level, std::vector<bool>(),
                std::make_shared<ASDF::datatype_t>(type), ASDF::host_byteorder(), dims);
            group->emplace(buffer->buffer_name, std::make_shared<ASDF::ndarray_entry>(ndarray));

            // Describe metadata

            group->emplace("nfreq", std::make_shared<ASDF::int_entry>(meta->nfreq));

            auto coarse_freq = std::make_shared<ASDF::sequence>();
            for (int freq = 0; freq < meta->nfreq; ++freq)
                coarse_freq->push_back(std::make_shared<ASDF::int_entry>(meta->coarse_freq[freq]));
            group->emplace("coarse_freq", coarse_freq);

            auto freq_upchan_factor = std::make_shared<ASDF::sequence>();
            for (int freq = 0; freq < meta->nfreq; ++freq)
                freq_upchan_factor->push_back(
                    std::make_shared<ASDF::int_entry>(meta->freq_upchan_factor[freq]));
            group->emplace("freq_upchan_factor", freq_upchan_factor);

            auto half_fpga_sample0 = std::make_shared<ASDF::sequence>();
            for (int freq = 0; freq < meta->nfreq; ++freq)
                half_fpga_sample0->push_back(
                    std::make_shared<ASDF::int_entry>(meta->half_fpga_sample0[freq]));
            group->emplace("half_fpga_sample0", half_fpga_sample0);

            auto time_downsampling_fpga = std::make_shared<ASDF::sequence>();
            for (int freq = 0; freq < meta->nfreq; ++freq)
                time_downsampling_fpga->push_back(
                    std::make_shared<ASDF::int_entry>(meta->time_downsampling_fpga[freq]));
            group->emplace("time_downsampling_fpga", time_downsampling_fpga);

            auto dim_names = std::make_shared<ASDF::sequence>();
            for (int d = 0; d < ndims; ++d)
                dim_names->push_back(
                    std::make_shared<ASDF::string_entry>(meta->get_dimension_name(d)));
            group->emplace("dim_names", dim_names);

            if (meta->ndishes >= 0)
                group->emplace("ndishes", std::make_shared<ASDF::int_entry>(meta->ndishes));

            if (meta->dish_index) {
                auto dish_index = std::make_shared<ASDF::ndarray>(
                    std::vector<int>(meta->dish_index,
                                     meta->dish_index
                                         + meta->n_dish_locations_ew * meta->n_dish_locations_ns),
                    ASDF::block_format_t::inline_array, ASDF::compression_t::none, -1,
                    std::vector<bool>(),
                    std::vector<int64_t>{meta->n_dish_locations_ns, meta->n_dish_locations_ew});
                auto dish_index_entry = std::make_shared<ASDF::ndarray_entry>(dish_index);
                group->emplace("dish_index", dish_index_entry);
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
                    char msg[1000];
                    char* p = strerror_r(errno, msg, sizeof msg);
                    if (!p)
                        FATAL_ERROR("Could not create directory \"{:s}\":\nerrno={:d}",
                                    base_dir.c_str(), errno);
                    else
                        FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", base_dir.c_str(),
                                    msg);
                }
            }
            project.write(full_path);

            // Stop timer
            const double t1 = current_time();
            const double elapsed = t1 - t0;
            write_time_metric.set(elapsed);

            // Mark frame as done
            buffer->mark_frame_empty(unique_name, frame_id);
        } // for

        if (exit_with_n_writers > 0 && ++n_finished >= exit_with_n_writers)
            exit_kotekan(ReturnCode::CLEAN_EXIT);
    }
};

REGISTER_KOTEKAN_STAGE(asdfFileWrite);

std::atomic<std::uint32_t> asdfFileWrite::n_finished{0};
