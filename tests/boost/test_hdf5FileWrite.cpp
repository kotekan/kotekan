// Boost tests for hdf5FileWrite with record_per_frame_metadata, read back with hdf5FileRead

#define BOOST_TEST_MODULE "test_hdf5FileWrite"

#include "Config.hpp"                // for Config
#include "NDArray.hpp"               // for GenericNDArray
#include "Stage.hpp"                 // for Stage
#include "StageFactory.hpp"          // for StageFactoryRegistry
#include "Telescope.hpp"             // for Telescope
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "chordMetadata.hpp"         // for chordMetadata, get_chord_metadata
#include "configUpdater.hpp"         // for configUpdater
#include "hdf5Files.hpp"             // for frame_metadata_group
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "test_utils.hpp"            // for add_test_telescope_config

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <cstdint>                // for int64_t, int8_t
#include <cstdlib>                // for mkdtemp
#include <filesystem>             // for path, remove_all, directory_iterator
#include <highfive/H5DataSet.hpp> // for DataSet
#include <highfive/H5File.hpp>    // for File
#include <memory>                 // for unique_ptr, shared_ptr
#include <set>                    // for set
#include <string>                 // for string
#include <sys/time.h>             // for timeval
#include <vector>                 // for vector

using json = nlohmann::json;
namespace fs = std::filesystem;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

static const int num_frames = 6;
static const int T = 4; // time samples per frame
static const int F = 4; // frequencies per sender
static const int64_t period = 8192;

// A new directory under /tmp, removed at the end of the test case.
struct tmp_dir {
    fs::path path;
    tmp_dir() {
        char name[] = "/tmp/test_hdf5FileWrite_XXXXXX";
        BOOST_REQUIRE(mkdtemp(name) != nullptr);
        path = name;
    }
    ~tmp_dir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

static std::set<std::string> list_dir(const fs::path& dir) {
    std::set<std::string> names;
    for (const auto& entry : fs::directory_iterator(dir))
        names.insert(entry.path().filename().string());
    return names;
}

static std::shared_ptr<kotekan::GenericNDArray> frame_desc() {
    return kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask", {T, F}, {"T", "F"},
                                             {period, 1});
}

// A config with the stage `name` reading or writing "mask" files in `dir`. Each stage needs its
// own name, since it registers a metric under it.
static kotekan::Config stage_config(const std::string& name, const std::string& stage,
                                    const fs::path& dir, const json& options) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg["num_polarizations"] = 2;
    cfg["num_dishes"] = 2;
    add_test_telescope_config(cfg);
    cfg[name] = {{"kotekan_stage", stage}, {"file_name", "mask"}, {"prefix_hostname", false}};
    cfg[name][stage == "hdf5FileWrite" ? "base_dir" : "input_dir"] = dir.string();
    cfg[name][stage == "hdf5FileWrite" ? "in_buf" : "out_buf"] = "buf";
    if (options.is_object())
        cfg[name].update(options);
    kotekan::Config config;
    config.update_config(cfg);
    kotekan::configUpdater::instance().apply_config(config);
    Telescope::instance(config);
    return config;
}

// A buffer of `num_frames` frames and a stage on it. The stage runs on this thread, where the
// logging fixture turns an error into a FatalError.
struct Pipeline {
    std::shared_ptr<metadataPool> pool;
    Buffer buf;
    kotekan::bufferContainer buffers;

    Pipeline() :
        pool(metadataPool::create(2 * num_frames, sizeof(chordMetadata), "pool", "chordMetadata")),
        buf(num_frames, T * F, pool, "buf", "standard", 0, false, false, std::vector<int>{}, true) {
        // Force metadata registration from the static library.
        chordMetadata metadata;
        buf.ensure_frame_desc(frame_desc());
        buffers.add_buffer("buf", &buf);
    }

    std::unique_ptr<kotekan::Stage> create(kotekan::Config& config, const std::string& name,
                                           const std::string& stage) {
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        return std::unique_ptr<kotekan::Stage>(
            makers.at(stage)->create(config, "/" + name, buffers));
    }
};

static int8_t sample(int frame, int i) {
    return int8_t(frame * 16 + i);
}

// Frames as two senders interleave them in one buffer: each sender has its own frequencies,
// both send the same windows, and the second sender is one frame late at the end, so a frame
// can start before the one in front of it. `extra_from` is the first frame that carries a key
// the earlier frames do not have.
static std::vector<json> write_frames(const fs::path& dir, const std::string& name,
                                      const json& options, int extra_from = num_frames) {
    json writer_options = {{"record_per_frame_metadata", true}, {"max_frames", num_frames}};
    writer_options.update(options);
    kotekan::Config config = stage_config(name, "hdf5FileWrite", dir, writer_options);

    Pipeline p;
    p.buf.register_producer("test");
    // The writer has to be a consumer before the frames are marked full.
    std::unique_ptr<kotekan::Stage> writer = p.create(config, name, "hdf5FileWrite");
    const std::vector<int64_t> window = {0, 0, 1, 2, 1, 2};
    std::vector<json> written;
    for (int i = 0; i < num_frames; i++) {
        uint8_t* frame = p.buf.wait_for_empty_frame("test", i);
        BOOST_REQUIRE(frame != nullptr);
        for (int j = 0; j < T * F; j++)
            frame[j] = uint8_t(sample(i, j));
        p.buf.allocate_new_metadata_object(i);
        auto meta = get_chord_metadata(&p.buf, i);
        meta->set_from_frame_desc(frame_desc());
        meta->set_fpga_seq_num((int64_t(1) << 40) + window[i] * T * period);
        meta->set_time_downsampling_fpga(period);
        meta->set_coarse_freq(i % 2 == 0 ? std::vector<int>{10, 12, 14, 16}
                                         : std::vector<int>{11, 13, 15, 17});
        meta->set_rfi_frame_excision_enabled(i % 2 == 0);
        meta->set_rfi_frame_excision_thresholds({{3.0f, 0.5f}, {4.0f, 0.25f * i}});
        meta->set_first_packet_recv_time(timeval{1000 + i, 20 * i});
        if (i >= extra_from)
            meta->set_lost_timesamples(i);
        written.push_back(meta->to_json());
        p.buf.mark_frame_full("test", i);
    }
    writer->main_thread();
    return written;
}

// Replay the files and compare the first `count` frames, which must be all that are read, with
// what was written.
static void check_replay(const fs::path& dir, const std::string& name, const json& options,
                         const std::vector<json>& written, int count = num_frames) {
    kotekan::Config config = stage_config(name, "hdf5FileRead", dir, options);
    Pipeline p;
    p.buf.register_consumer("test");
    p.create(config, name, "hdf5FileRead")->main_thread();

    BOOST_REQUIRE_EQUAL(p.buf.get_num_full_frames(), count);
    for (int i = 0; i < count; i++) {
        const uint8_t* frame = p.buf.wait_for_full_frame("test", i);
        BOOST_REQUIRE(frame != nullptr);
        for (int j = 0; j < T * F; j++)
            BOOST_CHECK_EQUAL(int8_t(frame[j]), sample(i, j));
        BOOST_CHECK_EQUAL(get_chord_metadata(&p.buf, i)->to_json(), written[i]);
        p.buf.mark_frame_empty("test", i);
    }
}

BOOST_AUTO_TEST_CASE(test_frames_per_file) {
    tmp_dir dir;
    const std::vector<json> written = write_frames(dir.path, "write", {{"frames_per_file", 4}});

    const std::set<std::string> expected = {"mask.00000000.h5", "mask.00000004.h5"};
    const std::set<std::string> found = list_dir(dir.path);
    BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(), expected.begin(), expected.end());

    // The first file holds four frames, although no frame continues the one before it.
    const HighFive::File file((dir.path / "mask.00000000.h5").string(), HighFive::File::ReadOnly);
    BOOST_CHECK_EQUAL(file.getDataSet("mask").getDimensions().at(0), 4 * T);
    const std::string group = std::string(hdf5::frame_metadata_group) + "/";
    const auto seq = file.getDataSet(group + "FPGA_SEQ_NUM").read<std::vector<int64_t>>();
    const std::vector<int64_t> expected_seq = {int64_t(1) << 40, int64_t(1) << 40,
                                               (int64_t(1) << 40) + T * period,
                                               (int64_t(1) << 40) + 2 * T * period};
    BOOST_CHECK_EQUAL_COLLECTIONS(seq.begin(), seq.end(), expected_seq.begin(), expected_seq.end());
    const auto freq =
        file.getDataSet(group + "COARSE_FREQ").read<std::vector<std::vector<int64_t>>>();
    BOOST_REQUIRE_EQUAL(freq.size(), 4);
    BOOST_CHECK_EQUAL(freq[0][0], 10);
    BOOST_CHECK_EQUAL(freq[1][0], 11);
    BOOST_CHECK(file.exist(group + "FIRST_PACKET_RECV_TIME/TV_SEC"));

    check_replay(dir.path, "read", {}, written);
}

BOOST_AUTO_TEST_CASE(test_new_file_for_other_metadata) {
    tmp_dir dir;
    const std::vector<json> written =
        write_frames(dir.path, "write_other", {{"frames_per_file", 4}}, 3);

    // Frame 3 has a key the first file has no dataset for, so it starts a file.
    const std::set<std::string> expected = {"mask.00000000.h5", "mask.00000003.h5"};
    const std::set<std::string> found = list_dir(dir.path);
    BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(), expected.begin(), expected.end());

    check_replay(dir.path, "read_other", {}, written);
}

BOOST_AUTO_TEST_CASE(test_single_file) {
    tmp_dir dir;
    const std::vector<json> written =
        write_frames(dir.path, "write_single", {{"create_single_file", true}});

    const std::set<std::string> expected = {"mask.h5"};
    const std::set<std::string> found = list_dir(dir.path);
    BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(), expected.begin(), expected.end());

    check_replay(dir.path, "read_single", {{"read_single_file", true}}, written);

    // A single file cannot take a frame with other metadata.
    tmp_dir dir2;
    BOOST_CHECK_THROW(
        write_frames(dir2.path, "write_single_other", {{"create_single_file", true}}, 3),
        FatalError);
}

BOOST_AUTO_TEST_CASE(test_unequal_columns) {
    tmp_dir dir;
    const std::vector<json> written =
        write_frames(dir.path, "write_unequal", {{"create_single_file", true}});

    // As a writer leaves the file when it is killed while it appends the last frame.
    {
        HighFive::File file((dir.path / "mask.h5").string(), HighFive::File::ReadWrite);
        file.getDataSet(std::string(hdf5::frame_metadata_group) + "/FPGA_SEQ_NUM")
            .resize({num_frames - 1});
    }

    check_replay(dir.path, "read_unequal", {{"read_single_file", true}}, written, num_frames - 1);
}

BOOST_AUTO_TEST_CASE(test_one_frame_per_file) {
    tmp_dir dir;
    const std::vector<json> written = write_frames(dir.path, "write_each", json::object());
    BOOST_CHECK_EQUAL(list_dir(dir.path).size(), num_frames);
    check_replay(dir.path, "read_each", {}, written);
}
