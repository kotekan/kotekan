#define BOOST_TEST_MODULE "test_rawFileWrite"

#include "Config.hpp"                // for Config
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "rawFileWrite.hpp"          // for rawFileWrite

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK, BOOST_REQUIRE
#include <filesystem>                        // for path, directory_iterator, file_size
#include <fstream>                           // for ifstream, ofstream
#include <iterator>                          // for istreambuf_iterator
#include <set>                               // for set
#include <stdint.h>                          // for uint32_t
#include <stdlib.h>                          // for mkdtemp
#include <string>                            // for string
#include <system_error>                      // for error_code
#include <unistd.h>                          // for gethostname
#include <vector>                            // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using json = nlohmann::json;
namespace fs = std::filesystem;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

static const size_t frame_size = 64;

// A new directory under /tmp, removed at the end of the test case.
struct tmp_dir {
    fs::path path;
    tmp_dir() {
        char name[] = "/tmp/test_rawFileWrite_XXXXXX";
        BOOST_REQUIRE(mkdtemp(name) != nullptr);
        path = name;
    }
    ~tmp_dir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

static void write_file(const fs::path& path, const std::string& contents) {
    std::ofstream(path) << contents;
}

static std::string read_file(const fs::path& path) {
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static std::set<std::string> list_dir(const fs::path& dir) {
    std::set<std::string> names;
    for (const auto& entry : fs::directory_iterator(dir))
        names.insert(entry.path().filename().string());
    return names;
}

// A config for one rawFileWrite named `name`. Each writer needs its own name, since the stage
// registers a metric under it.
static json writer_config(const std::string& name, const fs::path& base_dir) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg[name] = {{"kotekan_stage", "rawFileWrite"},
                 {"in_buf", "in_buf"},
                 {"base_dir", base_dir.string()},
                 {"file_name", "x"},
                 {"file_ext", "dat"},
                 {"prefix_hostname", false}};
    return cfg;
}

// Run the rawFileWrite `name` on `num_files` frames, one frame per file, until it exits. It runs
// on this thread, where the logging fixture turns an error into a FatalError.
static void write_files(json cfg, const std::string& name, uint32_t num_files) {
    cfg[name]["exit_after_n_files"] = num_files;
    Config config;
    config.update_config(cfg);

    Buffer buf(num_files, frame_size, nullptr, "in_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("in_buf", &buf);
    rawFileWrite writer(config, "/" + name, bc);
    for (uint32_t i = 0; i < num_files; i++) {
        BOOST_REQUIRE(buf.wait_for_empty_frame("test", i) != nullptr);
        buf.mark_frame_full("test", i);
    }
    writer.main_thread();
}

BOOST_AUTO_TEST_CASE(test_numbering_default) {
    tmp_dir dir;
    write_file(dir.path / "x_0000004.dat", "old");

    write_files(writer_config("default", dir.path), "default", 1);

    const std::set<std::string> expected = {"x_0000000.dat", "x_0000004.dat"};
    const std::set<std::string> found = list_dir(dir.path);
    BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(test_continue_numbering) {
    tmp_dir dir;
    // Only x_<7 digits>.dat counts. Each of the other names would give a higher number or
    // fail to parse as one.
    const std::set<std::string> seeded = {"x_0000003.dat", "x_0000004.dat",  "y_0000009.dat",
                                          "x_0000009.raw", "x_00000099.dat", "x_9999a99.dat",
                                          "x_abcdefg.dat", "x_.dat"};
    for (const std::string& name : seeded)
        write_file(dir.path / name, "old");

    json cfg = writer_config("continue", dir.path);
    cfg["continue"]["continue_numbering"] = true;
    write_files(cfg, "continue", 2);

    std::set<std::string> expected = seeded;
    expected.insert({"x_0000005.dat", "x_0000006.dat"});
    const std::set<std::string> found = list_dir(dir.path);
    BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(), expected.begin(), expected.end());
    for (const char* name : {"x_0000005.dat", "x_0000006.dat"})
        if (fs::exists(dir.path / name))
            BOOST_CHECK_EQUAL(fs::file_size(dir.path / name), sizeof(uint32_t) + frame_size);
    BOOST_CHECK_EQUAL(read_file(dir.path / "x_0000004.dat"), "old");
}

BOOST_AUTO_TEST_CASE(test_continue_numbering_past_7_digits) {
    tmp_dir dir;
    // Past 9999999 the names have more digits. Names the stage does not write are ignored: a
    // leading zero past 7 digits, more than 10 digits, or a number above UINT32_MAX.
    const std::set<std::string> seeded = {"x_9999999.dat",    "x_10000000.dat",
                                          "x_010000005.dat",  "x_10000000000.dat",
                                          "x_4294967296.dat", "x_99999999999999999999.dat"};
    for (const std::string& name : seeded)
        write_file(dir.path / name, "old");

    json cfg = writer_config("continue_long", dir.path);
    cfg["continue_long"]["continue_numbering"] = true;
    write_files(cfg, "continue_long", 1);

    std::set<std::string> expected = seeded;
    expected.insert("x_10000001.dat");
    const std::set<std::string> found = list_dir(dir.path);
    BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(), expected.begin(), expected.end());
    BOOST_CHECK_EQUAL(read_file(dir.path / "x_10000000.dat"), "old");
}

BOOST_AUTO_TEST_CASE(test_continue_numbering_hostname) {
    char hostname[64];
    BOOST_REQUIRE(gethostname(hostname, sizeof(hostname)) == 0);
    const std::string host = hostname;

    tmp_dir dir;
    write_file(dir.path / (host + "_x_0000004.dat"), "old");
    write_file(dir.path / "x_0000009.dat", "old");

    json cfg = writer_config("continue_host", dir.path);
    cfg["continue_host"]["prefix_hostname"] = true;
    cfg["continue_host"]["continue_numbering"] = true;
    write_files(cfg, "continue_host", 1);

    const std::set<std::string> expected = {host + "_x_0000004.dat", host + "_x_0000005.dat",
                                            "x_0000009.dat"};
    const std::set<std::string> found = list_dir(dir.path);
    BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(), expected.begin(), expected.end());
}

static bool exhausted(const FatalError& e) {
    return std::string(e.what()).find("are exhausted") != std::string::npos;
}

BOOST_AUTO_TEST_CASE(test_continue_numbering_exhausted) {
    // base_dir already holds the last number, so there is no next one.
    tmp_dir dir;
    write_file(dir.path / "x_4294967295.dat", "old");
    json cfg = writer_config("exhausted_at_start", dir.path);
    cfg["exhausted_at_start"]["continue_numbering"] = true;
    BOOST_CHECK_EXCEPTION(write_files(cfg, "exhausted_at_start", 1), FatalError, exhausted);
    BOOST_CHECK_EQUAL(list_dir(dir.path).size(), 1);
    BOOST_CHECK_EQUAL(read_file(dir.path / "x_4294967295.dat"), "old");

    // The run writes the last number, and the next file would wrap to file 0.
    tmp_dir dir2;
    write_file(dir2.path / "x_4294967294.dat", "old");
    write_file(dir2.path / "x_0000000.dat", "old");
    cfg = writer_config("exhausted_in_run", dir2.path);
    cfg["exhausted_in_run"]["continue_numbering"] = true;
    BOOST_CHECK_EXCEPTION(write_files(cfg, "exhausted_in_run", 2), FatalError, exhausted);
    const std::set<std::string> expected = {"x_0000000.dat", "x_4294967294.dat",
                                            "x_4294967295.dat"};
    const std::set<std::string> found = list_dir(dir2.path);
    BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(), expected.begin(), expected.end());
    BOOST_CHECK_EQUAL(read_file(dir2.path / "x_0000000.dat"), "old");
}

static bool cannot_list(const FatalError& e) {
    return std::string(e.what()).find("Cannot list base_dir") != std::string::npos;
}

BOOST_AUTO_TEST_CASE(test_continue_numbering_missing_base_dir) {
    tmp_dir dir;
    json cfg = writer_config("continue_missing", dir.path / "missing");
    cfg["continue_missing"]["continue_numbering"] = true;
    BOOST_CHECK_EXCEPTION(write_files(cfg, "continue_missing", 1), FatalError, cannot_list);
}

// Construct the rawFileWrite `name` without starting it.
static void construct_writer(const json& cfg, const std::string& name) {
    Config config;
    config.update_config(cfg);

    Buffer buf(1, frame_size, nullptr, "in_buf", "standard", 0, false, false, std::vector<int>{},
               true);
    bufferContainer bc;
    bc.add_buffer("in_buf", &buf);
    rawFileWrite writer(config, "/" + name, bc);
}

BOOST_AUTO_TEST_CASE(test_create_base_dir) {
    tmp_dir dir;
    json cfg = writer_config("create", dir.path / "a" / "b");
    construct_writer(cfg, "create");
    BOOST_CHECK(!fs::exists(dir.path / "a"));

    cfg["create"]["create_base_dir"] = true;
    construct_writer(cfg, "create");
    BOOST_CHECK(fs::is_directory(dir.path / "a" / "b"));

    // A base_dir that cannot be created fails at construction.
    write_file(dir.path / "file", "");
    cfg = writer_config("create", dir.path / "file" / "b");
    cfg["create"]["create_base_dir"] = true;
    BOOST_CHECK_THROW(construct_writer(cfg, "create"), FatalError);
}
