#define BOOST_TEST_MODULE "test_bufferRecv"

#include "Config.hpp"          // for Config
#include "N2Metadata.hpp"      // for N2Metadata
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "bufferFactory.hpp"   // for bufferFactory
#include "bufferRecv.hpp"      // for bufferRecv
#include "bufferSend.hpp"      // for bufferFrameHeaderNoConfigTracker
#include "chordMetadata.hpp"   // for chordMetadata
#include "kotekanLogging.hpp"  // for FatalError
#include "metadata.hpp"        // for metadataPool

#include "json.hpp" // for json

#include <algorithm>                         // for fill
#include <arpa/inet.h>                       // for htonl, htons, ntohs
#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK, BOOST_REQUIRE
#include <chrono>                            // for milliseconds
#include <csignal>                           // for signal, SIGTERM, SIG_IGN
#include <map>                               // for map
#include <memory>                            // for shared_ptr, unique_ptr
#include <netinet/in.h>                      // for sockaddr_in, INADDR_ANY, INADDR_LOOPBACK
#include <stdint.h>                          // for uint8_t, uint16_t, uint32_t
#include <string>                            // for string
#include <sys/socket.h>                      // for socket, bind, connect, send, recv
#include <sys/time.h>                        // for timeval
#include <thread>                            // for sleep_for
#include <time.h>                            // for clock_gettime, timespec
#include <unistd.h>                          // for close
#include <vector>                            // for vector

using kotekan::bufferContainer;
using kotekan::bufferFactory;
using kotekan::Config;
using json = nlohmann::json;

// No kotekan_logging_fixture: it fails the test on an ERROR logged from another thread, and the
// stage's worker thread logs one for each frame it refuses.

// FATAL_ERROR raises SIGTERM (exit_kotekan) before it throws. The refusal tests check the throw,
// so the signal is ignored for the life of the module.
struct ignore_sigterm {
    ignore_sigterm() {
        std::signal(SIGTERM, SIG_IGN);
    }
};
BOOST_GLOBAL_FIXTURE(ignore_sigterm);

static const int frame_size = 256;
static const int num_frames = 4;

// A port nothing listens on. The stage listens on a configured port, so it cannot be given an
// ephemeral one directly.
static uint16_t free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    BOOST_REQUIRE(fd >= 0);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = 0;
    BOOST_REQUIRE(bind(fd, (sockaddr*)&addr, sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    BOOST_REQUIRE(getsockname(fd, (sockaddr*)&addr, &len) == 0);
    close(fd);
    return ntohs(addr.sin_port);
}

static json recv_config(const std::string& name) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg[name] = {{"kotekan_stage", "bufferRecv"},
                 {"buf", "recv_buf"},
                 {"use_config_tracker", false},
                 {"allow_short_frames", true}};
    return cfg;
}

static void send_all(int fd, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    while (len > 0) {
        const ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        BOOST_REQUIRE(n > 0);
        p += n;
        len -= (size_t)n;
    }
}

// A bufferRecv with allow_short_frames on a standard buffer, and one connection to it.
struct shortFrameReceiver {
    std::shared_ptr<metadataPool> pool =
        metadataPool::create(2 * num_frames, sizeof(N2Metadata), "n2_pool", "N2Metadata");
    Buffer buf{num_frames, frame_size, pool,  "recv_buf",         "standard",
               0,          false,      false, std::vector<int>{}, true};
    Config config;
    bufferContainer bc;
    std::unique_ptr<bufferRecv> stage;
    size_t metadata_size = pool->request_metadata_object()->get_serialized_size();
    int conn = -1;

    explicit shortFrameReceiver(const std::string& name) {
        const uint16_t port = free_port();
        json cfg = recv_config(name);
        cfg[name]["listen_port"] = port;
        config.update_config(cfg);
        buf.register_consumer("test");
        bc.add_buffer("recv_buf", &buf);
        stage = std::make_unique<bufferRecv>(config, "/" + name, bc);
        stage->start();

        // The stage opens its listener in its own thread, after start() returns.
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(port);
        for (int i = 0; i < 50 && conn < 0; i++) {
            conn = socket(AF_INET, SOCK_STREAM, 0);
            BOOST_REQUIRE(conn >= 0);
            if (connect(conn, (sockaddr*)&addr, sizeof(addr)) != 0) {
                close(conn);
                conn = -1;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        BOOST_REQUIRE(conn >= 0);
    }

    ~shortFrameReceiver() {
        close(conn);
        stage->stop();
        buf.send_shutdown_signal();
        stage->join();
    }

    void send_header(uint32_t size) {
        const bufferFrameHeaderNoConfigTracker header = {(uint32_t)metadata_size, size};
        send_all(conn, &header, sizeof(header));
    }

    // Sends a frame of `size` bytes, each set to `value`, with zeroed metadata.
    void send_frame(uint32_t size, uint8_t value) {
        send_header(size);
        const std::vector<uint8_t> metadata(metadata_size, 0);
        send_all(conn, metadata.data(), metadata.size());
        const std::vector<uint8_t> frame(size, value);
        send_all(conn, frame.data(), frame.size());
    }

    // Waits up to 5 s for frame `frame_id` to be full, and returns a copy of it.
    std::vector<uint8_t> consume(int frame_id) {
        timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += 5;
        BOOST_REQUIRE_EQUAL(buf.wait_for_full_frame_timeout("test", frame_id, timeout), 0);
        const uint8_t* frame = buf.wait_for_full_frame("test", frame_id);
        const std::vector<uint8_t> data(frame, frame + frame_size);
        buf.mark_frame_empty("test", frame_id);
        return data;
    }

    // True if the stage closes the connection within 5 s.
    bool connection_closed() {
        const timeval timeout = {5, 0};
        BOOST_REQUIRE(setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);
        uint8_t byte;
        return recv(conn, &byte, 1, 0) == 0;
    }
};

BOOST_AUTO_TEST_CASE(test_short_frame_tail_is_zeroed) {
    shortFrameReceiver r("recv_short");

    // The stage reads each frame into a spare frame and swaps it into the buffer, so after
    // num_frames + 1 frames the spare frame holds the first frame's data.
    const std::vector<uint8_t> full(frame_size, 0xaa);
    for (int i = 0; i <= num_frames; i++) {
        r.send_frame(frame_size, 0xaa);
        BOOST_CHECK(r.consume(i % num_frames) == full);
    }

    const int short_size = frame_size / 4;
    r.send_frame(short_size, 0x55);
    std::vector<uint8_t> expected(frame_size, 0);
    std::fill(expected.begin(), expected.begin() + short_size, 0x55);
    BOOST_CHECK(r.consume((num_frames + 1) % num_frames) == expected);
}

// The next two send only the header: the stage must refuse the frame on its size alone.

BOOST_AUTO_TEST_CASE(test_frame_larger_than_buffer_closes_connection) {
    shortFrameReceiver r("recv_long");
    r.send_header(frame_size + 1);
    BOOST_CHECK(r.connection_closed());
    BOOST_CHECK_EQUAL(r.buf.get_num_full_frames(), 0);
}

BOOST_AUTO_TEST_CASE(test_empty_frame_closes_connection) {
    shortFrameReceiver r("recv_empty");
    r.send_header(0);
    BOOST_CHECK(r.connection_closed());
    BOOST_CHECK_EQUAL(r.buf.get_num_full_frames(), 0);
}

static bool is_short_frames_refusal(const FatalError& e) {
    return std::string(e.what()).find("allow_short_frames cannot be used") != std::string::npos;
}

BOOST_AUTO_TEST_CASE(test_refused_with_use_frame_desc) {
    json cfg = recv_config("refuse_frame_desc");
    cfg["refuse_frame_desc"]["use_frame_desc"] = true;
    Config config;
    config.update_config(cfg);
    auto pool = metadataPool::create(num_frames, sizeof(N2Metadata), "n2_pool", "N2Metadata");
    Buffer buf(num_frames, frame_size, pool, "recv_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    bufferContainer bc;
    bc.add_buffer("recv_buf", &buf);
    BOOST_CHECK_EXCEPTION(bufferRecv(config, "/refuse_frame_desc", bc), FatalError,
                          is_short_frames_refusal);
}

BOOST_AUTO_TEST_CASE(test_refused_with_chord_metadata) {
    Config config;
    config.update_config(recv_config("refuse_chord"));
    auto pool =
        metadataPool::create(num_frames, sizeof(chordMetadata), "chord_pool", "chordMetadata");
    Buffer buf(num_frames, frame_size, pool, "recv_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    bufferContainer bc;
    bc.add_buffer("recv_buf", &buf);
    BOOST_CHECK_EXCEPTION(bufferRecv(config, "/refuse_chord", bc), FatalError,
                          is_short_frames_refusal);
}

BOOST_AUTO_TEST_CASE(test_refused_with_declared_frame_desc) {
    // The buffer factory attaches the descriptor of an ndarray buffer before any stage is built.
    json cfg = recv_config("refuse_ndarray");
    cfg["recv_buf"] = {{"kotekan_buffer", "ndarray"},
                       {"value_type", "float32"},
                       {"extents", json::array({frame_size / 4})},
                       {"dimnames", json::array({"F"})},
                       {"num_frames", num_frames},
                       {"metadata_pool", "n2_pool"}};
    Config config;
    config.update_config(cfg);
    std::map<std::string, std::shared_ptr<metadataPool>> pools;
    pools["n2_pool"] =
        metadataPool::create(num_frames, sizeof(N2Metadata), "n2_pool", "N2Metadata");
    bufferFactory factory(config, pools);
    auto buffers = factory.build_buffers();
    bufferContainer bc;
    bc.add_buffer("recv_buf", buffers.at("recv_buf"));
    BOOST_CHECK_EXCEPTION(bufferRecv(config, "/refuse_ndarray", bc), FatalError,
                          is_short_frames_refusal);
    for (auto& [name, b] : buffers)
        delete b;
}
