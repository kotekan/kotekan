#define BOOST_TEST_MODULE "test_bufferSend"

#include "Config.hpp"          // for Config
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "bufferSend.hpp"      // for bufferSend
#include "chordMetadata.hpp"   // for chordMetadata
#include "errors.h"            // for get_exit_code, ReturnCode
#include "kotekanLogging.hpp"  // for kotekanLogging, FatalError
#include "metadata.hpp"        // for metadataPool

#include "json.hpp" // for json

#include <arpa/inet.h>                       // for htonl, ntohs
#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK, BOOST_REQUIRE
#include <chrono>                            // for steady_clock, duration, milliseconds
#include <csignal>                           // for signal, SIGTERM, SIG_IGN
#include <memory>                            // for shared_ptr
#include <netinet/in.h>                      // for sockaddr_in, INADDR_LOOPBACK
#include <stdint.h>                          // for uint16_t, uint8_t
#include <string>                            // for string
#include <sys/socket.h>                      // for socket, bind, listen, accept, recv
#include <thread>                            // for sleep_for
#include <unistd.h>                          // for close
#include <vector>                            // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using json = nlohmann::json;

// No kotekan_logging_fixture: it fails the test on an ERROR logged from another thread, and the
// stage is expected to log one with its FATAL_ERROR.

// FATAL_ERROR raises SIGTERM (exit_kotekan) before it throws. The tests check the exit code it
// sets, so the signal is ignored for the life of the module.
struct ignore_sigterm {
    ignore_sigterm() {
        std::signal(SIGTERM, SIG_IGN);
    }
};
BOOST_GLOBAL_FIXTURE(ignore_sigterm);

// A TCP socket bound to an ephemeral localhost port, which listens if `listening` and otherwise
// refuses connections.
static int bind_localhost(uint16_t& port, bool listening) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    BOOST_REQUIRE(fd >= 0);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    BOOST_REQUIRE(bind(fd, (sockaddr*)&addr, sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    BOOST_REQUIRE(getsockname(fd, (sockaddr*)&addr, &len) == 0);
    port = ntohs(addr.sin_port);
    if (listening)
        BOOST_REQUIRE(listen(fd, 1) == 0);
    return fd;
}

static json sender_config(const std::string& name, uint16_t port) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg[name] = {{"kotekan_stage", "bufferSend"},
                 {"buf", "send_buf"},
                 {"server_ip", "127.0.0.1"},
                 {"server_port", port}};
    return cfg;
}

// A pool for the send buffer. The tests never allocate from it.
static std::shared_ptr<metadataPool> send_pool() {
    return metadataPool::create(4, sizeof(chordMetadata), "send_pool", "chordMetadata");
}

BOOST_AUTO_TEST_CASE(test_frame_without_metadata, *boost::unit_test::timeout(10)) {
    uint16_t port = 0;
    const int listener = bind_localhost(port, true);

    json cfg = sender_config("send_no_metadata", port);
    // Wait for the connection instead of dropping the frame while it is down.
    cfg["send_no_metadata"]["drop_frames"] = false;
    Config config;
    config.update_config(cfg);

    Buffer buf(4, 1024, send_pool(), "send_buf", "standard", 0, false, false, std::vector<int>{},
               true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("send_buf", &buf);
    bufferSend send(config, "/send_no_metadata", bc);
    send.start();

    const int conn = accept(listener, nullptr, nullptr);
    BOOST_REQUIRE(conn >= 0);

    // The producer marks the frame full without allocating a metadata object.
    BOOST_REQUIRE(buf.wait_for_empty_frame("test", 0) != nullptr);
    buf.mark_frame_full("test", 0);

    // The stage stops kotekan with a FATAL_ERROR and closes the connection without sending.
    for (int i = 0; i < 50 && get_exit_code() != ReturnCode::FATAL_ERROR; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    BOOST_CHECK(get_exit_code() == ReturnCode::FATAL_ERROR);
    BOOST_CHECK_EQUAL(
        kotekan::kotekanLogging::get_error_message(),
        "Frame send_buf[0] has no metadata object: its producer did not allocate one");
    uint8_t byte;
    BOOST_CHECK_EQUAL(recv(conn, &byte, 1, MSG_DONTWAIT), 0);

    send.stop();
    buf.send_shutdown_signal();
    send.join();
    close(conn);
    close(listener);
}

BOOST_AUTO_TEST_CASE(test_stop_during_reconnect_wait, *boost::unit_test::timeout(10)) {
    // Bound but not listening, so every connection attempt is refused.
    uint16_t port = 0;
    const int refusing = bind_localhost(port, false);

    json cfg = sender_config("send_refused", port);
    cfg["send_refused"]["reconnect_time"] = 30;
    Config config;
    config.update_config(cfg);

    Buffer buf(4, 1024, send_pool(), "send_buf", "standard", 0, false, false, std::vector<int>{},
               true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("send_buf", &buf);
    bufferSend send(config, "/send_refused", bc);
    send.start();

    // Let the first attempt fail and the reconnect wait begin.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    const auto stop_start = std::chrono::steady_clock::now();
    send.stop();
    buf.send_shutdown_signal();
    send.join();
    const double stop_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stop_start).count();
    BOOST_CHECK_LT(stop_seconds, 3.0);

    close(refusing);
}

// test_frame_without_metadata has already set the FATAL_ERROR exit code, so this case checks the
// thrown FatalError and its message.
BOOST_AUTO_TEST_CASE(test_buffer_without_metadata_pool, *boost::unit_test::timeout(10)) {
    Config config;
    config.update_config(sender_config("send_no_pool", 11024));

    Buffer buf(4, 1024, nullptr, "send_buf", "standard", 0, false, false, std::vector<int>{}, true);
    bufferContainer bc;
    bc.add_buffer("send_buf", &buf);
    BOOST_CHECK_THROW(bufferSend(config, "/send_no_pool", bc), FatalError);
    BOOST_CHECK_EQUAL(
        kotekan::kotekanLogging::get_error_message(),
        "Buffer send_buf has no metadata_pool: every frame sent needs a metadata object");
}
