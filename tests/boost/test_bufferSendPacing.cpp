#define BOOST_TEST_MODULE "test_bufferSendPacing"

#include "Config.hpp"                // for Config
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "bufferSend.hpp"            // for bufferSend, bufferFrameHeaderNoConfigTracker
#include "errors.h"                  // for get_exit_code, ReturnCode
#include "kotekanLogging.hpp"        // for FatalError, log_event_hook, kotekanLogging
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "metadata.hpp"              // for metadataPool
#include "metadataFactory.hpp"       // for metadataFactory

#include "fmt.hpp"  // for fmt
#include "json.hpp" // for json

#include <arpa/inet.h>                       // for htonl, ntohs
#include <atomic>                            // for atomic
#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK, BOOST_REQUIRE
#include <cerrno>                            // for errno, ENOPROTOOPT
#include <chrono>                            // for steady_clock, duration, milliseconds
#include <csignal>                           // for sigaction, sig_atomic_t, SIGTERM
#include <dlfcn.h>                           // for dlsym, RTLD_NEXT
#include <memory>                            // for shared_ptr
#include <netinet/in.h>                      // for sockaddr_in, INADDR_LOOPBACK
#include <stdint.h>                          // for uint16_t, uint8_t
#include <string>                            // for string
#include <sys/socket.h>                      // for socket, bind, listen, accept, recv
#include <sys/time.h>                        // for timeval
#include <thread>                            // for sleep_for
#include <unistd.h>                          // for close
#include <vector>                            // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using json = nlohmann::json;

// An ERROR from the stage fails the test, except in a case that takes the hook (without_log_hook).
BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

#ifdef SO_MAX_PACING_RATE
// While set, setsockopt() refuses SO_MAX_PACING_RATE, as a kernel without the option does.
static std::atomic<bool> refuse_pacing{false};

// Sets refuse_pacing for its lifetime.
struct refusing_pacing {
    refusing_pacing() {
        refuse_pacing = true;
    }
    ~refusing_pacing() {
        refuse_pacing = false;
    }
};

// Interposing the function in the executable catches bufferSend's calls too.
extern "C" int setsockopt(int fd, int level, int name, const void* value, socklen_t len) noexcept {
    using setsockopt_t = int (*)(int, int, int, const void*, socklen_t);
    static const setsockopt_t real = reinterpret_cast<setsockopt_t>(dlsym(RTLD_NEXT, "setsockopt"));
    if (refuse_pacing && level == SOL_SOCKET && name == SO_MAX_PACING_RATE) {
        errno = ENOPROTOOPT;
        return -1;
    }
    return real(fd, level, name, value, len);
}
#endif

// Takes the logging fixture's hook for its lifetime. A FATAL_ERROR then runs to its own throw,
// after exit_kotekan(), instead of stopping at the fixture's throw on the ERROR it logs first;
// and an ERROR from the stage's own threads does not fail the run.
struct without_log_hook {
    const kotekan::log_event_handler hook = kotekan::log_event_hook.exchange(nullptr);
    ~without_log_hook() {
        kotekan::log_event_hook.store(hook);
    }
};

#ifdef SO_MAX_PACING_RATE
// Records a SIGTERM, which exit_kotekan() raises, for its lifetime. The fixture ignores SIGTERM
// otherwise.
static volatile sig_atomic_t got_sigterm = 0;
struct catch_sigterm {
    struct sigaction old_sigterm;
    catch_sigterm() {
        got_sigterm = 0;
        struct sigaction record = {};
        record.sa_handler = [](int) { got_sigterm = 1; };
        sigemptyset(&record.sa_mask);
        sigaction(SIGTERM, &record, &old_sigterm);
    }
    ~catch_sigterm() {
        sigaction(SIGTERM, &old_sigterm, nullptr);
    }
};

// A listening socket on a free loopback port, with accept() and recv() on it bounded by
// `timeout`, so that a stage that never connects or never sends fails the test.
static int listen_on_loopback(uint16_t& port, const timeval& timeout) {
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    BOOST_REQUIRE(listener >= 0);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    BOOST_REQUIRE(bind(listener, (sockaddr*)&addr, sizeof(addr)) == 0);
    socklen_t addr_len = sizeof(addr);
    BOOST_REQUIRE(getsockname(listener, (sockaddr*)&addr, &addr_len) == 0);
    BOOST_REQUIRE(listen(listener, 1) == 0);
    BOOST_REQUIRE(setsockopt(listener, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);
    port = ntohs(addr.sin_port);
    return listener;
}
#endif

// Stops the stage at the end of the test case. A failed BOOST_REQUIRE throws, and the stage's
// destructor hangs if its threads are still running.
struct stop_at_exit {
    bufferSend& stage;
    Buffer& buf;
    ~stop_at_exit() {
        stage.stop();
        buf.send_shutdown_signal();
        stage.join();
    }
};

static json sender_config(const std::string& name, uint16_t port, double max_pacing_rate_mbps) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg["main_pool"] = {{"kotekan_metadata_pool", "chordMetadata"}, {"num_metadata_objects", 2}};
    cfg[name] = {{"kotekan_stage", "bufferSend"},
                 {"buf", "send_buf"},
                 {"server_ip", "127.0.0.1"},
                 {"server_port", port},
                 {"use_config_tracker", false},
                 {"drop_frames", false},
                 {"max_pacing_rate_mbps", max_pacing_rate_mbps}};
    return cfg;
}

// The metadata pool for the send buffer, from the config's main_pool.
static std::shared_ptr<metadataPool> send_pool(Config& config) {
    kotekan::metadataFactory mfac(config);
    return mfac.build_pools()["main_pool"];
}

static bool names_max_pacing_rate(const FatalError& e) {
    return std::string(e.what()).find("max_pacing_rate_mbps") != std::string::npos;
}

#ifdef SO_MAX_PACING_RATE
BOOST_AUTO_TEST_CASE(test_pacing_rate) {
    // 16 Mbit/s is 2e6 bytes/s, so the frame takes about 2 s; unpaced, localhost takes a few ms.
    const double rate_mbps = 16;
    const double rate_bytes = rate_mbps * 1e6 / 8;
    const size_t frame_size = 4 << 20;

    const timeval timeout = {10, 0};
    uint16_t port;
    const int listener = listen_on_loopback(port, timeout);

    Config config;
    config.update_config(sender_config("send_paced", port, rate_mbps));
    Buffer buf(2, frame_size, send_pool(config), "send_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("send_buf", &buf);
    bufferSend send(config, "/send_paced", bc);
    send.start();
    stop_at_exit stop_send{send, buf};

    const int conn = accept(listener, nullptr, nullptr);
    BOOST_REQUIRE(conn >= 0);
    BOOST_REQUIRE(setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);

    BOOST_REQUIRE(buf.wait_for_empty_frame("test", 0) != nullptr);
    buf.allocate_new_metadata_object(0);
    const size_t message_size = sizeof(bufferFrameHeaderNoConfigTracker)
                                + buf.get_metadata(0)->get_serialized_size() + frame_size;
    buf.mark_frame_full("test", 0);

    // Time the middle of the frame. Linux sends a connection's first ten segments unpaced.
    const size_t start_bytes = 1 << 20;
    const size_t end_bytes = 3 << 20;
    size_t received = 0;
    size_t timed_start = 0, timed_end = 0;
    std::chrono::steady_clock::time_point t_start, t_end;
    std::vector<uint8_t> chunk(1 << 16);
    while (received < message_size) {
        const ssize_t n = recv(conn, chunk.data(), chunk.size(), 0);
        BOOST_REQUIRE(n > 0);
        received += n;
        if (timed_start == 0 && received >= start_bytes) {
            timed_start = received;
            t_start = std::chrono::steady_clock::now();
        }
        if (timed_end == 0 && received >= end_bytes) {
            timed_end = received;
            t_end = std::chrono::steady_clock::now();
        }
    }
    BOOST_CHECK_EQUAL(received, message_size);

    const double seconds = std::chrono::duration<double>(t_end - t_start).count();
    const double measured_bytes = (timed_end - timed_start) / seconds;
    BOOST_TEST_MESSAGE("received at " << measured_bytes << " bytes/s, paced at " << rate_bytes);
    BOOST_CHECK_GT(measured_bytes, 0.7 * rate_bytes);
    BOOST_CHECK_LT(measured_bytes, 1.3 * rate_bytes);

    close(conn);
    close(listener);
}

BOOST_AUTO_TEST_CASE(test_pacing_rate_refused) {
    Config config;
    config.update_config(sender_config("send_refused", 11024, 16));
    Buffer buf(2, 1024, send_pool(config), "send_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    bufferContainer bc;
    bc.add_buffer("send_buf", &buf);
    refusing_pacing refusing;
    without_log_hook no_hook;
    BOOST_CHECK_EXCEPTION(bufferSend(config, "/send_refused", bc), FatalError,
                          names_max_pacing_rate);
}

BOOST_AUTO_TEST_CASE(test_pacing_rate_refused_on_connect) {
    const timeval timeout = {10, 0};
    uint16_t port;
    const int listener = listen_on_loopback(port, timeout);

    Config config;
    config.update_config(sender_config("send_conn_refused", port, 16));
    Buffer buf(2, 1024, send_pool(config), "send_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("send_buf", &buf);
    // The option is refused only after construction, so the check at startup passes.
    bufferSend send(config, "/send_conn_refused", bc);
    // Declared before stop_send, so these last until the stage has stopped.
    refusing_pacing refusing;
    without_log_hook no_hook;
    catch_sigterm sigterm;
    kotekan::kotekanLogging::set_error_message(fmt("{:s}"), "");
    send.start();
    stop_at_exit stop_send{send, buf};

    const int conn = accept(listener, nullptr, nullptr);
    BOOST_REQUIRE(conn >= 0);
    BOOST_REQUIRE(setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);
    // The stage closes the connection rather than send unpaced, then stops kotekan.
    char c;
    BOOST_CHECK_EQUAL(recv(conn, &c, 1, 0), 0);
    for (int i = 0; i < 1000 && !got_sigterm; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    BOOST_CHECK(got_sigterm);
    BOOST_CHECK(get_exit_code() == ReturnCode::FATAL_ERROR);
    BOOST_CHECK(kotekan::kotekanLogging::get_error_message().find(
                    "bufferSend: setsockopt() SO_MAX_PACING_RATE failed")
                != std::string::npos);

    close(conn);
    close(listener);
}
#endif

BOOST_AUTO_TEST_CASE(test_pacing_rate_out_of_range) {
    int attempt = 0;
    // 4e-6 Mbit/s is 0.5 bytes/s, which truncates to 0, the value that leaves the socket unpaced.
    for (double rate_mbps : {-1.0, 4e-6, 40000.0}) {
        // Each stage needs its own name, since its constructor registers a metric under it.
        const std::string name = "send_bad_rate_" + std::to_string(attempt++);
        Config config;
        config.update_config(sender_config(name, 11024, rate_mbps));
        Buffer buf(2, 1024, send_pool(config), "send_buf", "standard", 0, false, false,
                   std::vector<int>{}, true);
        bufferContainer bc;
        bc.add_buffer("send_buf", &buf);
        without_log_hook no_hook;
        BOOST_CHECK_EXCEPTION(bufferSend(config, "/" + name, bc), FatalError,
                              names_max_pacing_rate);
    }
}
