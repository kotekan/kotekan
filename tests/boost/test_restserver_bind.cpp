#define BOOST_TEST_MODULE "test_restserver_bind"

#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "restClient.hpp"            // for restClient
#include "restServer.hpp"            // for restServer

#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK, BOOST_FIXTURE_TEST_CASE
#include <chrono>                            // for milliseconds
#include <string>                            // for string
#include <thread>                            // for sleep_for
#include <vector>                            // for vector


using kotekan::restServer;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

// One server bound on two loopback addresses answers /endpoints on both.
BOOST_AUTO_TEST_CASE(two_addresses_one_port) {
    const std::vector<std::string> addresses{"127.0.0.1", "127.0.0.2"};
    restServer::instance().start(addresses, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const unsigned short port = restServer::instance().port();
    BOOST_CHECK(port != 0);
    BOOST_CHECK_EQUAL(restServer::instance().bind_address(), "127.0.0.1,127.0.0.2");

    for (const std::string& address : addresses) {
        const auto reply =
            restClient::instance().make_request_blocking("/endpoints", {}, address, port);
        BOOST_CHECK_MESSAGE(reply.first, "no reply on " + address);
        BOOST_CHECK(reply.second.find("/endpoints") != std::string::npos);
    }
}
