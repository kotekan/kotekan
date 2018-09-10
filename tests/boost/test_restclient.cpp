#define BOOST_TEST_MODULE "test_restClient"

#include <boost/test/included/unit_test.hpp>
#include <unistd.h>

#include "restServer.hpp"
#include "restClient.hpp"
#include "errors.h"

using json = nlohmann::json;

struct TestContext {
    restClient client;
    json request;

    void init(std::function<void (connectionInstance&, nlohmann::json&)> fun) {
        restServer::instance().register_post_callback("/test_restclient", fun);

        usleep(3000);
        client = restClient();

        request["array"] = {1,2,3};
        request["flag"] = true;
    }

    void callback(connectionInstance& con, json json_request) {
        INFO("callback received json: %s", json_request.dump().c_str());
        std::vector<uint32_t> array;
        bool flag;
        try {
            array = json_request["array"].get<std::vector<uint32_t>>();
            flag = json_request["flag"].get<bool>();
            INFO("test: Received array with size %d and flag %d",
                 array.size(), flag);
        } catch (...) {
            INFO("test: Couldn't parse array parameter.");
            con.send_error("Couldn't parse array parameter.",
                            HTTP_RESPONSE::BAD_REQUEST);
            usleep(500);
            return;
        }

        con.send_empty_reply(HTTP_RESPONSE::OK);
        INFO("test: Response OK sent.");
    }

    void callback_text(connectionInstance& con, json json_request) {
        INFO("test (text): callback received json: %s",
             json_request.dump().c_str());
        std::vector<uint32_t> array;
        bool flag;
        try {
            array = json_request["array"].get<std::vector<uint32_t>>();
            flag = json_request["flag"].get<bool>();
            INFO("test (text): Received array with size %d and flag %d",
                 array.size(), flag);
        } catch (...) {
            INFO("test (text): Couldn't parse array parameter.");
            con.send_error("Couldn't parse array parameter.",
                            HTTP_RESPONSE::BAD_REQUEST);
            usleep(500);
            return;
        }

        if (flag == true && array[0] == 1 && array[1] == 2 && array[2] == 3)
            con.send_text_reply("this is a test", HTTP_RESPONSE::OK);
        else {
            json j;
            j["test"] = "failed";
            con.send_json_reply(j);
            INFO("test: sending back json: %s", j.dump().c_str());
         }
    }

    void pong(connectionInstance& con, json json_request) {
        INFO("pong: json: %s", json_request.dump().c_str());
        con.send_json_reply(json_request);
    }

    void check(json request) {
        restClient thread_client = restClient();
        std::unique_ptr<struct restReply> ret = thread_client.send("test_restclient",
                                                              request);
        BOOST_CHECK(ret->success == true);

        if (ret->success) {
            BOOST_CHECK(ret->data != nullptr);
            json js = json::parse(std::string((char*)ret->data, ret->datalen));

            DEBUG("Comparing %s with %s", js.dump().c_str(), request.dump().c_str());
            BOOST_CHECK(js == request);
        }
    }
};

BOOST_FIXTURE_TEST_CASE( _send_json, TestContext ) {
    BOOST_CHECKPOINT("Start.");
    __log_level = 4;
    __enable_syslog = 0;
    std::unique_ptr<struct restReply> ret;

    TestContext::init(std::bind(&TestContext::callback, this,
                          std::placeholders::_1,
                          std::placeholders::_2));
    BOOST_TEST_CHECKPOINT("Init done.");

    ret = TestContext::client.send("test_restclient", TestContext::request);
    BOOST_CHECK(ret->success == true);

    json bad_request;
    ret = TestContext::client.send("/test_restclient", bad_request);
    BOOST_CHECK(ret->success == false);

    bad_request["array"] = 0;
    ret = TestContext::client.send("/test_restclient", bad_request);
    BOOST_CHECK(ret->success == false);

    ret = TestContext::client.send("/doesntexist", TestContext::request);
    BOOST_CHECK(ret->success == false);

    ret = TestContext::client.send("/test_restclient",
                                        TestContext::request,
                                        "localhost", 1);
    BOOST_CHECK(ret->success == false);

    TestContext::init(std::bind(&TestContext::callback_text, this,
                          std::placeholders::_1,
                          std::placeholders::_2));

    ret = TestContext::client.send("test_restclient", TestContext::request);
    BOOST_CHECK(ret->success == true);
    if (ret->success)
        BOOST_CHECK(string("this is a test").compare((char*)ret->data) == 0);

    bad_request["flag"] = false;
    bad_request["array"] = {4,5,6};
    ret = TestContext::client.send("test_restclient", bad_request);
    BOOST_CHECK(ret->success == true);
    if (ret->success) {
        BOOST_CHECK(ret->data != nullptr);
        json js = json::parse(std::string((char*)ret->data, ret->datalen));
        BOOST_CHECK(js["test"] == "failed");
    }

    // override callback
    TestContext::init(std::bind(&TestContext::pong, this,
                          std::placeholders::_1,
                          std::placeholders::_2));
#define N 10
    std::thread t[N];
    for (int i = 0; i < N; i++) {
        bad_request["array"] = {i, i+1, i+2};
        check(bad_request);
    }
}
