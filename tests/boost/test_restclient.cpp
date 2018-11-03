#define BOOST_TEST_MODULE "test_restClient"

#include <boost/test/included/unit_test.hpp>
#include <unistd.h>
#include <fmt.hpp>

#include "restServer.hpp"
#include "restClient.hpp"
#include "errors.h"

using json = nlohmann::json;

// BOOST_CHECK can't be used in threads...
std::atomic<bool> error;
std::atomic<int> cb_called_count;

struct TestContext {
    static void init(std::function<void (connectionInstance&,
                                         nlohmann::json&)> fun,
                     std::string endpoint = "/test_restclient") {
        restServer::instance().register_post_callback(endpoint, fun);
        error = false;
        cb_called_count = 0;
        usleep(5000);
    }

    static void rq_callback(restReply reply) {
        if(reply.first != true) {
            error = true;
            ERROR("test_restclient: rq_callback: restReply::success should be" \
                  " true, was false.");
        }
        if(!reply.second.empty()) {
            error = true;
            ERROR("test_restclient: rq_callback: restReply::string should be " \
                  "empty but was not.");
        }
    }

    static void rq_callback_fail(restReply reply) {
        if (reply.first != false) {
            error = true;
            ERROR("test_restclient: rq_callback_fail: restReply::success" \
                  " should be true, was false.");
        }
        if(!reply.second.empty()) {
            error = true;
            ERROR("test_restclient: rq_callback_fail: restReply::string " \
                  "should be empty but was not.");
        }
    }

    static void rq_callback_thisisatest(restReply reply) {
        if (reply.first != true) {
            error = true;
            ERROR("test_restclient: rq_callback_thisisatest: restReply::" \
                  "success should be true, was false.");
        }
        if (reply.second != "this is a test") {
            error = true;
            ERROR("test_restclient: rq_callback_thisisatest: restReply::" \
                  "string should be 'this is a test', but was '%s'.",
                  reply.second.c_str());
        }
    }

    static void rq_callback_json(restReply reply) {
        if (reply.first != true) {
            error = true;
            ERROR("test_restclient: rq_callback_json: restReply::" \
                  "success should be true, was false.");
        }
        json js = json::parse(reply.second);
        if (js["test"] != "failed") {
            error = true;
            ERROR("test_restclient: rq_callback_json: json value" \
                  "'test' should be 'failed', but was '%s'.",
                  js["test"]);
        }
    }

    static void rq_callback_pong(restReply reply) {
        json request;
        request["array"] = {1,2,3};
        request["flag"] = true;

        if (reply.second.empty() != false) {
            error = true;
            ERROR("test_restclient: rq_callback_pong: restReply::" \
                  "string was empty.");
        }
        json js = json::parse(reply.second);

        if (js != request) {
            error = true;
            ERROR("test_restclient: rq_callback_thisisatest: restReply::" \
                  "string should be '%s', but was '%s'.",
                  request.dump().c_str(), js.dump().c_str());
        }
    }

    void callback(connectionInstance& con, json json_request) {
        cb_called_count++;
        INFO("test_restclient: json callback received json: %s",
             json_request.dump().c_str());
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
            return;
        }

        con.send_empty_reply(HTTP_RESPONSE::OK);
        INFO("test: Response OK sent.");
    }

    void callback_text(connectionInstance& con, json json_request) {
        cb_called_count++;
        INFO("test_restclient: text callback received json: %s",
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
        cb_called_count++;
        INFO("pong: json: %s", json_request.dump().c_str());
        con.send_json_reply(json_request);
    }

    static void check() {
        json request;
        request["array"] = {1,2,3};
        request["flag"] = true;

        std::function<void(restReply)> fun = TestContext::rq_callback_pong;
        std::string path("test_restclient_pong");
        bool ret = restClient::instance().make_request(path,
                                                       fun,
                                                       request);
        if (ret != true) {
            INFO("test: restClient::make_request failed.");
            error = true;
        }
    }
};

BOOST_FIXTURE_TEST_CASE( _test_restclient_send_json, TestContext ) {
    BOOST_CHECKPOINT("Start.");
    __log_level = 4;
    __enable_syslog = 0;
    bool ret;
    json request;
    request["array"] = {1,2,3};
    request["flag"] = true;

    TestContext::init(std::bind(&TestContext::callback, this,
                          std::placeholders::_1,
                          std::placeholders::_2));
    BOOST_CHECKPOINT("Init done.");

    std::function<void(restReply)> fun = TestContext::rq_callback;
    ret = restClient::instance().make_request("test_restclient", fun,
                                              request);
    BOOST_CHECK(ret == true);
    BOOST_CHECKPOINT("Test sending json done.");


    /* Test send a bad json */

    json bad_request;
    bad_request["bla"] = 0;
    std::function<void(restReply)> fun_fail = TestContext::rq_callback_fail;
    ret = restClient::instance().make_request("test_restclient", fun_fail,
                                              bad_request);
    BOOST_CHECK(ret == true);
    BOOST_CHECKPOINT("Test sending bad json #1 done.");

    bad_request["array"] = 0;
    ret = restClient::instance().make_request("test_restclient", fun_fail,
                                              bad_request);
    BOOST_CHECK(ret == true);
    BOOST_CHECKPOINT("Test sending bad json #1 done.");


    /* Test with bad URL */

    ret = restClient::instance().make_request("doesntexist", fun_fail,
                                              request);
    BOOST_CHECK(ret == true);
    BOOST_CHECKPOINT("Test bad endpoint done.");


    ret = restClient::instance().make_request("test_restclient", fun_fail,
                                              request, "localhost", 1);
    usleep(500000);
    BOOST_CHECK_MESSAGE(error == false,
                        "Run pytest with -s to see where the error is.");
    std::string fail_msg = fmt::format("Only {} callback functions where " \
                                       "called (expected 3). This suggests " \
                                       "some requests were never sent by the " \
                                       "restClient.", cb_called_count);
    BOOST_CHECK_MESSAGE(cb_called_count == 3, fail_msg);
}

BOOST_FIXTURE_TEST_CASE( _test_restclient_text_reply, TestContext ) {
    BOOST_CHECKPOINT("Start.");
    __log_level = 4;
    __enable_syslog = 0;
    bool ret;
    json request, bad_request;
    request["array"] = {1,2,3};
    request["flag"] = true;
    bad_request["array"] = 0;


    /* Test receiveing a text reply */

    TestContext::init(std::bind(&TestContext::callback_text, this,
                          std::placeholders::_1,
                          std::placeholders::_2),
                      "/test_restclient_json");

    std::function<void(restReply)> fun_test = TestContext::rq_callback_thisisatest;
    ret = restClient::instance().make_request("test_restclient_json", fun_test,
                                              request);
    BOOST_CHECK(ret == true);
    BOOST_CHECKPOINT("Test receiving text done.");


    /* Test with json in reply */

    bad_request["flag"] = false;
    bad_request["array"] = {4,5,6};

    INFO("sending bad json to callback_test");
    std::function<void(restReply)> fun_json = TestContext::rq_callback_json;
    ret = restClient::instance().make_request("test_restclient_json", fun_json,
                                              bad_request);
    usleep(500000);
    BOOST_CHECK_MESSAGE(error == false,
                        "Run pytest with -s to see where the error is.");
    std::string fail_msg = fmt::format("Only {} callback functions where " \
                                       "called (expected 2). This suggests " \
                                       "some requests were never sent by the " \
                                       "restClient.", cb_called_count);
    BOOST_CHECK_MESSAGE(cb_called_count == 2, fail_msg);}

BOOST_FIXTURE_TEST_CASE( _test_restclient_multithr_request, TestContext ) {
    BOOST_CHECKPOINT("Start.");
    __log_level = 4;
    __enable_syslog = 0;
    json request, bad_request;
    request["array"] = {1,2,3};
    request["flag"] = true;
    bad_request["array"] = 0;


    /* Test N threads */

    TestContext::init(std::bind(&TestContext::pong, this,
                          std::placeholders::_1,
                          std::placeholders::_2),
                      "/test_restclient_pong");
#define N 100
    std::thread t[N];
    for (int i = 0; i < N; i++) {
        t[i] = std::thread(check);
    }
    for (int i = 0; i < N; i++) {
        t[i].join();
    }
    usleep(30000000);
    BOOST_CHECK_MESSAGE(error == false,
                        "Run pytest with -s to see where the error is.");
    std::string fail_msg = fmt::format("Only {} callback functions where " \
                                       "called (expected {}). This suggests " \
                                       "some requests were never sent by the " \
                                       "restClient.", cb_called_count, N);
    BOOST_CHECK_MESSAGE(cb_called_count == N, fail_msg);
}
