#define BOOST_TEST_MODULE "test_restClient"

#include <boost/test/included/unit_test.hpp>
#include <unistd.h>

#include "restServer.hpp"
#include "restClient.hpp"
#include "mongoose.h"
#include "errors.h"

using json = nlohmann::json;

using namespace std;

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
        vector<uint32_t> array;
        bool flag;
        try {
            array = json_request["array"].get<vector<uint32_t>>();
            flag = json_request["flag"].get<bool>();
        } catch (...) {
            INFO("Couldn't parse array parameter.");
            con.send_error("Couldn't parse array parameter.",
                            HTTP_RESPONSE::BAD_REQUEST);
            usleep(500);
            return;
        }

        con.send_empty_reply(HTTP_RESPONSE::OK);
    }

    void callback_text(connectionInstance& con, json json_request) {
        INFO("callback received json: %s", json_request.dump().c_str());
        vector<uint32_t> array;
        bool flag;
        try {
            array = json_request["array"].get<vector<uint32_t>>();
            flag = json_request["flag"].get<bool>();
        } catch (...) {
            INFO("Couldn't parse array parameter.");
            con.send_error("Couldn't parse array parameter.",
                            HTTP_RESPONSE::BAD_REQUEST);
            usleep(500);
            return;
        }

        con.send_text_reply("this is a test", HTTP_RESPONSE::OK);
    }
};

BOOST_FIXTURE_TEST_CASE( _send_json, TestContext ) {
    __log_level = 4;
    __enable_syslog = 0;
    struct restReply ret;

    TestContext::init(std::bind(&TestContext::callback_text, this,
                          placeholders::_1,
                          placeholders::_2));

    ret = TestContext::client.send_json("localhost:12048/test_restclient",
                                        &(TestContext::request));
    BOOST_ASSERT(ret.success == true);
    INFO("...response: %s", (char*)ret.data);

    json bad_request;
    ret = TestContext::client.send_json("localhost:12048/test_restclient",
                                        &bad_request);
    BOOST_ASSERT(ret.success == false);

    bad_request["array"] = 0;
    ret = TestContext::client.send_json("localhost:12048/test_restclient",
                                        &bad_request);
    BOOST_ASSERT(ret.success == false);

    ret = TestContext::client.send_json("localhost:12048/doesntexist",
                                        &(TestContext::request));
    BOOST_ASSERT(ret.success == false);

    ret = TestContext::client.send_json("localhost:0000/doesntexist",
                                        &(TestContext::request));
    BOOST_ASSERT(ret.success == false);
}
