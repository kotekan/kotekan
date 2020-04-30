#define BOOST_TEST_MODULE "test_config"

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_PP_IIF_0, BOOST_PP_BO...
#include <memory>                            // for allocator, allocator_traits<>::value_type
#include <vector>                            // for vector

// the code to test:
#include "Config.hpp" // for Config

#include "json.hpp" // for json_ref, json

using kotekan::Config;

using json = nlohmann::json;

BOOST_AUTO_TEST_CASE(_get_value_recursive) {
    json json_config = {
        {"pi", 3.141},
        {"truth",
         {
             {"not", true},
             {"but", false},
         }},
        {"lie", {{"not", false}, {"but", true}}},
        {"happy", true},
        {"name", "Niels"},
        {"nothing", nullptr},
        {"answer", {{"everything", 42}}},
        {"list", {1, 0, 2}},
        {"object", {{"currency", "USD"}, {"value", 42.99}, {"hidden", {{"treasure", "found"}}}}}};
    Config config;
    config.update_config(json_config);

    BOOST_CHECK(
        (config.get_value("not").at(0).get<bool>() && config.get_value("not").at(1).get<bool>())
        == false);
    BOOST_CHECK(
        (config.get_value("not").at(0).get<bool>() || config.get_value("not").at(1).get<bool>())
        == true);
    BOOST_CHECK(
        (config.get_value("but").at(0).get<bool>() && config.get_value("but").at(1).get<bool>())
        == false);
    BOOST_CHECK(
        (config.get_value("but").at(0).get<bool>() || config.get_value("but").at(1).get<bool>())
        == true);

    std::vector<json> results;

    json j = "found";
    results.push_back(j);
    BOOST_CHECK_EQUAL(config.get_value("treasure"), results);

    j = {{"treasure", "found"}};
    results.clear();
    results.push_back(j);
    BOOST_CHECK_EQUAL(config.get_value("hidden"), results);

    j = 3.141;
    results.clear();
    results.push_back(j);
    BOOST_CHECK_EQUAL(config.get_value("pi"), results);

    j = {{"currency", "USD"}, {"value", 42.99}, {"hidden", {{"treasure", "found"}}}};
    results.clear();
    results.push_back(j);
    BOOST_CHECK_EQUAL(config.get_value("object"), results);

    j = 42.99;
    results.clear();
    results.push_back(j);
    BOOST_CHECK_EQUAL(config.get_value("value"), results);

    j = 42;
    results.clear();
    results.push_back(j);
    BOOST_CHECK_EQUAL(config.get_value("everything"), results);

    BOOST_CHECK(config.get_value("Niels").empty());
    BOOST_CHECK(config.get_value("found").empty());
    BOOST_CHECK(config.get_value("3.141").empty());
    BOOST_CHECK(config.get_value("USD").empty());
    BOOST_CHECK(config.get_value("treasures").empty());
    BOOST_CHECK(config.get_value("").empty());
    BOOST_CHECK(config.get_value(" ").empty());
    BOOST_CHECK(config.get_value("/").empty());
}
