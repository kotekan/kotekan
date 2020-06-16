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
        {"object", {{"currency", "USD"}, {"value", 42.99}, {"hidden", {{"treasure", "found"}}}}},
        {"add", "2 + pi"},
        {"one_var", "pi"},
        {"num_as_string", "-5.01"},
        {"int_as_string", "50"},
        {"int_as_string2", "-50"},
        {"multiply", "2 * pi"},
        {"subtract", "25.5 - 0.5"},
        {"divide", "10 / 4"},
        {"scientific", "-0.5e2"},
        {"scientific2", "-0.5e2 + -0.5e+4 + 1.2e-1 - 5.32e-2"},
        {"scientific3", "1.22e20"},
        {"scientific4", "1.258e-20"},
        {"broken", "5 + e3"},
        {"broken2", "5 +"},
        {"broken3", "5 +* 8"},
        {"broken4", "5 + (8"},
        {"broken5", "*5"},
        {"missing", "5 + not_set"},
        {"complicated", "1e4 + divide * 10 /(4 + 1) - 20 * (6-pi) + -1.2e2"}};
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

    // Check the arithmetic parser
    BOOST_CHECK_EQUAL(config.get<double>("/", "add"), 5.141);
    BOOST_CHECK_EQUAL(config.get<double>("/", "one_var"), 3.141);
    BOOST_CHECK_EQUAL(config.get<double>("/", "num_as_string"), -5.01);
    BOOST_CHECK_EQUAL(config.get<int32_t>("/", "int_as_string"), 50);
    BOOST_CHECK_EQUAL(config.get<int32_t>("/", "int_as_string2"), -50);
    BOOST_CHECK_EQUAL(config.get<double>("/", "subtract"), 25);
    BOOST_CHECK_EQUAL(config.get<double>("/", "divide"), 2.5);
    BOOST_CHECK_EQUAL(config.get<double>("/", "scientific"), -0.5e2);
    BOOST_CHECK_EQUAL(config.get<double>("/", "scientific2"), -0.5e2 + -0.5e+4 + 1.2e-1 - 5.32e-2);
    BOOST_CHECK_EQUAL(config.get<double>("/", "scientific3"), 1.22e20);
    BOOST_CHECK_EQUAL(config.get<double>("/", "scientific4"), 1.258e-20);
    BOOST_CHECK_THROW(config.get<double>("/", "broken"), std::runtime_error);
    BOOST_CHECK_THROW(config.get<double>("/", "broken2"), std::runtime_error);
    BOOST_CHECK_THROW(config.get<double>("/", "broken3"), std::runtime_error);
    BOOST_CHECK_THROW(config.get<double>("/", "broken4"), std::runtime_error);
    BOOST_CHECK_THROW(config.get<double>("/", "broken5"), std::runtime_error);
    BOOST_CHECK_THROW(config.get<double>("/", "missing"), std::runtime_error);
    BOOST_CHECK_EQUAL(config.get<double>("/", "complicated"),
                      1e4 + (2.5) * 10 / (4 + 1) - 20 * (6 - 3.141) + -1.2e2);


    BOOST_CHECK(config.get_value("Niels").empty());
    BOOST_CHECK(config.get_value("found").empty());
    BOOST_CHECK(config.get_value("3.141").empty());
    BOOST_CHECK(config.get_value("USD").empty());
    BOOST_CHECK(config.get_value("treasures").empty());
    BOOST_CHECK(config.get_value("").empty());
    BOOST_CHECK(config.get_value(" ").empty());
    BOOST_CHECK(config.get_value("/").empty());
}
