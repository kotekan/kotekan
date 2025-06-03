#define BOOST_TEST_MODULE "test_Metadata"

#include <Metadata.hpp>
#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

using namespace kotekan;

void writeMetadata(const std::shared_ptr<Metadata>& meta) {
    // Write some metadata entries
    meta->set_bool("test_run", false);
    meta->set_int("ndishes", 64);
    meta->set_float("frequency", 3000.0e+6);
    meta->set_string("time_standard", "UT1");

    meta->set_bool_vector("some flags", {true, false, true});
    meta->set_int_vector("dish_indices", {
                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, //
                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, //
                                             -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  //
                                             -1, 9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, //
                                             -1, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, //
                                             -1, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, //
                                             -1, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, //
                                             -1, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, //
                                         });
    meta->set_float_vector("dish_separations", {6.3, 8.5});
    meta->set_string_vector("some strings", {"hello", "world"});
}

template<typename T1, typename T2>
bool isequal(const T1& x1, const T2& x2) {
    return std::equal(x1.begin(), x1.end(), x2.begin());
}

void readMetadata(const std::shared_ptr<const Metadata>& meta) {
    // Test metadata entries
    BOOST_CHECK(meta->has_bool("test_run"));
    BOOST_CHECK(!meta->has_bool("test_run1"));
    BOOST_CHECK(meta->has_int("ndishes"));
    BOOST_CHECK(!meta->has_int("ndishes1"));
    BOOST_CHECK(meta->has_float("frequency"));
    BOOST_CHECK(!meta->has_float("frequency1"));
    BOOST_CHECK(meta->has_string("time_standard"));
    BOOST_CHECK(!meta->has_string("time_standard1"));

    BOOST_CHECK(meta->has_bool_vector("some flags"));
    BOOST_CHECK(!meta->has_bool_vector("some flags 1"));
    BOOST_CHECK(meta->has_int_vector("dish_indices"));
    BOOST_CHECK(!meta->has_int_vector("dish_indices1"));
    BOOST_CHECK(meta->has_float_vector("dish_separations"));
    BOOST_CHECK(!meta->has_float_vector("dish_separations1"));
    BOOST_CHECK(meta->has_string_vector("some strings"));
    BOOST_CHECK(!meta->has_string_vector("some strings1"));

    BOOST_CHECK_EQUAL(meta->bool_size(), meta->bool_keys().size());
    BOOST_CHECK_EQUAL(meta->int_size(), meta->int_keys().size());
    BOOST_CHECK_EQUAL(meta->float_size(), meta->float_keys().size());
    BOOST_CHECK_EQUAL(meta->string_size(), meta->string_keys().size());
    BOOST_CHECK_EQUAL(meta->bool_vector_size(), meta->bool_vector_keys().size());
    BOOST_CHECK_EQUAL(meta->int_vector_size(), meta->int_vector_keys().size());
    BOOST_CHECK_EQUAL(meta->float_vector_size(), meta->float_vector_keys().size());
    BOOST_CHECK_EQUAL(meta->string_vector_size(), meta->string_vector_keys().size());

    BOOST_CHECK(isequal(meta->bool_keys(), std::vector<std::string>{"test_run"}));
    BOOST_CHECK(isequal(meta->int_keys(), std::vector<std::string>{"ndishes"}));
    BOOST_CHECK(isequal(meta->float_keys(), std::vector<std::string>{"frequency"}));
    BOOST_CHECK(isequal(meta->string_keys(), std::vector<std::string>{"time_standard"}));
    BOOST_CHECK(isequal(meta->bool_vector_keys(), std::vector<std::string>{"some flags"}));
    BOOST_CHECK(isequal(meta->int_vector_keys(), std::vector<std::string>{"dish_indices"}));
    BOOST_CHECK(isequal(meta->float_vector_keys(), std::vector<std::string>{"dish_separations"}));
    BOOST_CHECK(isequal(meta->string_vector_keys(), std::vector<std::string>{"some strings"}));

    // Read metadata entries
    BOOST_CHECK_EQUAL(meta->get_bool("test_run"), false);
    BOOST_CHECK_EQUAL(meta->get_int("ndishes"), 64);
    BOOST_CHECK_EQUAL(meta->get_float("frequency"), 3000.0e+6);
    BOOST_CHECK_EQUAL(meta->get_string("time_standard"), "UT1");

    BOOST_CHECK(isequal(meta->get_bool_vector("some flags"), std::vector<bool>{true, false, true}));
    BOOST_CHECK(isequal(meta->get_int_vector("dish_indices"),
                        std::vector<int>{
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, //
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, //
                            -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  //
                            -1, 9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, //
                            -1, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, //
                            -1, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, //
                            -1, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, //
                            -1, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, //
                        }));
    BOOST_CHECK(isequal(meta->get_float_vector("dish_separations"), std::vector<double>{6.3, 8.5}));
    BOOST_CHECK(isequal(meta->get_string_vector("some strings"),
                        std::vector<std::string>{"hello", "world"}));
}

BOOST_AUTO_TEST_CASE(test1) {
    std::cout << "Testing Metadata class...\n";
    // Create a metadata container
    auto meta = std::make_shared<Metadata>();
    writeMetadata(meta);
    readMetadata(meta);

    std::cout << *meta;

    std::cout << "Success.\n";
}
