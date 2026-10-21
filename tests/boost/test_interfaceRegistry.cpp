#define BOOST_TEST_MODULE "test_interfaceRegistry"

#include "interfaceRegistry.hpp"     // for canonical_json, build_interfaces, canonical_double
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_CHECK_EQUAL, BOOST_...
#include <stdexcept>                         // for runtime_error, invalid_argument
#include <stdint.h>                          // for INT64_MAX, INT64_MIN, uint32_t, uint64_t
#include <string>                            // for string

using json = nlohmann::json;
using kotekan::build_interfaces;
using kotekan::canonical_double;
using kotekan::canonical_json;
using kotekan::fnv1a64_digest;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

namespace {

json build_example() {
    json sections;
    sections["zeta"] = {{"b", -1}, {"a", {1, 2, 3}}};
    sections["alpha"] = {
        {"name", "L5 \"Q\" \\"}, {"flag", true}, {"carrier_hz", canonical_double(1176.45e6)}};
    return sections;
}

json build_throws() {
    throw std::runtime_error("no constants");
}

json build_bad_message() {
    throw std::runtime_error("bad \xff byte\n");
}

json build_float() {
    return {{"telem", {{"rate", 1.5}}}};
}

json build_array() {
    return json::array();
}

json build_twice_1() {
    return json::object();
}

json build_twice_2() {
    return json::object();
}

} // namespace

REGISTER_KOTEKAN_INTERFACE("example", 3, build_example);
REGISTER_KOTEKAN_INTERFACE("throws", 1, build_throws);
REGISTER_KOTEKAN_INTERFACE("bad_message", 1, build_bad_message);
REGISTER_KOTEKAN_INTERFACE("float", 1, build_float);
REGISTER_KOTEKAN_INTERFACE("array", 1, build_array);
REGISTER_KOTEKAN_INTERFACE("twice", 1, build_twice_1);
REGISTER_KOTEKAN_INTERFACE("twice", 1, build_twice_2);

BOOST_AUTO_TEST_CASE(_fnv1a64) {
    // Known answers from the FNV reference (http://www.isthe.com/chongo/tech/comp/fnv/)
    BOOST_CHECK_EQUAL(fnv1a64_digest(""), "fnv1a64:cbf29ce484222325");
    BOOST_CHECK_EQUAL(fnv1a64_digest("a"), "fnv1a64:af63dc4c8601ec8c");
    BOOST_CHECK_EQUAL(fnv1a64_digest("foobar"), "fnv1a64:85944171f73967e8");
}

BOOST_AUTO_TEST_CASE(_canonical_bytes) {
    json j;
    j["z"] = {{"b", {1, -2, INT64_MAX, INT64_MIN}}, {"a", true}};
    j["A"] = "quote \" and backslash \\ and /slash";
    j["m"] = {{"empty_obj", json::object()},
              {"empty_arr", json::array()},
              {"f", false},
              {"u", uint32_t(4294967295u)}};
    j["_"] = {json::array(), json::array({json::object()}), "~"};

    /* From python3:
    >>> o = {"z": {"b": [1, -2, 9223372036854775807, -9223372036854775808], "a": True},
    ...      "A": 'quote " and backslash \\ and /slash',
    ...      "m": {"empty_obj": {}, "empty_arr": [], "f": False, "u": 4294967295},
    ...      "_": [[], [{}], "~"]}
    >>> json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    */
    std::string expected = R"({"A":"quote \" and backslash \\ and /slash","_":[[],[{}],"~"],)"
                           R"("m":{"empty_arr":[],"empty_obj":{},"f":false,"u":4294967295},)"
                           R"("z":{"a":true,"b":[1,-2,9223372036854775807,-9223372036854775808]}})";
    BOOST_CHECK_EQUAL(canonical_json(j), expected);
}

BOOST_AUTO_TEST_CASE(_canonical_double) {
    // From python3: "%.17g" % x
    BOOST_CHECK_EQUAL(canonical_double(1176.45e6), "1176450000");
    BOOST_CHECK_EQUAL(canonical_double(1e-3), "0.001");
    BOOST_CHECK_EQUAL(canonical_double(0.1), "0.10000000000000001");
    BOOST_CHECK_EQUAL(canonical_double(-2.5e-12), "-2.4999999999999998e-12");
    BOOST_CHECK_EQUAL(canonical_double(5.12e-6), "5.1200000000000001e-06");
    BOOST_CHECK_EQUAL(canonical_double(1e22), "1e+22");
    BOOST_CHECK_EQUAL(canonical_double(-0.0), "-0");
}

BOOST_AUTO_TEST_CASE(_rejects) {
    BOOST_CHECK_THROW(canonical_json({{"a", {{"b", 1.0}}}}), std::invalid_argument);
    BOOST_CHECK_THROW(canonical_json({{"a", nullptr}}), std::invalid_argument);
    BOOST_CHECK_THROW(canonical_json({{"a", uint64_t(INT64_MAX) + 1}}), std::invalid_argument);
    BOOST_CHECK_NO_THROW(canonical_json({{"a", uint64_t(INT64_MAX)}}));

    // Not printable ASCII, in a string or a key: UTF-8, a control character, DEL.
    BOOST_CHECK_THROW(canonical_json({{"a", "caf\xc3\xa9"}}), std::invalid_argument);
    BOOST_CHECK_THROW(canonical_json({{"caf\xc3\xa9", 1}}), std::invalid_argument);
    BOOST_CHECK_THROW(canonical_json({{"a", {"line\n"}}}), std::invalid_argument);
    BOOST_CHECK_THROW(canonical_json({{"a\x7f", 1}}), std::invalid_argument);

    try {
        canonical_json({{"s", {{"x", {0, 2.5}}}}});
        BOOST_FAIL("float accepted");
    } catch (const std::invalid_argument& e) {
        BOOST_CHECK_EQUAL(std::string(e.what()),
                          "float at '/s/x/1'; write doubles with canonical_double()");
    }
}

BOOST_AUTO_TEST_CASE(_build) {
    json interfaces = build_interfaces();

    /* From python3, with canonical() the json.dumps above and fnv() FNV-1a 64 of its bytes:
    >>> sections = {"zeta": {"b": -1, "a": [1, 2, 3]},
    ...             "alpha": {"carrier_hz": "1176450000", "flag": True, "name": 'L5 "Q" \\'}}
    >>> fnv(canonical({"interface": "example", "schema": 3, "sections": sections}))
    >>> {k: fnv(canonical(v)) for k, v in sections.items()}
    */
    const json& example = interfaces.at("example");
    BOOST_CHECK_EQUAL(example.at("schema"), 3);
    BOOST_CHECK_EQUAL(example.at("digest"), "fnv1a64:1dacc0ea03bc8e79");
    BOOST_CHECK_EQUAL(example.at("sections").at("alpha"), "fnv1a64:3c5995dc6b90b081");
    BOOST_CHECK_EQUAL(example.at("sections").at("zeta"), "fnv1a64:9a999fafd7525af6");
    BOOST_CHECK_EQUAL(canonical_json(example.at("doc")),
                      R"({"interface":"example","schema":3,"sections":{"alpha":)"
                      R"({"carrier_hz":"1176450000","flag":true,"name":"L5 \"Q\" \\"},)"
                      R"("zeta":{"a":[1,2,3],"b":-1}}})");

    // Every failure becomes an error block, with a message that is itself canonical.
    BOOST_CHECK_EQUAL(interfaces.at("throws"), json({{"error", "no constants"}}));
    BOOST_CHECK_EQUAL(interfaces.at("bad_message"), json({{"error", "bad ? byte?"}}));
    BOOST_CHECK_EQUAL(
        interfaces.at("float"),
        json(
            {{"error", "float at '/sections/telem/rate'; write doubles with canonical_double()"}}));
    BOOST_CHECK_EQUAL(interfaces.at("array"),
                      json({{"error", "builder must return an object of sections"}}));
    BOOST_CHECK_EQUAL(interfaces.at("twice"),
                      json({{"error", "interface registered more than once"}}));

    // What --interfaces-json prints is canonical, and the same every time.
    BOOST_CHECK_EQUAL(canonical_json(interfaces), canonical_json(build_interfaces()));
}
