#define BOOST_TEST_MODULE "test_config"

#include "Hash.hpp" // for Hash, hash

#include "fmt.hpp"  // for format
#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_PP_IIF_0, BOOST_PP_BO...
#include <stdint.h>                          // for uint64_t
#include <stdlib.h>                          // for strtoull
#include <string>                            // for string, allocator


using json = nlohmann::json;

BOOST_AUTO_TEST_CASE(_test_serialise) {

    /* The hash was calculated in python using:

    >>> import mmh3
    >>> h = mmh3.hash128(b"This is a long and random string.", seed=1420)
    >>> print(f"{h:032x}")
    >>> print(f"{h:032x}")
    4dcf87995e06b6b97f012becdab1a2d5
    */

    std::string s = "This is a long and random string.";
    std::string hash_string = "4dcf87995e06b6b97f012becdab1a2d5";
    Hash h0 = Hash::from_string(hash_string);

    // Check the hash against one deserialised from a string
    Hash h1 = hash(s);
    BOOST_CHECK_EQUAL(h0, h1);

    // Check that fmt outputs it right
    BOOST_CHECK_EQUAL(fmt::format("{}", h1), hash_string);

    // round trip through json
    json j = h1;
    Hash h2 = j.get<Hash>();
    BOOST_CHECK_EQUAL(h1, h2);
}


BOOST_AUTO_TEST_CASE(_test_hash) {

    std::string s = "This is a long and random string.";

    // Split the hex from above in two, convert to uint64_t's and compare
    uint64_t high = strtoull("4dcf87995e06b6b9", nullptr, 16);
    uint64_t low = strtoull("7f012becdab1a2d5", nullptr, 16);

    Hash h1 = hash(s);

    BOOST_CHECK_EQUAL(h1.h, high);
    BOOST_CHECK_EQUAL(h1.l, low);
}
