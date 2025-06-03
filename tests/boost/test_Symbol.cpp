#define BOOST_TEST_MODULE "test_Symbol"

#include <Symbol.hpp>
#include <boost/test/included/unit_test.hpp>
#include <cassert>
#include <iostream>
#include <set>
#include <unordered_set>

using namespace kotekan;

BOOST_AUTO_TEST_CASE(test1) {
    std::cout << "Testing Symbol class...\n";

    Symbol sym("Hello, World!");

    const Symbol x("x");
    const Symbol y("y");
    const Symbol x2("x");

    std::cout << "x: " << x << "\n";
    std::cout << "y: " << y << "\n";
    std::cout << "x2: " << x2 << "\n";

    BOOST_CHECK(x != y);
    BOOST_CHECK(x == x2);

    std::set<Symbol> set1;
    std::unordered_set<Symbol> set2;

    set1.insert(x);
    set2.insert(x);
    BOOST_CHECK(set1.count(x));
    BOOST_CHECK(!set1.count(y));
    BOOST_CHECK(set2.count(x));
    BOOST_CHECK(!set2.count(y));

    std::cout << "Success.\n";
}
