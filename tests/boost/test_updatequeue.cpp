#define BOOST_TEST_MODULE "test_updateQueue"

#include "updateQueue.hpp" // for updateQueue
#include "visUtil.hpp"     // for operator==

#include "fmt.hpp" // for format

#include <algorithm>                         // for copy, copy_backward, max
#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_CHECK, BOOST_PP_BOOL_2
#include <ctime>                             // for timespec
#include <iostream>                          // for cout, ostream, std
#include <memory>
#include <string>                            // for operator<<
#include <utility>                           // for pair
#include <vector>                            // for vector

struct timespec;

using namespace std;

timespec zero = {0, 0};
timespec one = {1, 0};
timespec two = {2, 0};
timespec three = {3, 0};
timespec four = {4, 0};
timespec five = {5, 0};
timespec six = {6, 0};

BOOST_AUTO_TEST_CASE(_updateQueue) {
    updateQueue<int> q(3);

    q.insert({1, 0}, 1);
    q.insert({2, 0}, 2);
    q.insert({3, 0}, 3);

    pair<timespec, shared_ptr<const int>> r;

    r = q.get_update({1, 1});
    BOOST_CHECK(*(r.second) == 1);
    BOOST_CHECK(r.first == one);

    r = q.get_update({2, 1});
    BOOST_CHECK(*(r.second) == 2);
    BOOST_CHECK(r.first == two);

    r = q.get_update({3, 1});
    BOOST_CHECK(*(r.second) == 3);
    BOOST_CHECK(r.first == three);

    r = q.get_update({4, 1});
    BOOST_CHECK(*(r.second) == 3);
    BOOST_CHECK(r.first == three);

    // All updates are in the future, so a nullptr will be returned here
    r = q.get_update({0, 0});
    BOOST_CHECK(r.second == nullptr);
}

BOOST_AUTO_TEST_CASE(_updateQueue_one_update) {
    updateQueue<int> q(3);

    q.insert({1, 0}, 1);

    pair<timespec, shared_ptr<const int>> r;

    r = q.get_update({1, 1});
    BOOST_CHECK(*(r.second) == 1);
    BOOST_CHECK(r.first == one);

    r = q.get_update({2, 1});
    BOOST_CHECK(*(r.second) == 1);
    BOOST_CHECK(r.first == one);

    // All updates are in the future, so a nullptr will be returned here
    r = q.get_update({0, 0});
    BOOST_CHECK(r.second == nullptr);
}

BOOST_AUTO_TEST_CASE(_updateQueue_no_update) {
    updateQueue<std::vector<int>> q(3);

    pair<timespec, shared_ptr<const std::vector<int>>> r;

    r = q.get_update({1, 1});
    BOOST_CHECK(r.second == nullptr);
    BOOST_CHECK(r.first == zero);
}

BOOST_AUTO_TEST_CASE(_updateQueue_zero_len) {
    updateQueue<std::vector<int>> q(0);

    pair<timespec, shared_ptr<const std::vector<int>>> r;

    r = q.get_update({1, 1});
    BOOST_CHECK(r.second == nullptr);
    BOOST_CHECK(r.first == zero);
}

BOOST_AUTO_TEST_CASE(_updateQueue_pop) {
    updateQueue<int> q(3);

    q.insert({1, 0}, 1);
    q.insert({2, 0}, 2);
    q.insert({3, 0}, 3);
    q.insert({4, 0}, 4);
    q.insert({5, 0}, 5);
    q.insert({6, 0}, 6);

    pair<timespec, shared_ptr<const int>> r;

    // Request the exact update timestamp
    r = q.get_update({4, 0});
    BOOST_CHECK(*(r.second) == 4);
    BOOST_CHECK(r.first == four);

    r = q.get_update({5, 1});
    BOOST_CHECK(*(r.second) == 5);
    BOOST_CHECK(r.first == five);

    r = q.get_update({6, 1});
    BOOST_CHECK(*(r.second) == 6);
    BOOST_CHECK(r.first == six);

    // The queue is too short for this update to appear
    r = q.get_update({1, 0});
    BOOST_CHECK(r.second == nullptr);

    r = q.get_update({7, 0});
    BOOST_CHECK(*(r.second) == 6);
    BOOST_CHECK(r.first == six);
}

BOOST_AUTO_TEST_CASE(_updateQueue_out_of_order) {
    updateQueue<int> q(3);

    q.insert({6, 0}, 1);
    q.insert({5, 0}, 2);
    q.insert({4, 0}, 3);
    q.insert({3, 0}, 4);
    q.insert({2, 0}, 5);
    q.insert({1, 0}, 6);

    pair<timespec, shared_ptr<const int>> r;

    // Request the exact update timestamp
    r = q.get_update({4, 0});
    BOOST_CHECK(*(r.second) == 3);
    BOOST_CHECK(r.first == four);

    r = q.get_update({5, 1});
    BOOST_CHECK(*(r.second) == 2);
    BOOST_CHECK(r.first == five);

    r = q.get_update({6, 1});
    BOOST_CHECK(*(r.second) == 1);
    BOOST_CHECK(r.first == six);

    // The queue is too short for this update to appear
    r = q.get_update({1, 0});
    BOOST_CHECK(r.second == nullptr);

    r = q.get_update({7, 0});
    BOOST_CHECK(*(r.second) == 1);
    BOOST_CHECK(r.first == six);

    // A pretty stupid "test" for the fmt formatter:
    std::cout << fmt::format("This queue should have timestamps 4, 5 and 6 now: {}", q);
}
