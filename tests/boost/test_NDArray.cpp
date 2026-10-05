#define BOOST_TEST_MODULE "test_NDArray"

#include <NDArray.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <utility> // for move

using namespace kotekan;

void examineNDArray(const GenericNDArray& arr) {
    arr.output_framedesc(std::cout);
}

BOOST_AUTO_TEST_CASE(test1) {
    NDArray<int, 0> a0("a0", {}, {}, {});
    NDArray<long long, 1> a1("a1", {1}, {"a"}, {1});
    NDArray<float, 2> a2("a2", {2, 3}, {"u", "v"}, {2, 2});
    NDArray<double, 3> a3("a3", {4, 5, 6}, {"x", "y", "z"}, {4, 4, 4});

    examineNDArray(a0);
    examineNDArray(a1);
    examineNDArray(a2);
    examineNDArray(a3);
}

BOOST_AUTO_TEST_CASE(test_describe_has_no_data) {
    const auto desc = NDArray<float, 2>::describe("d", {1024, 1024}, {"u", "v"}, {1, 1});
    BOOST_CHECK(desc->data() == nullptr);
    BOOST_CHECK_EQUAL(desc->get_byte_size(), 1024 * 1024 * sizeof(float));
}

BOOST_AUTO_TEST_CASE(test_move_owned_data) {
    NDArray<float, 1> b("b", {4}, {"x"}, {1});
    {
        NDArray<float, 1> a("a", {3}, {"x"}, {1});
        a(2) = 42;
        // The target's own data must be freed, and the moved data must outlive `a`
        b = std::move(a);
    }
    BOOST_CHECK_EQUAL(b.extent(0), 3);
    BOOST_CHECK_EQUAL(b(2), 42);

    NDArray<float, 1> c(std::move(b));
    BOOST_CHECK_EQUAL(c(2), 42);
}
