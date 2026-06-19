#define BOOST_TEST_MODULE "test_NDArray"

#include <NDArray.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>

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
