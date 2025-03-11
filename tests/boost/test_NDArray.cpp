#define BOOST_TEST_MODULE "test_NDArray"

#include <NDArray.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>

using namespace kotekan;

void examineNDArray(const std::string& name, const GenericNDArray& arr) {
    std::cout << "name: " << name << "\n";
    arr.output_metadata(std::cout);
}

BOOST_AUTO_TEST_CASE(test1) {
    NDArray<int, 0> a0({}, {});
    NDArray<long long, 1> a1({1}, {"a"});
    NDArray<float, 2> a2({2, 3}, {"u", "v"});
    NDArray<double, 3> a3({4, 5, 6}, {"x", "y", "z"});

    NDArray<double, 3> a4({{"x", 4}, {"y", 5}, {"z", 6}});

    examineNDArray("a0", a0);
    examineNDArray("a1", a1);
    examineNDArray("a2", a2);
    examineNDArray("a3", a3);
    examineNDArray("a4", a4);
}
