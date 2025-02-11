#include <NDArray.hpp>

// Test

NDArray<int, 0> a0({}, {});
NDArray<long long, 1> a1({1}, {"a"});
NDArray<float, 2> a2({2, 3}, {"u", "v"});
NDArray<double, 3> a3({4, 5, 6}, {"x", "y", "z"});

NDArray<double, 3> a4({{"x", 4}, {"y", 5}, {"z", 6}});
