#include <NDArray.hpp>
#include <iostream>
#include <sstream>
#include <string>

namespace {

const char* format_bool(bool b) {
    return b ? "true" : "false";
}

template<typename T>
std::string format_vector(const std::vector<T>& vec) {
    std::ostringstream buf;
    buf << "[";
    bool isfirst = true;
    for (const auto& x : vec)
        buf << (isfirst ? "" : ", ") << x, isfirst = false;
    buf << "]";
    return buf.str();
}

void examineNDArray(const std::string& name, const GenericNDArray& arr) {
    std::cout << "name: " << name << "\n"
              << "    type:      " << arr.get_value_type() << "\n"
              << "    type size: " << arr.get_value_type_size() << "\n"
              << "    rank:      " << arr.get_rank() << "\n"
              << "    extents:   " << format_vector(arr.get_extents()) << "\n"
              << "    empty:     " << format_bool(arr.get_empty()) << "\n"
              << "    size:      " << arr.get_size() << "\n"
              << "    dimnames:  " << format_vector(arr.get_dimnames()) << "\n"
              << "    strides:   " << format_vector(arr.get_strides()) << "\n";
}
} // namespace

int main() {

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

    return 0;
}
