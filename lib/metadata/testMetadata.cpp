#include <Metadata.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace {

void writeMetadata(const std::shared_ptr<Metadata>& meta) {
    // Write some metadata entries
    meta->set_bool("test_run", false);
    meta->set_int("ndishes", 64);
    meta->set_real("frequency", 3000.0e+6);
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
    meta->set_real_vector("dish_separations", {6.3, 8.5});
    meta->set_string_vector("some strings", {"hello", "world"});
}

template<typename T1, typename T2>
bool isequal(const T1& x1, const T2& x2) {
    return std::equal(x1.begin(), x1.end(), x2.begin());
}

void readMetadata(const std::shared_ptr<const Metadata>& meta) {
    // Test metadata entries
    assert(meta->has_bool("test_run"));
    assert(!meta->has_bool("test_run1"));
    assert(meta->has_int("ndishes"));
    assert(!meta->has_int("ndishes1"));
    assert(meta->has_real("frequency"));
    assert(!meta->has_real("frequency1"));
    assert(meta->has_string("time_standard"));
    assert(!meta->has_string("time_standard1"));

    assert(meta->has_bool_vector("some flags"));
    assert(!meta->has_bool_vector("some flags 1"));
    assert(meta->has_int_vector("dish_indices"));
    assert(!meta->has_int_vector("dish_indices1"));
    assert(meta->has_real_vector("dish_separations"));
    assert(!meta->has_real_vector("dish_separations1"));
    assert(meta->has_string_vector("some strings"));
    assert(!meta->has_string_vector("some strings1"));

    assert(meta->bool_size() == meta->bool_keys().size());
    assert(meta->int_size() == meta->int_keys().size());
    assert(meta->real_size() == meta->real_keys().size());
    assert(meta->string_size() == meta->string_keys().size());
    assert(meta->bool_vector_size() == meta->bool_vector_keys().size());
    assert(meta->int_vector_size() == meta->int_vector_keys().size());
    assert(meta->real_vector_size() == meta->real_vector_keys().size());
    assert(meta->string_vector_size() == meta->string_vector_keys().size());

    assert(isequal(meta->bool_keys(), std::vector<std::string>{"test_run"}));
    assert(isequal(meta->int_keys(), std::vector<std::string>{"ndishes"}));
    assert(isequal(meta->real_keys(), std::vector<std::string>{"frequency"}));
    assert(isequal(meta->string_keys(), std::vector<std::string>{"time_standard"}));
    assert(isequal(meta->bool_vector_keys(), std::vector<std::string>{"some flags"}));
    assert(isequal(meta->int_vector_keys(), std::vector<std::string>{"dish_indices"}));
    assert(isequal(meta->real_vector_keys(), std::vector<std::string>{"dish_separations"}));
    assert(isequal(meta->string_vector_keys(), std::vector<std::string>{"some strings"}));

    // Read metadata entries
    assert(meta->get_bool("test_run") == false);
    assert(meta->get_int("ndishes") == 64);
    assert(meta->get_real("frequency") == 3000.0e+6);
    assert(meta->get_string("time_standard") == "UT1");

    assert(isequal(meta->get_bool_vector("some flags"), std::vector<bool>{true, false, true}));
    assert(isequal(meta->get_int_vector("dish_indices"),
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
    assert(isequal(meta->get_real_vector("dish_separations"), std::vector<double>{6.3, 8.5}));
    assert(isequal(meta->get_string_vector("some strings"),
                   std::vector<std::string>{"hello", "world"}));
}

} // namespace

int main() {
    std::cout << "Testing Metadata class...\n";
    // Create a metadata container
    auto meta = std::make_shared<Metadata>();
    writeMetadata(meta);
    readMetadata(meta);

    std::cout << *meta;

    std::cout << "Success.\n";
    return 0;
}
