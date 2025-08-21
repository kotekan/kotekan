#define BOOST_TEST_MODULE "test_DataType"

#include <DataType.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace kotekan;

BOOST_AUTO_TEST_CASE(test1) {
    std::cout << "Testing DataType class...\n";

    std::cout << "Test 1:\n";
    std::cout << "    uint1x8:     " << uint1x8 << "\n";
    std::cout << "    uint4x2:     " << uint4x2 << "\n";
    std::cout << "    uint8:       " << uint8 << "\n";
    std::cout << "    uint16:      " << uint16 << "\n";
    std::cout << "    uint32:      " << uint32 << "\n";
    std::cout << "    uint64:      " << uint64 << "\n";
    std::cout << "    int4x2:      " << int4x2 << "\n";
    std::cout << "    int8:        " << int8 << "\n";
    std::cout << "    int16:       " << int16 << "\n";
    std::cout << "    int32:       " << int32 << "\n";
    std::cout << "    int64:       " << int64 << "\n";
    std::cout << "    float16:     " << float16 << "\n";
    std::cout << "    float32:     " << float32 << "\n";
    std::cout << "    float64:     " << float64 << "\n";

    std::cout << "Test 2:\n";
    std::cout << "    unsigned char:      " << GetDataType_v<unsigned char> << "\n";
    std::cout << "    unsigned short:     " << GetDataType_v<unsigned short> << "\n";
    std::cout << "    unsigned int:       " << GetDataType_v<unsigned int> << "\n";
    std::cout << "    unsigned long:      " << GetDataType_v<unsigned long> << "\n";
    std::cout << "    unsigned long long: " << GetDataType_v<unsigned long long> << "\n";
    std::cout << "    signed char:        " << GetDataType_v<signed char> << "\n";
    std::cout << "    short:              " << GetDataType_v<short> << "\n";
    std::cout << "    int:                " << GetDataType_v<int> << "\n";
    std::cout << "    long:               " << GetDataType_v<long> << "\n";
    std::cout << "    long long:          " << GetDataType_v<long long> << "\n";
#if KOTEKAN_FLOAT16
    std::cout << "    float16_t:          " << GetDataType_v<float16_t> << "\n";
#endif
    std::cout << "    float:              " << GetDataType_v<float> << "\n";
    std::cout << "    double:             " << GetDataType_v<double> << "\n";

    std::cout << "Success.\n";
}
