#include "testDataCheck.hpp"


using testDataCheckInt = testDataCheck<int32_t>;
using testDataCheckUchar = testDataCheck<uint8_t>;
using testDataCheckFloat = testDataCheck<float>;

REGISTER_KOTEKAN_PROCESS(testDataCheckInt);
REGISTER_KOTEKAN_PROCESS(testDataCheckUchar);
REGISTER_KOTEKAN_PROCESS(testDataCheckFloat);
