#include "testDataCheck.hpp"


using testDataCheckInt = testDataCheck<int32_t>;
using testDataCheckUchar = testDataCheck<uint8_t>;
using testDataCheckFloat = testDataCheck<float>;

REGISTER_KOTEKAN_STAGE(testDataCheckInt);
REGISTER_KOTEKAN_STAGE(testDataCheckUchar);
REGISTER_KOTEKAN_STAGE(testDataCheckFloat);
