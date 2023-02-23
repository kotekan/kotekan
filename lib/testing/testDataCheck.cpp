#include "testDataCheck.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "visUtil.hpp"

using testDataCheckInt = testDataCheck<int32_t>;
using testDataCheckUchar = testDataCheck<uint8_t>;
using testDataCheckFloat = testDataCheck<float>;
#if KOTEKAN_FLOAT16
using testDataCheckFloat16 = testDataCheck<float16_t>;
#endif

REGISTER_KOTEKAN_STAGE(testDataCheckInt);
REGISTER_KOTEKAN_STAGE(testDataCheckUchar);
REGISTER_KOTEKAN_STAGE(testDataCheckFloat);
#if KOTEKAN_FLOAT16
REGISTER_KOTEKAN_STAGE(testDataCheckFloat16);
#endif
