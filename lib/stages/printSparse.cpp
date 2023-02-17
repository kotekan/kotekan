#include "printSparse.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate

using printSparseUint8 = printSparse<uint8_t>;
using printSparseUint32 = printSparse<uint32_t>;
#if KOTEKAN_FLOAT16
using printSparseFloat16 = printSparse<float16_t>;
#endif

REGISTER_KOTEKAN_STAGE(printSparseUint8);
REGISTER_KOTEKAN_STAGE(printSparseUint32);
#if KOTEKAN_FLOAT16
REGISTER_KOTEKAN_STAGE(printSparseFloat16);
#endif
