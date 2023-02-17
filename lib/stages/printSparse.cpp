#include "printSparse.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate

using printSparseUint8 = printSparse<uint8_t>;
using printSparseUint32 = printSparse<uint32_t>;

REGISTER_KOTEKAN_STAGE(printSparseUint8);
REGISTER_KOTEKAN_STAGE(printSparseUint32);
