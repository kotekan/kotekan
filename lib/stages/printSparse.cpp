#include "printSparse.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate

#include <array>
#include <bits/floatn-common.h>
#include <cassert>
#include <complex.h>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using float16_t = _Float16;

using printSparseUint8 = printSparse<uint8_t>;
using printSparseUint32 = printSparse<uint32_t>;
using printSparseFloat16 = printSparse<float16_t>;

REGISTER_KOTEKAN_STAGE(printSparseUint8);
REGISTER_KOTEKAN_STAGE(printSparseUint32);
REGISTER_KOTEKAN_STAGE(printSparseFloat16);
