#ifndef FRB_FUNCTIONS_H
#define FRB_FUNCTIONS_H

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)
// trickey to get rename the structs without changing imported files
#undef L0_L1_HEADER_VARIABLE_SIZE_PART
#define L0_L1_header FRBHeader
#define nbeam nbeams
#define fpga_frame0_ns fpga0_ns
#include "L0_L1_packet.hpp"
#undef fpga_frame0_ns
#undef nbeam
#undef L0_L1_header
#pragma pack(pop)


#ifdef __cplusplus
}
#endif

#endif /* FRB_FUNCTIONS_H */
