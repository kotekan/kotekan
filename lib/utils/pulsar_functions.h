#ifndef PULSAR_FUNCTIONS_H
#define PULSAR_FUNCTIONS_H

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

// the following structure is equivalent to the VDIF header definition 
// contained in "vdif_functions.h", but the pulsar team is gradually 
// moving on to use its own custom header/packet definition. most of the 
// following definition will change in the near future, though some of the 
// key info that we'lll preserve are noted below with comments.
struct PSRHeader {
    uint32_t seconds : 30;
    uint32_t legacy : 1;
    uint32_t invalid : 1;
    uint32_t data_frame : 24;
    uint32_t ref_epoch : 6;
    uint32_t unused : 2;
    uint32_t frame_len : 24;
    uint32_t log_num_chan : 5;
    uint32_t vdif_version : 3;
    uint32_t station_id : 16;
    uint32_t thread_id : 10;    // index of first packed frequency.
    uint32_t bits_depth : 5;
    uint32_t data_type : 1;
    uint32_t eud1 : 24;         // UD: beam number [0 to 9]
    uint32_t edv : 8;           
    uint32_t eud2 : 32;         // _psr_scaling from metadata
    uint32_t eud3 : 32;         // 32-b number encoding other three frequency indeces.
    uint32_t eud4 : 32;         // 32-b number encoding 16-b RA + 16-b Dec
};


#ifdef __cplusplus
}
#endif

#endif /* PULSAR_FUNCTIONS_H */
