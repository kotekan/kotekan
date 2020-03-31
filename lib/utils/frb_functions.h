#ifndef FRB_FUNCTIONS_H
#define FRB_FUNCTIONS_H

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)
struct FRBHeader {
    uint32_t protocol_version;
    int16_t data_nbytes;
    uint16_t fpga_counts_per_sample;
    uint64_t fpga_count;
    uint16_t nbeams;
    uint16_t nfreq_coarse;
    uint16_t nupfreq;
    uint16_t ntsamp;

    /*Here are some dynamic paramters of the header
      that I have to allocate in frbPostProcess.cpp*/

    // uint16_t * beam_ids = nullptr; //size of [nbeams]
    // uint16_t * coarse_freq_ids; //size of [nfreq_coarse];
    // float *scale ; //size of [nbeams * nfreq_coarse];
    // float *offset ; //size of [nbeams * nfreq_coarse] ;
};
#pragma pack(pop)


#ifdef __cplusplus
}
#endif

#endif /* FRB_FUNCTIONS_H */
