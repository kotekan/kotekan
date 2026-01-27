#define POINTING get_group_id(0)   
#define freq get_group_id(1)       
#define time get_group_id(2)       
#define local_unit get_local_id(0)  
#define group_size get_local_size(0)
#define NINPUT 256
#define NFREQ 8  
#define POLBLOCK 16 // for hco and gbo  
#define NTIME get_global_size(2)
#define NPOL 2

#define ROW_SHIFT_BIT 0x100

// imaginary data helpers
#define RE x //x
#define IM y //y

// author: Shion Andrew
// NOTE: group size must be 64 - amdgpu wavefront

#define DPP_SHIFT(x, ctrl) \
    as_float(__builtin_amdgcn_mov_dpp(as_uint(x), ctrl, 0xF, 0xF, 0))

// Macro for ds_bpermute with float input
#define DS_BPERMUTE(x, offset) \
    as_float(__builtin_amdgcn_ds_bpermute(((offset)+get_local_id(0))*4, as_uint(x)))


// static inline float dpp_shift(const float x, const int ctrl)
// {
//     return as_float(__builtin_amdgcn_mov_dpp(
//         as_uint(x),  // value
//         ctrl,        // #lanes over
//         0xF,  
//         0xF,  
//         0 
//     ));
// }

static inline float2 truncate_uint8(float2 val)
{
    val.RE = (val.RE > 15) ? 15 : val.RE;
    val.RE = (val.RE < 0) ? 0 : val.RE;

    val.IM = (val.IM > 15) ? 15 : val.IM;
    val.IM = (val.IM < 0) ? 0 : val.IM;

    return val;
}


// notes: 
// 1) assuming phase map should contain gains and are correctly conjugated
// 2) not accounting for output type (int v float) and pol/axis ordering should be checked
__kernel void gpu_beamforming(global uchar* inputData, 
                              global float2* phaseMap,
                              global uchar* outputData,  
                              private float scaling,
                              private uint NPOINTING) {

    int groupData_chunk_start = time * (NFREQ * NINPUT) + freq * NINPUT; // (ntime,nfreq,ninput) -> (ntime,nfreq,npointing)
    int groupPhase_chunk_start = POINTING * (NFREQ * NINPUT) + freq * NINPUT; // assuming (npointing,nfreq,ninput) i.e. npointing is slowest varying
    
    float2 xpol = {0.0f, 0.0f}; 
    float2 ypol = {0.0f, 0.0f}; 

    // assuming a group size of 64, this adds up 4 items for xpol and ypol
    for (int idx = local_unit; idx < NINPUT; idx = idx + group_size) {
        uchar data_temp = inputData[groupData_chunk_start + idx];
        float2 ph = phaseMap[groupPhase_chunk_start + idx];
        // assuming phases are already conjugated 
        if ((idx / POLBLOCK ) % 2 == 0) {
            xpol.RE += ph.RE*((float)((data_temp & 0xf0)>> 4u)-8)
                    -  ph.IM*((float)((data_temp & 0x0f)>> 0u)-8);     
            xpol.IM += ph.IM*((float)((data_temp & 0xf0)>> 4u)-8)
                    +  ph.RE*((float)((data_temp & 0x0f)>> 0u)-8);       
            //double check that integer division floors here
            // also double check float promotion in opencl
        } else {
            ypol.RE += ph.RE*((float)((data_temp & 0xf0)>> 4u)-8)
                    -  ph.IM*((float)((data_temp & 0x0f)>> 0u)-8);     
            ypol.IM += ph.IM*((float)((data_temp & 0xf0)>> 4u)-8)
                    +  ph.RE*((float)((data_temp & 0x0f)>> 0u)-8); 
        }
        // TODO clarify -8. Signed vs unsigned interpretation.
        // The data is 4bit real 4bit imaginary. Should interpret these values as 
        //     signed or unsigned values. Meaning, is 0000 min, or 1000 min.
        // I believe it should come in signed, so I should NOT subtract 8.
    }

    // pattern matching to lib/hsa/kernels/tracking_beamformer_nbeam.cl
    // using `mov_dpp` since it is faster than `ds_bpermute`.

    // reduce-sum using __builtin_amdgcn_mov_dpp
    // this reads directly from other lanes using hardware paths, without going 
    // into group memory.
    // const uint ROW_SHIFT_BIT = 0x100;

    // // x polarization RE
    // xpol.RE += dpp_shift(xpol.RE, 0x120); // 0x120
    // xpol.RE += dpp_shift(xpol.RE, 0x110);
    // xpol.RE += dpp_shift(xpol.RE, 0x108);
    // xpol.RE += dpp_shift(xpol.RE, 0x104);
    // xpol.RE += dpp_shift(xpol.RE, 0x102); // 0x102

    // DDP_SHIFT is not defined for all offsets. Need to use BPERMUTE
    // reduce by 4x across 0-63. Now in 0-15.
    xpol.RE += DS_BPERMUTE(xpol.RE, 16) +
              DS_BPERMUTE(xpol.RE, 32) +
              DS_BPERMUTE(xpol.RE, 48);

    xpol.IM += DS_BPERMUTE(xpol.IM, 16) +
              DS_BPERMUTE(xpol.IM, 32) +
              DS_BPERMUTE(xpol.IM, 48);
              
    ypol.RE += DS_BPERMUTE(ypol.RE, 16) +
              DS_BPERMUTE(ypol.RE, 32) +
              DS_BPERMUTE(ypol.RE, 48);
              
    ypol.IM += DS_BPERMUTE(ypol.IM, 16) +
              DS_BPERMUTE(ypol.IM, 32) +
              DS_BPERMUTE(ypol.IM, 48);
                 

    // Now reducing with stride of 4. Vals in 0-3
    xpol.RE += DPP_SHIFT(xpol.RE, ROW_SHIFT_BIT | 4)+
               DPP_SHIFT(xpol.RE, ROW_SHIFT_BIT | 8)+
               DPP_SHIFT(xpol.RE, ROW_SHIFT_BIT | 12);

    xpol.IM += DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 4)+
               DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 8)+
               DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 12);
               
    ypol.RE += DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 4)+
               DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 8)+
               DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 12);

    ypol.IM += DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 4)+
               DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 8)+
               DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 12);

    // Now adding up 0-3
    xpol.RE += DPP_SHIFT(xpol.RE, ROW_SHIFT_BIT | 1)+
               DPP_SHIFT(xpol.RE, ROW_SHIFT_BIT | 2)+
               DPP_SHIFT(xpol.RE, ROW_SHIFT_BIT | 3);

    xpol.IM += DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 1)+
               DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 2)+
               DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 3);

    ypol.RE += DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 1)+
               DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 2)+
               DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 3);

    ypol.IM += DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 1)+
               DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 2)+
               DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 3);

    // // x polarization IM
    // xpol.IM += DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 32);
    // xpol.IM += DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 16);
    // xpol.IM += DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 8);
    // xpol.IM += DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 4);
    // xpol.IM += DPP_SHIFT(xpol.IM, ROW_SHIFT_BIT | 2);

    // // y polarization RE
    // ypol.RE += DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 32);
    // ypol.RE += DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 16);
    // ypol.RE += DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 8);
    // ypol.RE += DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 4);
    // ypol.RE += DPP_SHIFT(ypol.RE, ROW_SHIFT_BIT | 2);

    // // y polarization IM
    // ypol.IM += DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 32);
    // ypol.IM += DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 16);
    // ypol.IM += DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 8);
    // ypol.IM += DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 4);
    // ypol.IM += DPP_SHIFT(ypol.IM, ROW_SHIFT_BIT | 2);    
    
    if (local_unit == 0) {
        int out_idx = freq * (NTIME * NPOINTING*NPOL) + time * NPOINTING*NPOL + POINTING*NPOL; 
        // note, not making scaling pointing dependent
        //xpol = xpol / scaling[POINTING] + 8;
        xpol = xpol / scaling + 8;
        xpol = truncate_uint8(xpol);
        outputData[out_idx] = (((int)xpol.RE << 4) & 0xF0) + ((int)xpol.IM & 0x0F); 

        //ypol = ypol / scaling[POINTING] + 8;
        ypol = ypol / scaling + 8;
        ypol = truncate_uint8(ypol);
        outputData[out_idx + 1] = (((int)ypol.RE << 4) & 0xF0) + ((int)ypol.IM & 0x0F); 
    }
}

