#define POINTING get_group_id(0)   
#define freq get_group_id(1)       
#define time get_group_id(2)       
#define local_unit get_local_id(0)  
#define group_size get_local_size(0)
#define NINPUT 256
#define NFREQ 8  
#define POLBLOCK 16 // for hco and gbo  
// author: Shion Andrew

// notes: 
// 1) assuming phase map should contain gains and are correctly conjugated
// 2) not accounting for output type (int v float) and pol/axis ordering should be checked
__kernel void gpu_beamforming(global uchar* inputData, 
                              global float2* phaseMap,
                              global uchar* outputData,  
                              local float2* outputPartialX,
                              local float2* outputPartialY, 
                              global float *scaling,
                              private uint NPOINTING) {

    int groupData_chunk_start = time * (NFREQ * NINPUT) + freq * NINPUT; // (ntime,nfreq,ninput) -> (ntime,nfreq,npointing)
    int groupPhase_chunk_start = POINTING * (NFREQ * NINPUT) + freq * NINPUT; // assuming (npointing,nfreq,ninput) i.e. npointing is slowest varying
    
    float2 xpol = {0.0f, 0.0f}; 
    float2 ypol = {0.0f, 0.0f}; 

    for (int idx = local_unit; idx < NINPUT; idx = idx + group_size) {
        uchar data_temp = inputData[groupData_chunk_start + idx];
        float2 ph = phaseMap[groupPhase_chunk_start + idx];
        // assuming phases are already conjugated 
        if ((idx / POLBLOCK ) % 2 == 0) {
            xpol.x += ph.x*((float)((data_temp & 0xf0)>> 4u))
                    - ph.y*((float)((data_temp & 0x0f)>> 0u));     
            xpol.y += ph.y*((float)((data_temp & 0xf0)>> 4u))
                    + ph.x*((float)((data_temp & 0x0f)>> 0u));       
            //double check that integer division floors here
            // also double check float promotion in opencl
        } else {
            ypol.x += ph.x*((float)((data_temp & 0xf0)>> 4u))
                    - ph.y*((float)((data_temp & 0x0f)>> 0u));     
            ypol.y += ph.y*((float)((data_temp & 0xf0)>> 4u))
                    + ph.x*((float)((data_temp & 0x0f)>> 0u)); 
        }
    }

    outputPartialX[local_unit] = xpol;
    outputPartialY[local_unit] = ypol;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride >= 1; stride /= 2) {
        if (local_unit < stride) {
            outputPartialX[local_unit] += outputPartialX[local_unit + stride];
        } else {
            int temp_id = local_unit - stride;
            if (temp_id < stride) {
                outputPartialY[temp_id] += outputPartialY[temp_id + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    sum.RE = (sum.RE > 15) ? 15 : sum.RE;
    sum.RE = (sum.RE < 0) ? 0 : sum.RE;

    sum.IM = (sum.IM > 15) ? 15 : sum.IM;
    sum.IM = (sum.IM < 0) ? 0 : sum.IM;
    
    if (local_unit == 0) {
        //int out_idx = time * (NFREQ * NPOINTING*2) + freq * NPOINTING*2 + POINTING*2; 
        int out_idx = freq * (ntime * NPOINTING*2) + time * NPOINTING*2 + POINTING*2; 
        outputData[out_idx] = outputPartialX[0] / scaling[POINTING]; 
    }
    if (local_unit == 1) {
        //int out_idx = time * (NFREQ * NPOINTING*2) + freq * NPOINTING*2 + POINTING*2;
        int out_idx = freq * (ntime * NPOINTING*2) + time * NPOINTING*2 + POINTING*2; 
        outputData[out_idx + 1] = outputPartialY[0] / scaling[POINTING];
    }

}
