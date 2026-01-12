#define POINTING get_group_id(0)   // pointing
#define freq get_group_id(1)       // freq
#define time get_group_id(2)       // time
#define local_unit get_local_id(0) // polarization
#define group_size get_local_size(0)
#define NINPUT 256
#define NFREQ 8  

// notes: 
// 1) assuming phase map should contain gains
// 2) not accounting for real vs imaginary 
// 3) not accounting for output type (int v float) 
// 4) exact olarization hand locations still need to be checked for input and output
__kernel void gpu_beamforming(global uint* inputData, global float* phaseMap,
                              global float* outputData, //local uint* inputDataBlock,local float* phaseBlock, 
                              local float* outputPartialX,
                              local float* outputPartialY, private uint NPOINTING) {
    // copy inputData inputDataBlock
    // N_jobs = NINPUT/group_size // by construction will have remainder 0
    int groupData_chunk_start = time * (NFREQ * NINPUT) + freq * NINPUT; // (ntime,nfreq,ninput) -> (ntime,nfreq,npointing)
    int groupPhase_chunk_start = POINTING * (NFREQ * NINPUT) + freq * NINPUT; // assuming (npointing,nfreq,ninput) i.e. npointing is slowest varying
    // stride copy
    
    
    // this very likely can be deleted
    // for (int idx = local_unit; idx < NINPUT; idx = idx + group_size){
    //     inputDataBlock[idx] = inputData[groupData_chunk_start + idx];
    //     phaseBlock[idx] = phaseMap[groupPhase_chunk_start + idx];
    // } 
    // barrier(CLK_LOCAL_MEM_FENCE);

    float xpol = 0.0;
    float ypol = 0.0;

    for (int idx = local_unit; idx < NINPUT; idx = idx + group_size) {
        if (idx % 2 == 0) {
            xpol += inputData[groupData_chunk_start + idx]*phaseMap[groupPhase_chunk_start + idx];
            //inputDataBlock[idx] * phaseBlock[idx];
        } else {
            ypol += inputData[groupData_chunk_start + idx]*phaseMap[groupPhase_chunk_start + idx];
            //inputDataBlock[idx] * phaseBlock[idx];
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
    // //Reduce X
    // for (int stride = group_size / 2; stride > 0; stride /= 2) {
    //     if (local_unit < stride) {
    //         outputPartialX[local_unit] += outputPartialX[local_unit + stride];
    //     }
    //     barrier(CLK_LOCAL_MEM_FENCE);
    // }

    // // Reduce Y
    // for (int stride = group_size / 2; stride > 0; stride /= 2) {
    //     if (local_unit < stride) {
    //         outputPartialY[local_unit] += outputPartialY[local_unit + stride];
    //     }
    //     barrier(CLK_LOCAL_MEM_FENCE);
    // }


    if (local_unit == 0) {
        int out_idx = time * (NFREQ * NPOINTING*2) + freq * NPOINTING*2 + POINTING*2; //(ntime,nfreq,npointing*2)
        outputData[out_idx] = outputPartialX[0]; // double check
    }
    if (local_unit == 1) {
        int out_idx = time * (NFREQ * NPOINTING*2) + freq * NPOINTING*2 + POINTING*2;
        outputData[out_idx + 1] = outputPartialY[0]; // double check
    }
    // if (local_unit == 0) {
    //     int out_idx = time * (NFREQ * NPOINTING*2) + freq * NPOINTING*2 + POINTING*2;
    //     outputData[out_idx]     = outputPartialX[0];
    //     outputData[out_idx + 1] = outputPartialY[0];
    // }

}
