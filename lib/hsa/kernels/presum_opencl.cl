#define NUM_TIMESTEPS_LOCAL         128

__kernel void CHIME_presum (__global uint *inputData,
                                        __global void *nah,
                                        int gx_over_4,
                                        __global uint *outputData){
    uint presum_re[4]={0,0,0,0};
    uint presum_im[4]={0,0,0,0};
    for (uint t=0; t<NUM_TIMESTEPS_LOCAL; t++){
        uint addr = (t + get_global_id(1) * NUM_TIMESTEPS_LOCAL) * get_global_size(0) + get_global_id(0);
        uint xv = inputData[addr];

        #pragma unroll
        for (uint e=0; e<4; e++){
            presum_im[e] += amd_bfe(xv,e*8+0,4);
            presum_re[e] += amd_bfe(xv,e*8+4,4);
        }
    }
    #pragma unroll
    for (uint e=0; e<4; e++){
        atomic_add(&outputData[(get_global_id(0)*4+e)*2+0],8*presum_im[e]);
        atomic_add(&outputData[(get_global_id(0)*4+e)*2+1],8*presum_re[e]);
    }
}

/*
{
    uint    data;
    uint4   dataExpanded = (uint4)(0u,0u,0u,0u);
    uint4   temp;
    uint address;

    for (int t = 0; t < NUM_TIMESTEPS_LOCAL; t++){
        address = (t + get_global_id(1) * NUM_TIMESTEPS_LOCAL) * OFFSET_FOR_1_TIMESTEP + get_global_id(0);
        data = inputData[address]; //should be a coalesced load with local work group

        //unpack
        temp.s0 = ((data & 0x000000f0) << 12u) | ((data & 0x0000000f) >>   0u);
        temp.s1 = ((data & 0x0000f000) <<  4u) | ((data & 0x00000f00) >>   8u);
        temp.s2 = ((data & 0x00f00000) >>  4u) | ((data & 0x000f0000) >>  16u);
        temp.s3 = ((data & 0xf0000000) >> 12u) | ((data & 0x0f000000) >>  24u);

        //accumulate
        dataExpanded += temp;
    }

    //output reduced data set, expanding one more time (store as uint to avoid a cast to int)--recasting will be done when expanding the smaller N*M*2 matrix to N(N+1)/2*M*2
    //8 output values
    address = (get_local_id(0) + get_group_id(1)*64u)*8u; //([0,504] + [0,M*N/256)*512)
    atomic_add(&outputData[address+0u], (dataExpanded.s0>>16) );         //real value
    atomic_add(&outputData[address+1u], (dataExpanded.s0&0x0000ffff) );  //imaginary
    atomic_add(&outputData[address+2u], (dataExpanded.s1>>16) );         //real value
    atomic_add(&outputData[address+3u], (dataExpanded.s1&0x0000ffff) );  //imaginary
    atomic_add(&outputData[address+4u], (dataExpanded.s2>>16) );         //real value
    atomic_add(&outputData[address+5u], (dataExpanded.s2&0x0000ffff) );  //imaginary
    atomic_add(&outputData[address+6u], (dataExpanded.s3>>16) );         //real value
    atomic_add(&outputData[address+7u], (dataExpanded.s3&0x0000ffff) );  //imaginary
}
*/