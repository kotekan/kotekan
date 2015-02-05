//CASPER ordering
//new offset accumulator/ data reducer...
//the offsets required need each time step to simply be added
//N elements, M frequencies, Q timesteps
//for 256 or less elements
//  global worksize: (64, N*M/256, REDUCED_Q)  // N*M / 256  to say how many blocks should be processed, REDUCED_Q = Q/NUM_TIMESTEPS_LOCAL
//   local worksize: (64,       1,         1)
//each workitem will load and process 4 bytes so the local worksize will take 256 B at a time
//
//want to pack a complex pair (unsigned vals each) into a 4B int while reducing data
//leaves 12 bits, so max number of elements added together before saving to global memory is ~2^12 = 4096,
//but the best number of local adds will likely be a balancing act, based on how many things are running concurrently
//
// data arrives packed as packed data: 1 B for shifted real and imaginary data for frequency bands for antennas
// (4 bits Re 4 bits Im)
// (Re Re Re Re Im Im Im Im)
//
// outputData array must be pre-initialized to zeros
// it will be N*M*2*sizeof(uint) long

// errors in results have (at least partially) to do with the fact that (int)(N*M/256) = 7 for M=126 frequencies (we need 8 and a check!)
//ACTUAL_NUM_ELEMENTS, ACTUAL_NUM_FREQUENCIES defined at compile time
//#define ACTUAL_NUM_ELEMENTS         16u
//#define ACTUAL_NUM_FREQUENCIES      256u
#define NUM_TIMESTEPS_LOCAL         BASE_TIMESAMPLES_INT_ACCUM //256u //if you change this, check the gws_accum in the main program, too! (also note that the max possible value here is 4096)
#define OFFSET_FOR_1_TIMESTEP       (ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES/4u) //divide by 4 to account for the 4 B per uint
#define TIME_DIV_256                (get_group_id(2))
#define TIME_FOR_256_TIMESTEPS      65536u //time (in 10 ns units) for 256 timesteps
#define ELEMENT_AND_FREQUENCY_GROUP (get_global_id(1)) //in this case the group is the global id--not sure which is quicker, so using id first

__kernel void offsetAccumulateElements (__global const uint *inputData,
                                        __global uint *outputData,
                                        __global uint *counters,
                                        const uint offset_remainder, //a 'wrapped' offset time
                                        const uint delta_time_bin,
                                        const uint offset_remainder2, //a 'wrapped' offset time
                                        const uint delta_time_bin2,
                                        const uint offset_remainder3, //a 'wrapped' offset time
                                        const uint delta_time_bin3) //time for 1 bin in the gating (in units of 10 ns)
{
    uint    data;
    uint4   dataExpanded = (uint4)(0u,0u,0u,0u);
    uint4   temp;
    //we load 4 Byte words, addresses are based on that size
    uint    address = get_local_id(0) + //0-63
                      get_group_id(1)*64u;//   the 64 comes from the fact we're using 4 B words... that is we're mutiplying by 256 B / 4 B

    //only compute values if the output address is going to be valid
    //address * 8 accounts for the 8 values (4 pairs of complex numbers) accessed by loading 4 B of packed data
    if ((address * 8u + 7u) < ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2u){ //check to see if the output address falls in a useful range (i.e. < Num_Elements x Num_Freq)
        address += TIME_DIV_256*NUM_TIMESTEPS_LOCAL*OFFSET_FOR_1_TIMESTEP;

        for (int i = 0; i < NUM_TIMESTEPS_LOCAL; i++){
            data = inputData[address]; //should be a coalesced load with local work group

            //unpack
            temp.s0 = ((data & 0x000000f0) << 12u) | ((data & 0x0000000f) >>   0u);
            temp.s1 = ((data & 0x0000f000) <<  4u) | ((data & 0x00000f00) >>   8u);
            temp.s2 = ((data & 0x00f00000) >>  4u) | ((data & 0x000f0000) >>  16u);
            temp.s3 = ((data & 0xf0000000) >> 12u) | ((data & 0x0f000000) >>  24u);

            //accumulate
            dataExpanded += temp;

            //update address for next iteration
            address += OFFSET_FOR_1_TIMESTEP;
        }

        //output reduced data set, expanding one more time (store as uint to avoid a cast to int)--recasting will be done when expanding the smaller N*M*2 matrix to N(N+1)/2*M*2
        //8 output values
        address = (offset_remainder + TIME_DIV_256*TIME_FOR_256_TIMESTEPS);
        address = address % (NUM_DATASETS*delta_time_bin);
        address = address / delta_time_bin; //essentially the gate bin here
        if (ELEMENT_AND_FREQUENCY_GROUP==0 && get_local_id(0)==0)
            atomic_inc(&counters[address]); //increment the counter for a particular dataset (only once for a particular bin) //these are correct--have been checked Apr 8/14
        address *= ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;//for multi-dataset was simply this: get_group_id(0) * ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;
        address += (get_local_id(0) + ELEMENT_AND_FREQUENCY_GROUP*64u)*8u; //([0,504] + [0,M*N/256)*512)
        atomic_add(&outputData[address+0u], (dataExpanded.s0>>16) );         //real value
        atomic_add(&outputData[address+1u], (dataExpanded.s0&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+2u], (dataExpanded.s1>>16) );         //real value
        atomic_add(&outputData[address+3u], (dataExpanded.s1&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+4u], (dataExpanded.s2>>16) );         //real value
        atomic_add(&outputData[address+5u], (dataExpanded.s2&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+6u], (dataExpanded.s3>>16) );         //real value
        atomic_add(&outputData[address+7u], (dataExpanded.s3&0x0000ffff) );  //imaginary
        
        ///
        address = (offset_remainder2 + TIME_DIV_256*TIME_FOR_256_TIMESTEPS);
        address = address % (NUM_DATASETS*delta_time_bin2);
        address = address / delta_time_bin2; //essentially the gate bin here
        if (ELEMENT_AND_FREQUENCY_GROUP==0 && get_local_id(0)==0)
            atomic_inc(&counters[address+NUM_DATASETS]); //increment the counter for a particular dataset (only once for a particular bin) //these are correct--have been checked Apr 8/14
        address *= ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;//for multi-dataset was simply this: get_group_id(0) * ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;
        address += (get_local_id(0) + ELEMENT_AND_FREQUENCY_GROUP*64u)*8u; //([0,504] + [0,M*N/256)*512)
        address += NUM_DATASETS*ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;
        atomic_add(&outputData[address+0u], (dataExpanded.s0>>16) );         //real value
        atomic_add(&outputData[address+1u], (dataExpanded.s0&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+2u], (dataExpanded.s1>>16) );         //real value
        atomic_add(&outputData[address+3u], (dataExpanded.s1&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+4u], (dataExpanded.s2>>16) );         //real value
        atomic_add(&outputData[address+5u], (dataExpanded.s2&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+6u], (dataExpanded.s3>>16) );         //real value
        atomic_add(&outputData[address+7u], (dataExpanded.s3&0x0000ffff) );  //imaginary
        
        ///
        address = (offset_remainder3 + TIME_DIV_256*TIME_FOR_256_TIMESTEPS);
        address = address % (NUM_DATASETS*delta_time_bin3);
        address = address / delta_time_bin3; //essentially the gate bin here
        if (ELEMENT_AND_FREQUENCY_GROUP==0 && get_local_id(0)==0)
            atomic_inc(&counters[address+2*NUM_DATASETS]); //increment the counter for a particular dataset (only once for a particular bin) //these are correct--have been checked Apr 8/14
        address *= ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;//for multi-dataset was simply this: get_group_id(0) * ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;
        address += (get_local_id(0) + ELEMENT_AND_FREQUENCY_GROUP*64u)*8u; //([0,504] + [0,M*N/256)*512)
        address += NUM_DATASETS*ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2*2;
        atomic_add(&outputData[address+0u], (dataExpanded.s0>>16) );         //real value
        atomic_add(&outputData[address+1u], (dataExpanded.s0&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+2u], (dataExpanded.s1>>16) );         //real value
        atomic_add(&outputData[address+3u], (dataExpanded.s1&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+4u], (dataExpanded.s2>>16) );         //real value
        atomic_add(&outputData[address+5u], (dataExpanded.s2&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+6u], (dataExpanded.s3>>16) );         //real value
        atomic_add(&outputData[address+7u], (dataExpanded.s3&0x0000ffff) );  //imaginary
    }
}
