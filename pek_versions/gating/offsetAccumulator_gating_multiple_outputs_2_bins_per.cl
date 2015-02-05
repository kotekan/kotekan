//#pragma OPENCL EXTENSION cl_amd_printf : enable
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
//#define BASE_TIMESAMPLES_INT_ACCUM  256u
//#define ACTUAL_NUM_ELEMENTS         256u
//#define ACTUAL_NUM_FREQUENCIES      8u
#define NUM_TIMESTEPS_LOCAL         BASE_TIMESAMPLES_INT_ACCUM //256u //if you change this, check the gws_accum in the main program, too! (also note that the max possible value here is 4096)
#define OFFSET_FOR_1_TIMESTEP       (ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES/4u) //divide by 4 to account for the 4 B per uint
#define TIME_DIV_256                (get_group_id(2))
#define TIME_FOR_256_TIMESTEPS      65536u //time (in 10 ns units) for 256 timesteps
#define ELEMENT_AND_FREQUENCY_GROUP (get_global_id(1)) //in this case the group is the global id--not sure which is quicker, so using id first

__kernel void offsetAccumulateElements (__global const uint *inputData,
                                        __global uint *outputData,
                                        __global uint *counters,
                                        const uint gating_group_count, //number of gating periods to check
                                        __global const uint *num_time_bins, //number of bins to sort things into (part of duty cycle)
                                        __global const uint *offset_remainder, //a 'wrapped' offset time
                                        __global const uint *delta_time_bin, //time for 1 bin in the gating (in units of 10 ns)
                                        __global const uint *bin_cut) //number of bins to keep in the 'on' section (entry values must be >= 1)
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

        int trick_the_compiler = NUM_TIMESTEPS_LOCAL;
#pragma unroll 1
        for (int i = 0; i < trick_the_compiler; i++){
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
    //    int i = 0;
        for (int i = 0; i < gating_group_count; i++){
            address = (offset_remainder[i] + TIME_DIV_256*TIME_FOR_256_TIMESTEPS);
            address = address % (num_time_bins[i]*delta_time_bin[i]);
            address = address / delta_time_bin[i]; //essentially the gate bin here
            address = (address >= bin_cut[i]) ? 1 : 0; //compress all output to two bins--Need to define a duty cycle that accounts for the current NUM_DATASETS...  NUM_DATASETS will go to 2 x number of distinct periods...
            if (ELEMENT_AND_FREQUENCY_GROUP==0 && get_local_id(0)==0)
    //            atomic_inc(&counters[0]);
                atomic_inc(&counters[address+2*i]); //increment the counter for a particular dataset (only once for a particular bin) //these are correct--have been checked Apr 8/14
            address *= ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;//for multi-dataset was simply this: get_group_id(0) * ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*2;
            address += (get_local_id(0) + ELEMENT_AND_FREQUENCY_GROUP*64u)*8u; //([0,504] + [0,M*N/256)*512)
    //        address = (get_local_id(0) + ELEMENT_AND_FREQUENCY_GROUP*64u)*8u; //([0,504] + [0,M*N/256)*512)
            address += i*ACTUAL_NUM_ELEMENTS*ACTUAL_NUM_FREQUENCIES*4; //i*2: 2 bins (on/off) * 2 (vals per complex) = *4
            atomic_add(&outputData[address+0u], (dataExpanded.s0>>16) );         //real value
            atomic_add(&outputData[address+1u], (dataExpanded.s0&0x0000ffff) );  //imaginary
            atomic_add(&outputData[address+2u], (dataExpanded.s1>>16) );         //real value
            atomic_add(&outputData[address+3u], (dataExpanded.s1&0x0000ffff) );  //imaginary
            atomic_add(&outputData[address+4u], (dataExpanded.s2>>16) );         //real value
            atomic_add(&outputData[address+5u], (dataExpanded.s2&0x0000ffff) );  //imaginary
            atomic_add(&outputData[address+6u], (dataExpanded.s3>>16) );         //real value
            atomic_add(&outputData[address+7u], (dataExpanded.s3&0x0000ffff) );  //imaginary
            //if (get_global_id(0)==0 && get_global_id(1)== 0 && get_global_id(2) == 0)
            //    printf("%d \n",i);
        }
    }
}
