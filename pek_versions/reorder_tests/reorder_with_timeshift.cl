// Kernel dimensions (designed for NumElem = 256*const)
// Local size : (64,                1,       1)
// Global size: (64*(NumElem/256),  NumFreq, time)

// Two versions of essentially the same kernel.  1 with and 1 without the time-shifting capability.
// The kernels do not strictly care if multiple data chunks are loaded.  They do a simple check that they stay in bounds.  This could be expanded
// (quite easily--see the bonus kernel) if the the output data buffer is a different size from the input buffer though the additional checks and variables could give
// a slight performance hit.

//#pragma OPENCL EXTENSION cl_amd_printf : enable
#define TIME_REDUCTION_FACTOR       4
#define LOCAL_SIZE                  64
#define NUM_COMPLEX_PAIRS           4   //4 B per uint
#define NUM_ELEMENTS_DIV_DATASIZE   (ACTUAL_NUM_ELEMENTS/NUM_COMPLEX_PAIRS)    //DATASIZE = NUM_COMPLEX_PAIRS here since each complex pair = 1 B
#define LOCAL_SIZE_x_COMPLEX_PAIRS  (LOCAL_SIZE*NUM_COMPLEX_PAIRS)
//#define NUM_TIMESTEPS               128  //total number of timesteps

__kernel void reorder_input_data_with_timeshift(__global unsigned int  *input_data,
                                                __global unsigned char *output_data,
                                                __global const    int  *remap_lookup, //a mapping of elements from FPGA ordering to natural cylinder ordering (or other)
                                                __global const    char *timeshift_map //NOTE: the map for timeshifting is based on the remapped addressing
                                                ) {
    __local unsigned char packed_temp[LOCAL_SIZE_x_COMPLEX_PAIRS];

    //Load 4 packed pairs at a time (would it be best to up this to 16 to increase bus usage for GPUs like Fury?)
    int input_address = get_local_id(0) + get_group_id(0)*LOCAL_SIZE + get_group_id(1)*NUM_ELEMENTS_DIV_DATASIZE + get_group_id(2)*NUM_ELEMENTS_DIV_DATASIZE*NUM_FREQUENCIES;
    uint temp = input_data[input_address]; //this would change with how things load (say if a vector were loaded instead)

    //unroll this
    for (int i = 0; i < NUM_COMPLEX_PAIRS; i++){
        packed_temp[get_local_id(0)*4+i] = (temp >>(i*8)) & 0xFF;
    }

    //Each output needs to be treated individually--things are not necessarily output in consecutive locations, and timeshifts make temporary arrays dangerous

    int lookup_address;//
    int output_element_address;//
    int output_timeshift;//
    int output_address;//

    //test this when unrolled, too.
    for (int i = 0; i < NUM_COMPLEX_PAIRS; i++){
        //figure out the output addresses
        lookup_address          = get_group_id(0)*LOCAL_SIZE_x_COMPLEX_PAIRS + i*LOCAL_SIZE + get_local_id(0);
        output_element_address  = remap_lookup[lookup_address];
        output_timeshift        = timeshift_map[output_element_address]; //NOTE: the map for timeshifting is based on the remapped addressing
        output_address          = output_element_address +
                                    get_group_id(1)*NUM_ELEMENTS_DIV_DATASIZE*NUM_COMPLEX_PAIRS +
                                    (output_timeshift+get_group_id(2))*NUM_ELEMENTS_DIV_DATASIZE*NUM_COMPLEX_PAIRS*NUM_FREQUENCIES;
        //save individually since scattering can go any which way with the remaps
        if (output_address >=0 && output_address < (NUM_ELEMENTS_DIV_DATASIZE*NUM_COMPLEX_PAIRS*NUM_FREQUENCIES*NUM_TIMESAMPLES))
            output_data[output_address] = packed_temp[get_local_id(0)+i*LOCAL_SIZE];
    }
}

__kernel void reorder_input_data_with_timeshift_cache_lookups(__global unsigned int  *input_data,
                                                __global unsigned char *output_data,
                                                __global const    int  *remap_lookup, //a mapping of elements from FPGA ordering to natural cylinder ordering (or other)
                                                __global const    char *timeshift_map //NOTE: the map for timeshifting is based on the remapped addressing
                                                ) {
    __local unsigned char packed_temp[LOCAL_SIZE_x_COMPLEX_PAIRS];

    //load and cache the remaps in 4 steps
    //__local int remap_lookup[LOCAL_SIZE*4];
    int remap_address[4];
    int time_shift[4];

//#pragma unroll
    for (int i = 0; i < 4; i++){
        remap_address[i] = remap_lookup[i*LOCAL_SIZE + get_local_id(0) + get_group_id(0)*LOCAL_SIZE_x_COMPLEX_PAIRS];
        time_shift[i] = timeshift_map[remap_address[i]];
    }

    //Load 4 packed pairs at a time (would it be best to up this to 16 to increase bus usage for GPUs like Fury?)
    int input_address;
    uint temp;



    //Each output needs to be treated individually--things are not necessarily output in consecutive locations, and timeshifts make temporary arrays dangerous


    //test this when unrolled, too.
    for (int j = 0; j < TIME_REDUCTION_FACTOR; j++){
        input_address = get_local_id(0) + get_group_id(0)*LOCAL_SIZE + get_group_id(1)*NUM_ELEMENTS_DIV_DATASIZE + (get_group_id(2)*TIME_REDUCTION_FACTOR+j)*NUM_ELEMENTS_DIV_DATASIZE*NUM_FREQUENCIES;
        temp = input_data[input_address]; //this would change with how things load (say if a vector were loaded instead)

//#pragma unroll
        for (int i = 0; i < NUM_COMPLEX_PAIRS; i++){
            packed_temp[get_local_id(0)*4+i] = (temp >>(i*8)) & 0xFF;
        }
        int output_address;//
//#pragma unroll
        for (int i = 0; i < NUM_COMPLEX_PAIRS; i++){
            //figure out the output addresses
            //lookup_address          = get_group_id(0)*LOCAL_SIZE_x_COMPLEX_PAIRS + i*LOCAL_SIZE + get_local_id(0);
            //output_element_address  = remap_lookup[lookup_address];
            //output_timeshift        = timeshift_map[output_element_address]; //NOTE: the map for timeshifting is based on the remapped addressing
            output_address          = remap_address[i] +
                                        get_group_id(1)*NUM_ELEMENTS_DIV_DATASIZE*NUM_COMPLEX_PAIRS +
                                        (time_shift[i]+get_group_id(2)*TIME_REDUCTION_FACTOR+j)*NUM_ELEMENTS_DIV_DATASIZE*NUM_COMPLEX_PAIRS*NUM_FREQUENCIES;
            //save individually since scattering can go any which way with the remaps
            if (output_address >=0 && output_address < (NUM_ELEMENTS_DIV_DATASIZE*NUM_COMPLEX_PAIRS*NUM_FREQUENCIES*NUM_TIMESAMPLES))
                output_data[output_address] = packed_temp[get_local_id(0)+i*LOCAL_SIZE];
        }
    }
}
//version two--without timeshifts--means that data can be moved temporarily in a local array and output in an essentially coalesced manner
//__kernel void reorder_input_data(__global unsigned int  *input_data,
//                                 __global unsigned char *output_data,
//                                 __global const    int  *remap_lookup){
//}
