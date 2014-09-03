// CASPER tools, according to JF Cliche, store complex pairs in a Real,Imaginary order, so when we have a 1 B packed pair, 
// the order should be:
//   RRRRIIII
// where Real values are in the high nibble.  Previous versions of code had used IIIIRRRR ordering within a single uchar/Byte

#include <complex.h>    // I is used to get complex parts (not i (nor j for engineers)--I suppose they realized those get used as simple loop counters often?)

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <assert.h>
#include "SFMT-src-1.4/SFMT.h" //Mersenne Twister library

#define NUM_CL_FILES                    3u
#define OPENCL_FILENAME_1               "test0xB_multifreq_multiple.cl"
#define OPENCL_FILENAME_2               "offsetAccumulator_multiple.cl"
#define OPENCL_FILENAME_2b              "offsetAccumulatorXOR_multiple.cl"
#define OPENCL_FILENAME_3               "preseed_multifreq_multiple.cl"

#define HI_NIBBLE(b)                    (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b)                    ((b) & 0x0F)

#define SDK_SUCCESS                     0u

// //////////////////////change these as desired/////////////////////////////////////////
#define NUM_ELEM                        256u//32u //minimum needs to be 32 for the kernels
#define NUM_FREQ                        1u//63u
#define ACTUAL_NUM_ELEM                 256u//16u
#define ACTUAL_NUM_FREQ                 1u//126u //product of ACTUAL_NUM_ELEM*ACTUAL_NUM_FREQ should equal NUM_ELEM*NUM_FREQ
#define HDF5_FREQ                       1024
#define UPPER_TRIANGLE                  1
#define INTERLEAVED                     0
#define XOR_MANUAL                      0
#define TIMER_FOR_PROCESSING_ONLY       0

#define NUM_DATA_SETS                   1

#define NUM_TIMESAMPLES                 4u*1024u*1024u //
#define NUM_REPEATS_GPU                 1000u
// ////////////////////////////////////////////////////////////////////////////////////

#define N_STAGES                        2 //write to CL_Mem, Kernel (Read is done after many runs since answers are accumulated)
#define N_QUEUES                        2 //have 2 separate queues so transfer and process paths can be queued nicely

#define DEBUG                           0
#define DEBUG_GENERATOR                 0
//check pagesize:
//getconf PAGESIZE
// result: 4096
#define PAGESIZE_MEM                    4096u
#define TIME_ACCUM                      256u

//enumerations/definitions: don't change
#define GENERATE_DATASET_CONSTANT       1u
#define GENERATE_DATASET_RAMP_UP        2u
#define GENERATE_DATASET_RAMP_DOWN      3u
#define GENERATE_DATASET_RANDOM_SEEDED  4u
#define ALL_FREQUENCIES                -1

//parameters for data generator: you can change these. (Values will be shifted and clipped as needed, so these are signed 4bit numbers for input)
#define GEN_TYPE                        GENERATE_DATASET_RANDOM_SEEDED
#define GEN_DEFAULT_SEED                42u
#define GEN_DEFAULT_RE                  0u
#define GEN_DEFAULT_IM                  0u
#define GEN_INITIAL_RE                  0u //-8
#define GEN_INITIAL_IM                  0u //7
#define GEN_FREQ                        ALL_FREQUENCIES
#define GEN_REPEAT_RANDOM               1u

#define CHECKING_VERBOSE                0u

double e_time(void){
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

//error codes from an amd firepro demo:
char* oclGetOpenCLErrorCodeStr(cl_int input)
{
    int errorCode = (int)input;
    switch(errorCode)
    {
        case CL_SUCCESS:
            return (char*) "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return (char*) "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return (char*) "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return (char*) "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return (char*) "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return (char*) "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return (char*) "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return (char*) "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return (char*) "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return (char*) "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return (char*) "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return (char*) "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return (char*) "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return (char*) "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return (char*) "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:
            return (char*) "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:
            return (char*) "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:
            return (char*) "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:
            return (char*) "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return (char*) "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:
            return (char*) "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return (char*) "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return (char*) "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return (char*) "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return (char*) "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return (char*) "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return (char*) "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return (char*) "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return (char*) "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return (char*) "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return (char*) "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return (char*) "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return (char*) "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return (char*) "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return (char*) "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return (char*) "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return (char*) "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return (char*) "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return (char*) "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return (char*) "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return (char*) "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return (char*) "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return (char*) "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return (char*) "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return (char*) "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return (char*) "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return (char*) "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return (char*) "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return (char*) "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return (char*) "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return (char*) "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return (char*) "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return (char*) "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return (char*) "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return (char*) "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:
            return (char*) "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:
            return (char*) "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:
            return (char*) "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return (char*) "CL_INVALID_DEVICE_PARTITION_COUNT";
        default:
            return (char*) "unknown error code";
    }

    return (char*) "unknown error code";
}

int offset_and_clip_value(int input_value, int offset_value, int min_val, int max_val){
    int offset_and_clipped = input_value + offset_value;
    if (offset_and_clipped > max_val)
        offset_and_clipped = max_val;
    else if (offset_and_clipped < min_val)
        offset_and_clipped = min_val;
    return(offset_and_clipped);
}

void generate_char_data_set(int generation_Type,
                            int random_seed,
                            int default_real,
                            int default_imaginary,
                            int initial_real,
                            int initial_imaginary,
                            int single_frequency,
                            int num_timesteps,
                            int num_frequencies,
                            int num_elements,
                            int num_data_sets,
                            unsigned char *packed_data_set){

    sfmt_t sfmt; //for the Mersenne Twister
    if (single_frequency > num_frequencies || single_frequency < 0)
        single_frequency = ALL_FREQUENCIES;

    //printf("single_frequency: %d \n",single_frequency);
    default_real =offset_and_clip_value(default_real,8,0,15);
    default_imaginary = offset_and_clip_value(default_imaginary,8,0,15);
    initial_real = offset_and_clip_value(initial_real,8,0,15);
    initial_imaginary = offset_and_clip_value(initial_imaginary,8,0,15);
    unsigned char clipped_offset_default_real = (unsigned char) default_real;
    unsigned char clipped_offset_default_imaginary = (unsigned char) default_imaginary;
    unsigned char clipped_offset_initial_real = (unsigned char) initial_real;
    unsigned char clipped_offset_initial_imaginary = (unsigned char) initial_imaginary;
    unsigned char temp_output;


    //printf("clipped_offset_initial_real: %d, clipped_offset_initial_imaginary: %d, clipped_offset_default_real: %d, clipped_offset_default_imaginary: %d\n", clipped_offset_initial_real, clipped_offset_initial_imaginary, clipped_offset_default_real, clipped_offset_default_imaginary);
    for (int m = 0; m < num_data_sets; m++){
        if (generation_Type == GENERATE_DATASET_RANDOM_SEEDED){
            sfmt_init_gen_rand(&sfmt, random_seed);
            //srand(random_seed);
        }

        for (int k = 0; k < num_timesteps; k++){
            //printf("k: %d\n",k);
            if (generation_Type == GENERATE_DATASET_RANDOM_SEEDED && GEN_REPEAT_RANDOM){
                sfmt_init_gen_rand(&sfmt, random_seed);
                //srand(random_seed);
            }
            for (int j = 0; j < num_frequencies; j++){
                if (DEBUG_GENERATOR && k == 0)
                    printf("j: %d Vals: ",j);
                for (int i = 0; i < num_elements; i++){
                    int currentAddress = m*num_timesteps*num_frequencies*num_elements +k*num_frequencies*num_elements + j*num_elements + i;
                    unsigned char new_real;
                    unsigned char new_imaginary;
                    switch (generation_Type){
                        case GENERATE_DATASET_CONSTANT:
                            new_real = clipped_offset_initial_real;
                            new_imaginary = clipped_offset_initial_imaginary;
                            break;
                        case GENERATE_DATASET_RAMP_UP:
                            new_real = (j+clipped_offset_initial_real+i)%16;
                            new_imaginary = (j+clipped_offset_initial_imaginary+i)%16;
                            break;
                        case GENERATE_DATASET_RAMP_DOWN:
                            new_real = 15-((j+clipped_offset_initial_real+i)%16);
                            new_imaginary = 15 - ((j+clipped_offset_initial_imaginary+i)%16);
                            break;
                        case GENERATE_DATASET_RANDOM_SEEDED:
                            new_real = (unsigned char)(floor(sfmt_genrand_res53(&sfmt)*16));//rand()%16; //to put the pseudorandom value in the range 0-15
                            new_imaginary = (unsigned char)(floor(sfmt_genrand_res53(&sfmt)*16));//rand()%16;
                            break;
                        default: //shouldn't happen, but in case it does, just assign the default values everywhere
                            new_real = clipped_offset_default_real;
                            new_imaginary = clipped_offset_default_imaginary;
                            break;
                    }

                    if (single_frequency == ALL_FREQUENCIES){
                        temp_output = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                        if (XOR_MANUAL){
                            temp_output = temp_output ^ 0x88; //bit flip on sign bit shifts the value by 8: makes unsigned into signed and vice versa.  Currently turns back into signed
                        }
                        packed_data_set[currentAddress] = temp_output;
                    }
                    else{
                        if (j == single_frequency){
                            temp_output = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                        }
                        else{
                            temp_output = ((clipped_offset_default_real<<4) & 0xF0) + (clipped_offset_default_imaginary & 0x0F);
                        }
                        if (XOR_MANUAL){
                            temp_output = temp_output ^ 0x88; //bit flip on sign bit shifts the value by 8: makes unsigned into signed and vice versa.  Currently turns back into signed
                        }
                        packed_data_set[currentAddress] = temp_output;
                    }
                    if (DEBUG_GENERATOR && k == 0)
                        printf("%d ",packed_data_set[currentAddress]);
                }
                if (DEBUG_GENERATOR && k == 0)
                    printf("\n");
            }
        }
    }

    if (DEBUG_GENERATOR)
        printf("END OF DATASET\n");
    return;
}

int reorder_data_phaseB_breakData (int num_timesteps,
                                   int num_frequencies,
                                   int num_elements,
                                   unsigned char *packed_data_set){
    //data, when created, has the different elements grouped together.  In Phase B of the test plan, data is not arranged in this manner--
    //data is instead packed such that 8 elements are grouped together for NUM_TIMESAMPLES*NUM_FREQ groups
    //Need to reorganize data to test the kernel that processes data in that format.
    int n_elements_div_2 = num_elements/2;
    unsigned char *packed_data_set2 = (unsigned char*)malloc(num_timesteps*num_frequencies*num_elements*sizeof(unsigned char));
    if (packed_data_set2 == NULL){
        printf("Error allocating memory for reordering of data\n");
        return(-1);
    }

    //split data up
    for (int k = 0; k < num_timesteps; k++){
        for (int j = 0; j < num_frequencies; j++){
            for (int i = 0; i < num_elements; i++){
                int inputAddress = k*num_frequencies*num_elements+j*num_elements+i;
                int outputAddress = k*num_frequencies*n_elements_div_2 + j*n_elements_div_2;
                if (i < num_elements/2){
                    outputAddress += i;
                }
                else{
                    outputAddress += num_timesteps*num_frequencies*n_elements_div_2 + i - n_elements_div_2;
                }
                packed_data_set2[outputAddress] = packed_data_set[inputAddress];
            }
        }
    }
    //copy back to original array
    int index = 0;
    for (int k = 0; k < num_timesteps; k++){
        for (int j = 0; j < num_frequencies; j++){
            for (int i = 0; i < num_elements; i++){
                packed_data_set[index] = packed_data_set2[index];
                index++;
            }
        }
    }

    free(packed_data_set2);
    return(0);
}

int reorder_data_interleave_2_frequencies (int num_timesteps,
                                           int num_frequencies,
                                           int num_elements,
                                           int num_data_sets,
                                           unsigned char *packed_data_set){
    //data, when created, has the different elements grouped together.  In this phase of the 16 element correlator
    //the data needs to be interleaved such that F0El0 F1El0 F0El1 F1El0 ... F0El(max-1) F1El(max-1) F2El0 F3El0 F2El1 F3El0 ... F2El(max-1) F3El(max-1)....
    //This function does takes data F0El0 F0El1 ... F0El(max-1) F1El0 F1El1 ... F1El(max-1) ....

    unsigned char *packed_data_set2 = (unsigned char*)malloc(num_timesteps*num_frequencies*num_elements*sizeof(unsigned char));
    if (packed_data_set2 == NULL){
        printf("Error allocating memory for reordering of data\n");
        return(-1);
    }

    //split data up
    int outputAddress = 0;
    for (int m = 0; m < num_data_sets; m++){
        for (int k = 0; k < num_timesteps; k++){
            for (int j = 0; j < num_frequencies/2; j++){
                for (int i = 0; i < num_elements; i++){
                    int inputAddress = m*num_timesteps*num_frequencies*num_elements + k*num_frequencies*num_elements+j*num_elements*2+i;
                    packed_data_set2[outputAddress++] = packed_data_set[inputAddress];
                    inputAddress += num_elements;
                    packed_data_set2[outputAddress++] = packed_data_set[inputAddress];
                }
            }
        }
    }
    //copy back to original array
    int index = 0;
    for (int m = 0; m < num_data_sets; m++){
        for (int k = 0; k < num_timesteps; k++){
            for (int j = 0; j < num_frequencies; j++){
                for (int i = 0; i < num_elements; i++){
                    packed_data_set[index] = packed_data_set2[index];
                    index++;
                }
            }
        }
    }

    free(packed_data_set2);
    return(0);
}

int drop_packets_pseudo_random(double drop_fraction,
                               int packet_size,
                               int drop_value,
                               int drop_seed,
                               int array_length,
                               unsigned char *packed_data_set){

    unsigned char *drop_vals = (unsigned char *)malloc(packet_size*sizeof(unsigned char));
    if (drop_vals == NULL){
        printf("error allocating memory in drop_packets_pseudo_random\n");
        return (-1);
    }

    drop_value = offset_and_clip_value(drop_value, 8, 0, 15);
    unsigned char clipped_offset_drop_value = (unsigned char) drop_value;
    unsigned char unsigned_packed_drop_val = ((clipped_offset_drop_value<<4) & 0xF0) + (clipped_offset_drop_value & 0x0F);
    memset(drop_vals,unsigned_packed_drop_val,packet_size);

    //set pseudorandom number set
    srand(drop_seed);
    if (drop_fraction != 0){
        int one_over_drop_fraction = (int) (1./drop_fraction);
        for (int i = 0; i < array_length/packet_size; i++){
            if (rand() % one_over_drop_fraction == 1){
                memcpy(&packed_data_set[i*packet_size],drop_vals,packet_size);
            }
        }
    }
    return (0);
}



void print_element_data(int num_timesteps, int num_frequencies, int num_elements, int particular_frequency, unsigned char *data){
    printf("Number of timesteps to print: %d, ", num_timesteps);
    if (particular_frequency == ALL_FREQUENCIES)
        printf("number of frequency bands: %d, number of elements: %d\n", num_frequencies, num_elements);
    else
        printf("frequency band: %d, number of elements: %d\n", particular_frequency, num_elements);

    for (int k = 0; k < num_timesteps; k++){
        if (num_timesteps > 1){
            printf("Time Step %d\n", k);
        }
        printf("            ");
        for (int header_i = 0; header_i < num_elements; header_i++){
            printf("%3dR %3dI ", header_i, header_i);
        }
        printf("\n");
        for (int j = 0; j < num_frequencies; j++){
            if (particular_frequency == ALL_FREQUENCIES || particular_frequency == j){
                if (particular_frequency != j)
                    printf("Freq: %4d: ", j);

                for (int i = 0; i < num_elements; i++){
                    unsigned char temp = data[k*num_frequencies*num_elements+j*num_elements+i];
                    if (XOR_MANUAL){
                        temp = temp ^ 0x88;
                    }
                    printf("%4d %4d ",(int)(HI_NIBBLE(temp))-8,(int)(LO_NIBBLE(temp))-8);
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}

int cpu_data_generate_and_correlate(int num_timesteps, int num_frequencies, int num_elements, int num_data_sets, int *correlated_data){
    //correlatedData will be returned as num_frequencies blocks, each num_elements x num_elements x 2

    //generate a dataset that should be the same as what the gpu is testing
    //dataset will be num_timesteps x num_frequencies x num_elements large
    unsigned char *generated = (unsigned char *)malloc(num_timesteps*num_frequencies*num_elements*num_data_sets*sizeof(unsigned char));
    //check the array was allocated properly
    if (generated == NULL){
        printf ("Error allocating memory: cpu_data_generate_and_correlate\n");
        return (-1);
    }

    generate_char_data_set(GEN_TYPE,GEN_DEFAULT_SEED,GEN_DEFAULT_RE, GEN_DEFAULT_IM,GEN_INITIAL_RE,GEN_INITIAL_IM,GEN_FREQ, num_timesteps, num_frequencies, num_elements, num_data_sets, generated);

    if (CHECKING_VERBOSE){
        print_element_data(1, num_frequencies, num_elements, ALL_FREQUENCIES, generated);
    }

    unsigned char temp_char;
    //correlate based on generated data
    for (int m = 0; m < num_data_sets; m++){
        for (int k = 0; k < num_timesteps; k++){
            for (int j = 0; j < num_frequencies; j++){
                for (int element_y = 0; element_y < num_elements; element_y++){
                    temp_char = generated[m*num_timesteps*num_frequencies*num_elements + k*num_frequencies*num_elements + j*num_elements + element_y];
                    if (XOR_MANUAL){
                        temp_char = temp_char ^ 0x88;
                    }
                    int element_y_re = (int)(HI_NIBBLE(temp_char)) - 8; //-8 is to put the number back in the range -8 to 7 from 0 to 15
                    int element_y_im = (int)(LO_NIBBLE(temp_char)) - 8;
                    for (int element_x = 0; element_x < num_elements; element_x++){
                        temp_char = generated[m*num_timesteps*num_frequencies*num_elements + k*num_frequencies*num_elements + j*num_elements + element_x];
                        if (XOR_MANUAL){
                            temp_char = temp_char ^ 0x88;
                        }
                        int element_x_re = (int)(HI_NIBBLE(temp_char)) - 8;
                        int element_x_im = (int)(LO_NIBBLE(temp_char)) - 8;
                        if (k != 0){
                            correlated_data[(m*num_frequencies*num_elements*num_elements + j*num_elements*num_elements+element_y*num_elements+element_x)*2]   += element_x_re*element_y_re + element_x_im*element_y_im;
                            correlated_data[(m*num_frequencies*num_elements*num_elements + j*num_elements*num_elements+element_y*num_elements+element_x)*2+1] += element_x_im*element_y_re - element_x_re*element_y_im;
                        }
                        else{
                            correlated_data[(m*num_frequencies*num_elements*num_elements + j*num_elements*num_elements+element_y*num_elements+element_x)*2]   = element_x_re*element_y_re + element_x_im*element_y_im;
                            correlated_data[(m*num_frequencies*num_elements*num_elements + j*num_elements*num_elements+element_y*num_elements+element_x)*2+1] = element_x_im*element_y_re - element_x_re*element_y_im;
                        }
                    }
                }
            }
        }
    }
    //clean up parameters as needed
    free(generated);
    return (0);
}

int cpu_data_generate_and_correlate_upper_triangle_only(int num_timesteps, int num_frequencies, int num_elements, int num_data_sets, int *correlated_data_triangle){
    //correlatedData will be returned as num_frequencies blocks, each num_elements x num_elements x 2

    //generate a dataset that should be the same as what the gpu is testing
    //dataset will be num_timesteps x num_frequencies x num_elements large
    unsigned char *generated = (unsigned char *)malloc(num_timesteps*num_frequencies*num_elements*num_data_sets*sizeof(unsigned char));
    //check the array was allocated properly
    if (generated == NULL){
        printf ("Error allocating memory: cpu_data_generate_and_correlate\n");
        return (-1);
    }

    generate_char_data_set(GEN_TYPE,GEN_DEFAULT_SEED,GEN_DEFAULT_RE, GEN_DEFAULT_IM,GEN_INITIAL_RE,GEN_INITIAL_IM,GEN_FREQ, num_timesteps, num_frequencies, num_elements, num_data_sets,generated);

    if (CHECKING_VERBOSE){
        print_element_data(1, num_frequencies, num_elements, ALL_FREQUENCIES, generated);
    }

    unsigned char temp_char;
    //correlate based on generated data
    for (int m = 0; m < num_data_sets; m++){
        for (int k = 0; k < num_timesteps; k++){
            int output_counter = m*num_frequencies*num_elements*(num_elements+1);// /2*2;
            for (int j = 0; j < num_frequencies; j++){
                for (int element_y = 0; element_y < num_elements; element_y++){
                    temp_char = generated[m*num_timesteps*num_frequencies*num_elements + k*num_frequencies*num_elements+j*num_elements+element_y];
                    if (XOR_MANUAL){
                        temp_char = temp_char ^ 0x88;
                    }
                    int element_y_re = (int)(HI_NIBBLE(temp_char)) - 8; //-8 is to put the number back in the range -8 to 7 from 0 to 15
                    int element_y_im = (int)(LO_NIBBLE(temp_char)) - 8;
                    for (int element_x = element_y; element_x < num_elements; element_x++){
                        temp_char = generated[m*num_timesteps*num_frequencies*num_elements + k*num_frequencies*num_elements+j*num_elements+element_x];
                        if (XOR_MANUAL){
                            temp_char = temp_char ^ 0x88;
                        }
                        int element_x_re = (int)(HI_NIBBLE(temp_char)) - 8;
                        int element_x_im = (int)(LO_NIBBLE(temp_char)) - 8;
                        if (k != 0){
                            correlated_data_triangle[output_counter++] += element_x_re*element_y_re + element_x_im*element_y_im;
                            correlated_data_triangle[output_counter++] += element_x_im*element_y_re - element_x_re*element_y_im;
                        }
                        else{
                            correlated_data_triangle[output_counter++] = element_x_re*element_y_re + element_x_im*element_y_im;
                            correlated_data_triangle[output_counter++] = element_x_im*element_y_re - element_x_re*element_y_im;
                        }
                    }
                }
            }
        }
    }

    //clean up parameters as needed
    free(generated);
    return (0);
}

void reorganize_32_to_16_feed_GPU_Correlated_Data(int actual_num_frequencies, int actual_num_elements, int num_data_sets, int *correlated_data){
    //data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements exist.
    //This is equivalent to processing 2 elements at the same time, where the desired correlations live in the first and fourth quadrants
    //This function is to reorganize the data so that comparisons can be done more easily

    //The input dataset is larger than the output, so can reorganize in the same array

    int input_frequencies = actual_num_frequencies/2;
    int input_elements = actual_num_elements*2;
    int address = 0;
    int address_out = 0;
    for (int m = 0; m < num_data_sets; m++){
        for (int freq = 0; freq < input_frequencies; freq++){
            for (int element_y = 0; element_y < input_elements; element_y++){
                for (int element_x = 0; element_x < input_elements; element_x++){
                    if (element_x < actual_num_elements && element_y < actual_num_elements){
                        correlated_data[address_out++] = correlated_data[address++];
                        correlated_data[address_out++] = correlated_data[address++]; //real and imaginary at each spot
                    }
                    else if (element_x >=actual_num_elements && element_y >=actual_num_elements){
                        correlated_data[address_out++] = correlated_data[address++];
                        correlated_data[address_out++] = correlated_data[address++];
                    }
                    else
                        address += 2;
                }
            }
        }
    }
    return;
}

void reorganize_32_to_16_feed_GPU_Correlated_Data_Interleaved(int actual_num_frequencies, int actual_num_elements, int num_data_sets, int *correlated_data){
    //data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements exist.
    //There are two frequencies interleaved...  Need to sort the output values properly
    //This function is to reorganize the data so that comparisons can be done more easily

    //The input dataset is larger than the output, so can reorganize in the same array
    int *temp_output = (int *)malloc(actual_num_elements*actual_num_elements*actual_num_frequencies*num_data_sets*2*sizeof(int));

    int input_elements = actual_num_elements*2;
    int address = 0;
    int address_out = 0;
    for (int m = 0; m < num_data_sets; m++){
        for (int freq = 0; freq < actual_num_frequencies; freq++){
            address = m*actual_num_frequencies*input_elements*input_elements*2 + (freq >>1) * input_elements*input_elements *2; // freq>>1 == freq/2 
            for (int element_y = 0; element_y < input_elements; element_y++){
                for (int element_x = 0; element_x < input_elements; element_x++){
                    if (freq & 1){//odd frequencies
                        if ((element_x & 1) && (element_y & 1)){
                            temp_output[address_out++] = correlated_data[address++];
                            temp_output[address_out++] = correlated_data[address++];
                        }
                        else{
                            address += 2;
                        }
                    }
                    else{ // even frequencies
                        if ((!(element_x & 1)) && (!(element_y & 1))){
                            temp_output[address_out++] = correlated_data[address++];
                            temp_output[address_out++] = correlated_data[address++];
                        }
                        else{
                            address += 2;
                        }
                    }
                }
            }
        }
    }

    //copy the results back into correlated_data
    for (int i = 0; i < actual_num_frequencies * actual_num_elements*actual_num_elements * num_data_sets * 2; i++)
        correlated_data[i] = temp_output[i];

    free(temp_output);
    return;
}


void reorganize_GPU_to_full_Matrix_for_comparison(int block_side_length, int num_blocks, int actual_num_frequencies, int actual_num_elements, int num_data_sets, int *gpu_data, int *final_matrix){
    //takes the output data, grouped in blocks of block_dim x block_dim x 2 (complex pairs (ReIm)of ints), and fills a num_elements x num_elements x 2

    for (int m = 0; m < num_data_sets; m++){
        for (int frequency_bin = 0; frequency_bin < actual_num_frequencies; frequency_bin++ ){
            int block_x_ID = 0;
            int block_y_ID = 0;
            int num_blocks_x = actual_num_elements/block_side_length;
            int block_check = num_blocks_x;

            for (int block_ID = 0; block_ID < num_blocks; block_ID++){
                if (block_ID == block_check){ //at the end of a row in the upper triangle
                    num_blocks_x--;
                    block_check += num_blocks_x;
                    block_y_ID++;
                    block_x_ID = block_y_ID;
                }
                for (int y_ID_local = 0; y_ID_local < block_side_length; y_ID_local++){
                    int y_ID_global = block_y_ID * block_side_length + y_ID_local;
                    for (int x_ID_local = 0; x_ID_local < block_side_length; x_ID_local++){
                        int GPU_address = m*actual_num_frequencies*num_blocks*block_side_length*block_side_length*2 + frequency_bin*(num_blocks*block_side_length*block_side_length*2) + block_ID *(block_side_length*block_side_length*2) + y_ID_local*block_side_length*2+x_ID_local*2; ///TO DO :simplify this statement after getting everything working
                        int x_ID_global = block_x_ID * block_side_length + x_ID_local;
                        if (x_ID_global >= y_ID_global){
                            if (x_ID_global > y_ID_global){ //store the conjugate: x and y addresses get swapped and the imaginary value is the negative of the original value
                                final_matrix[(m*actual_num_frequencies*actual_num_elements*actual_num_elements +frequency_bin*actual_num_elements*actual_num_elements+x_ID_global*actual_num_elements+y_ID_global)*2]   =  gpu_data[GPU_address];
                                final_matrix[(m*actual_num_frequencies*actual_num_elements*actual_num_elements +frequency_bin*actual_num_elements*actual_num_elements+x_ID_global*actual_num_elements+y_ID_global)*2+1] = -gpu_data[GPU_address+1];
                            }
                            //store the value for the upper triangle
                            final_matrix[(m*actual_num_frequencies*actual_num_elements*actual_num_elements +frequency_bin*actual_num_elements*actual_num_elements+y_ID_global*actual_num_elements+x_ID_global)*2]   = gpu_data[GPU_address];
                            final_matrix[(m*actual_num_frequencies*actual_num_elements*actual_num_elements +frequency_bin*actual_num_elements*actual_num_elements+y_ID_global*actual_num_elements+x_ID_global)*2+1] = gpu_data[GPU_address+1];
                        }
                    }
                }
                //printf("block_ID: %d, block_y_ID: %d, block_x_ID: %d\n", block_ID, block_y_ID, block_x_ID);
                //update block offset values
                block_x_ID++;
            }
        }
    }
    return;
}

void reorganize_GPU_to_upper_triangle(int block_side_length, int num_blocks, int actual_num_frequencies, int actual_num_elements, int num_data_sets, int *gpu_data, int *final_matrix){

    for (int m = 0; m < num_data_sets; m++){
        int GPU_address = m*(actual_num_frequencies*(num_blocks*(block_side_length*block_side_length*2))); //we go through the gpu data sequentially and map it to the proper locations in the output array
        for (int frequency_bin = 0; frequency_bin < actual_num_frequencies; frequency_bin++ ){
            int block_x_ID = 0;
            int block_y_ID = 0;
            int num_blocks_x = actual_num_elements/block_side_length;
            int block_check = num_blocks_x;
            int frequency_offset = m*actual_num_frequencies*(actual_num_elements* (actual_num_elements+1))/2 + frequency_bin * (actual_num_elements* (actual_num_elements+1))/2;// frequency_bin * number of items in an upper triangle

            for (int block_ID = 0; block_ID < num_blocks; block_ID++){
                if (block_ID == block_check){ //at the end of a row in the upper triangle
                    num_blocks_x--;
                    block_check += num_blocks_x;
                    block_y_ID++;
                    block_x_ID = block_y_ID;
                }

                for (int y_ID_local = 0; y_ID_local < block_side_length; y_ID_local++){

                    for (int x_ID_local = 0; x_ID_local < block_side_length; x_ID_local++){

                        int x_ID_global = block_x_ID * block_side_length + x_ID_local;
                        int y_ID_global = block_y_ID * block_side_length + y_ID_local;

                        /// address_1d_output = frequency_offset, plus the number of entries in the rectangle area (y_ID_global*actual_num_elements), minus the number of elements in lower triangle to that row (((y_ID_global-1)*y_ID_global)/2), plus the contributions to the address from the current row (x_ID_global - y_ID_global)
                        int address_1d_output = frequency_offset + y_ID_global*actual_num_elements - ((y_ID_global-1)*y_ID_global)/2 + (x_ID_global - y_ID_global); 

                        if (block_x_ID != block_y_ID){ //when we are not in the diagonal blocks
                            final_matrix[address_1d_output*2  ] = gpu_data[GPU_address++];
                            final_matrix[address_1d_output*2+1] = gpu_data[GPU_address++];
                        }
                        else{ // the special case needed to deal with the diagonal pieces
                            if (x_ID_local >= y_ID_local){
                                final_matrix[address_1d_output*2  ] = gpu_data[GPU_address++];
                                final_matrix[address_1d_output*2+1] = gpu_data[GPU_address++];
                            }
                            else{
                                GPU_address += 2;
                            }
                        }
                    }
                }
                //offset_GPU += (block_side_length*block_side_length);
                //update block offset values
                block_x_ID++;
            }
        }
    }
    return;
}

void shuffle_data_to_frequency_major_output (int num_frequencies_final, int num_frequencies, int num_elements, int num_data_sets, int *input_data, double complex *output_data){
    //input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of complex visibilities for frequencies
    //output array will be sparsely to moderately filled, so loop such that writing is done in sequential order
    int num_complex_visibilities = (num_elements*(num_elements+1))/2;
    for (int m = 0; m < num_data_sets; m++){
        int output_counter = m*num_frequencies_final*num_complex_visibilities;
        for (int data_count = 0; data_count < num_complex_visibilities; data_count++){
            for (int freq_count = 0; freq_count < num_frequencies_final; freq_count++){
                if (freq_count < num_frequencies){
                    output_data [output_counter++] = (double)input_data[(m*num_frequencies*num_complex_visibilities +freq_count*num_complex_visibilities + data_count)*2]
                                                    + I * (double)input_data[(m*num_frequencies*num_complex_visibilities +freq_count*num_complex_visibilities + data_count)*2+1];
                    //output_data [(data_count*num_frequencies_final + freq_count)] = (double)input_data[(freq_count*num_complex_visibilities + data_count)*2] + I * (double)input_data[(freq_count*num_complex_visibilities + data_count)*2+1];
                }
                else{
                    output_data [output_counter++] = (double)0.0 + I * (double)0.0;
                    //output_data [(data_count*num_frequencies_final + freq_count)] = (double)0.0 + I * (double)0.0;
                }
            }
        }
    }
    return;
}

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion (int num_frequencies_final, int actual_num_frequencies, int num_data_sets, int *input_data, double complex *output_data){
    //input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of complex visibilities for frequencies
    //output array will be sparsely to moderately filled, so loop such that writing is done in sequential order
    //int num_complex_visibilities = 136;//16*(16+1)/2; //(n*(n+1)/2)
    for (int m = 0; m < num_data_sets; m++){
        int output_counter = m*num_frequencies_final*136;//16*(16+1)/2 = 136
        for (int y = 0; y < 16; y++){
            for (int x = y; x < 16; x++){
                for (int freq_count = 0; freq_count < num_frequencies_final; freq_count++){
                    if (freq_count < actual_num_frequencies){
                        int input_index = (m*actual_num_frequencies*256 + freq_count*256 + y*16 + x)*2; //num of array elements in a block: 16*16 = 256 if you are curious where that came from
                        output_data [output_counter++] = (double) input_data[input_index]
                                                        + I * (double) input_data[input_index+1];
                        //output_data [(data_count*num_frequencies_final + freq_count)] = (double)input_data[input_index] + I * (double)input_data[input_index+1];
                    }
                    else{
                        output_data [output_counter++] = (double)0.0 + I * (double)0.0;
                        //output_data [(data_count*num_frequencies_final + freq_count)] = (double)0.0 + I * (double)0.0;
                    }
                }
            }
        }
    }
    return;
}

void reorganize_data_16_element_with_triangle_conversion (int num_frequencies_final, int actual_num_frequencies, int num_data_sets, int *input_data, int *output_data){
    //input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of complex visibilities for frequencies
    //output array will be sparsely to moderately filled, so loop such that writing is done in sequential order
    for (int m = 0; m < num_data_sets; m++){
        int output_counter = m * num_frequencies_final * 136 *2;
        for (int freq_count = 0; freq_count < num_frequencies_final; freq_count++){
            for (int y = 0; y < 16; y++){
                for (int x = y; x < 16; x++){
                    if (freq_count < actual_num_frequencies){
                        int input_index = (m*actual_num_frequencies*256 + freq_count*256 + y*16 + x)*2; //blocks of data are 16 x 16 = 256 and row_stride is 16
                        output_data [output_counter++] = input_data[input_index];
                        output_data [output_counter++] = input_data[input_index+1];
                        //output_data [(data_count*num_frequencies_final + freq_count)] = (double)input_data[input_index] + I * (double)input_data[input_index+1];
                    }
                    else{
                        output_data [output_counter++] = 0;
                        output_data [output_counter++] = 0;
                        //output_data [(data_count*num_frequencies_final + freq_count)] = (double)0.0 + I * (double)0.0;
                    }
                }
            }
        }
    }
    return;
}

void correct_GPU_correlation_results (int num_timesteps, int num_frequencies, int num_elements, int *correlated_data_GPU, int *accumulates){
    //since data are processed within the GPU as unsigned values (packing to optimize calculations) correction terms must be taken account of
    //Future versions of kernels will automatically correct on the GPU so this function will not be required then
    int address = 0;
    int offset = num_timesteps * 128;
    for (int freq = 0; freq < num_frequencies; freq++){
        for (int element_y = 0; element_y < num_elements; element_y++){
            int element_y_re = accumulates[(freq*num_elements+element_y)*2];
            int element_y_im = accumulates[(freq*num_elements+element_y)*2+1];
            for (int element_x = 0; element_x < num_elements; element_x++){
                int element_x_re = accumulates[(freq*num_elements+element_x)*2];
                int element_x_im = accumulates[(freq*num_elements+element_x)*2+1];
                correlated_data_GPU[address++] += offset - 8 * (element_x_re + element_x_im + element_y_re + element_y_im);
                correlated_data_GPU[address++] += 8*(element_x_re-element_x_im-element_y_re+element_y_im);
            }
        }
    }
    return;
}

void correct_GPU_correlation_results_Split (int num_timesteps, int num_frequencies, int num_elements, int *correlated_data_GPU, int *accumulates){
    //since data are processed within the GPU as unsigned values (packing to optimize calculations) correction terms must be taken account of
    //Future versions of kernels will automatically correct on the GPU so this function will not be required then
    int address = 0;
    int offset = num_timesteps * 128;
    for (int freq = 0; freq < num_frequencies; freq++){
        for (int element_y = 0; element_y < num_elements; element_y++){
            int y_addr = freq*num_elements/2;
            if (element_y < num_elements/2){
                y_addr += element_y;
            }
            else{
                y_addr += num_frequencies*num_elements/2 + element_y - num_elements/2;
            }
            int element_y_re = accumulates[y_addr*2];
            int element_y_im = accumulates[y_addr*2+1];
            for (int element_x = 0; element_x < num_elements; element_x++){
                int x_addr = freq*num_elements/2;
                if (element_x < num_elements/2){
                    x_addr += element_x;
                }
                else{
                    x_addr += num_frequencies*num_elements/2 + element_x - num_elements/2;
                }
                int element_x_re = accumulates[x_addr*2];
                int element_x_im = accumulates[x_addr*2+1];
                correlated_data_GPU[address++] += offset - 8 * (element_x_re + element_x_im + element_y_re + element_y_im);
                correlated_data_GPU[address++] += 8*(element_x_re-element_x_im-element_y_re+element_y_im);
            }
        }
    }
    return;
}

void compare_NSquared_correlator_results ( int *num_err, int64_t *err_2, int num_frequencies, int num_elements, int num_data_sets, int *data_set_GPU, int *data_set_CPU, double *ratio_GPU_div_CPU, double *phase_difference, int verbosity){
    //this will compare the values of the two arrays and give information about the comparison
    int address = 0;
    int local_Address = 0;
    *num_err = 0;
    *err_2 = 0;
    int max_error = 0;
    int amplitude_squared_error;
    double amplitude_squared_CPU;
    double amplitude_squared_GPU;
    double phase_angle_CPU;
    double phase_angle_GPU;
    for (int m = 0; m < num_data_sets; m++){
        for (int freq = 0; freq < num_frequencies; freq++){
            for (int element_y = 0; element_y < num_elements; element_y++){
                for (int element_x = 0; element_x < num_elements; element_x++){
                    //compare real results
                    int data_Real_GPU = data_set_GPU[address];
                    int data_Real_CPU = data_set_CPU[address++];
                    int difference_real = data_Real_GPU - data_Real_CPU;
                    //compare imaginary results
                    int data_Imag_GPU = data_set_GPU[address];
                    int data_Imag_CPU = data_set_CPU[address++];
                    int difference_imag = data_Imag_GPU - data_Imag_CPU;

                    //get amplitude_squared
                    amplitude_squared_CPU = data_Real_CPU*data_Real_CPU + data_Imag_CPU*data_Imag_CPU;
                    amplitude_squared_GPU = data_Real_GPU*data_Real_GPU + data_Imag_GPU*data_Imag_GPU;
                    phase_angle_CPU = atan2((double)data_Imag_CPU,(double)data_Real_CPU);
                    phase_angle_GPU = atan2((double)data_Imag_GPU,(double)data_Real_GPU);

                    if (amplitude_squared_CPU != 0){
                        ratio_GPU_div_CPU[local_Address] = amplitude_squared_GPU/amplitude_squared_CPU;
                    }
                    else{
                        ratio_GPU_div_CPU[local_Address] = -1;//amplitude_squared_GPU/amplitude_squared_CPU;
                    }

                    phase_difference[local_Address++] = phase_angle_GPU - phase_angle_CPU;

                    if (difference_real != 0 || difference_imag !=0){
                        (*num_err)++;
                        if (verbosity ){
                            printf ("freq: %6d element_x: %6d element_y: %6d Real CPU/GPU %8d %8d Imaginary CPU/GPU %8d %8d ERR: %7d\n",freq, element_x, element_y, data_Real_CPU, data_Real_GPU, data_Imag_CPU, data_Imag_GPU, *num_err);
                        }
                        amplitude_squared_error = difference_imag*difference_imag+difference_real*difference_real;
                        *err_2 += amplitude_squared_error;
                        if (amplitude_squared_error > max_error)
                            max_error = amplitude_squared_error;
                    }
                    else{
                        if (verbosity){
                            printf ("freq: %6d element_x: %6d element_y: %6d Real CPU/GPU %8d %8d Imaginary CPU/GPU %8d %8d\n",freq, element_x, element_y, data_Real_CPU, data_Real_GPU, data_Imag_CPU, data_Imag_GPU);
                        }
                    }
                }
            }
        }
    }
    printf("\nTotal number of errors: %d, Sum of Squared Differences: %lld \n",*num_err, (long long int) *err_2);
    printf("sqrt(sum of squared differences/numberElements): %f \n", sqrt((*err_2)*1.0/local_Address));//add some more data--find maximum error, figure out other statistical properties
    printf("Maximum amplitude squared error: %d\n", max_error);
    return;
}

void compare_NSquared_correlator_results_data_has_upper_triangle_only ( int *num_err, int64_t *err_2, int total_output_frequencies, int actual_num_frequencies, int actual_num_elements, int num_data_sets,int *data_set_GPU, int *data_set_CPU, double *ratio_GPU_div_CPU, double *phase_difference, int verbosity){
    //this will compare the values of the two arrays and give information about the comparison
    int address = 0;
    int addressGPU = 0;
    int local_Address = 0;
    *num_err = 0;
    *err_2 = 0;
    int max_error = 0;
    int amplitude_squared_error;
    double amplitude_squared_CPU;
    double amplitude_squared_GPU;
    double phase_angle_CPU;
    double phase_angle_GPU;
    for (int m = 0; m < num_data_sets; m++){
        if (actual_num_elements == 16)
            addressGPU = m*total_output_frequencies*actual_num_elements*(actual_num_elements+1); // /2*2
        for (int freq = 0; freq < actual_num_frequencies; freq++){
            for (int element_y = 0; element_y < actual_num_elements; element_y++){
                for (int element_x = element_y; element_x < actual_num_elements; element_x++){
                    //compare real results
                    int data_Real_GPU = data_set_GPU[addressGPU++];
                    int data_Real_CPU = data_set_CPU[address++];
                    int difference_real = data_Real_GPU - data_Real_CPU;
                    //compare imaginary results
                    int data_Imag_GPU = data_set_GPU[addressGPU++];
                    int data_Imag_CPU = data_set_CPU[address++];
                    int difference_imag = data_Imag_GPU - data_Imag_CPU;

                    //get amplitude_squared
                    amplitude_squared_CPU = data_Real_CPU*data_Real_CPU + data_Imag_CPU*data_Imag_CPU;
                    amplitude_squared_GPU = data_Real_GPU*data_Real_GPU + data_Imag_GPU*data_Imag_GPU;
                    phase_angle_CPU = atan2((double)data_Imag_CPU,(double)data_Real_CPU);
                    phase_angle_GPU = atan2((double)data_Imag_GPU,(double)data_Real_GPU);

                    if (amplitude_squared_CPU != 0){
                        ratio_GPU_div_CPU[local_Address] = amplitude_squared_GPU/amplitude_squared_CPU;
                    }
                    else{
                        ratio_GPU_div_CPU[local_Address] = -1;//amplitude_squared_GPU/amplitude_squared_CPU;
                    }

                    phase_difference[local_Address++] = phase_angle_GPU - phase_angle_CPU;

                    if (difference_real != 0 || difference_imag !=0){
                        (*num_err)++;
                        if (verbosity ){
                            printf ("freq: %6d element_x: %6d element_y: %6d Real CPU/GPU %8d %8d Imaginary CPU/GPU %8d %8d ERR: %7d\n",freq, element_x, element_y, data_Real_CPU, data_Real_GPU, data_Imag_CPU, data_Imag_GPU, *num_err);
                        }
                        amplitude_squared_error = difference_imag*difference_imag+difference_real*difference_real;
                        *err_2 += amplitude_squared_error;
                        if (amplitude_squared_error > max_error)
                            max_error = amplitude_squared_error;
                    }
                    else{
                        if (verbosity){
                            printf ("freq: %6d element_x: %6d element_y: %6d Real CPU/GPU %8d %8d Imaginary CPU/GPU %8d %8d\n",freq, element_x, element_y, data_Real_CPU, data_Real_GPU, data_Imag_CPU, data_Imag_GPU);
                        }
                    }
                }
            }
        }
    }
    printf("\nTotal number of errors: %d, Sum of Squared Differences: %lld \n",*num_err, (long long int) *err_2);
    printf("sqrt(sum of squared differences/numberElements): %f \n", sqrt((*err_2)*1.0/local_Address));//add some more data--find maximum error, figure out other statistical properties
    printf("Maximum amplitude squared error: %d\n", max_error);
    return;
}

int main(int argc, char ** argv) {
    double cputime=0;

    if (argc == 1){
        printf("This program expects the user to run the executable as \n $ ./executable GPU_card[0-3] num_repeats\n");
        return -1;
    }

    int dev_number = atoi(argv[1]);
    int nkern= atoi(argv[2]);//NUM_REPEATS_GPU;

    //basic setup of CL devices
    cl_int err;
    //cl_int err2;

    // 1. Get a platform.
    cl_platform_id platform;
    clGetPlatformIDs( 1, &platform, NULL );

    // 2. Find a gpu device.
    cl_device_id deviceID[5];

    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 4, deviceID, NULL);

    if (err != CL_SUCCESS){
        printf("Error getting device IDs\n");
        return (-1);
    }
    cl_ulong lm;
    err = clGetDeviceInfo(deviceID[dev_number], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lm, NULL);
    if (err != CL_SUCCESS){
        printf("Error getting device info\n");
        return (-1);
    }
    //printf("Local Mem: %i\n",lm);

    cl_uint mcl,mcm;
    clGetDeviceInfo(deviceID[dev_number], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &mcl, NULL);
    clGetDeviceInfo(deviceID[dev_number], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &mcm, NULL);
    float card_tflops = mcl*1e6 * mcm*16*4*2 / 1e12;

    // 3. Create a context and command queues on that device.
    cl_context context = clCreateContext( NULL, 1, &deviceID[dev_number], NULL, NULL, NULL);
    cl_command_queue queue[N_QUEUES];
    for (int i = 0; i < N_QUEUES; i++){
        queue[i] = clCreateCommandQueue( context, deviceID[dev_number], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err );
        //add a more robust error check at some point?
        if (err){ //success returns a 0
            printf("Error initializing queues.  Exiting program.\n");
            return (-1);
        }

    }

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    int size1_block = 32;
    int num_blocks = (NUM_ELEM / size1_block) * (NUM_ELEM / size1_block + 1) / 2.; // 256/32 = 8, so 8 * 9/2 (= 36) //needed for the define statement

    // 4a load the source files //this load routine is based off of example code in OpenCL in Action by Matthew Scarpino
    char cl_fileNames[3][256];
    sprintf(cl_fileNames[0],OPENCL_FILENAME_1);

    if (XOR_MANUAL)
        sprintf(cl_fileNames[1],OPENCL_FILENAME_2b);
    else
        sprintf(cl_fileNames[1],OPENCL_FILENAME_2);
    sprintf(cl_fileNames[2],OPENCL_FILENAME_3);

    char cl_options[1024];
    sprintf(cl_options,"-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du", ACTUAL_NUM_ELEM, ACTUAL_NUM_FREQ, NUM_ELEM, NUM_FREQ, num_blocks, NUM_TIMESAMPLES);
    size_t cl_programSize[NUM_CL_FILES];
    FILE *fp;
    char *cl_programBuffer[NUM_CL_FILES];


    for (int i = 0; i < NUM_CL_FILES; i++){
        fp = fopen(cl_fileNames[i], "r");
        if (fp == NULL){
            printf("error loading file: %s\n", cl_fileNames[i]);
            return (-1);
        }
        fseek(fp, 0, SEEK_END);
        cl_programSize[i] = ftell(fp);
        rewind(fp);
        cl_programBuffer[i] = (char*)malloc(cl_programSize[i]+1);
        cl_programBuffer[i][cl_programSize[i]] = '\0';
        int sizeRead = fread(cl_programBuffer[i], sizeof(char), cl_programSize[i], fp);
        if (sizeRead < cl_programSize[i])
            printf("Error reading the file!!!");
        fclose(fp);
    }

    cl_program program = clCreateProgramWithSource( context, NUM_CL_FILES, (const char**)cl_programBuffer, cl_programSize, &err );
    if (err){
        printf("Error in clCreateProgramWithSource: %i\n",err);
        return(-1);
    }

    //printf("here1\n");
    err = clBuildProgram( program, 1, &deviceID[dev_number], cl_options, NULL, NULL );
    if (err){
        printf("Error in clBuildProgram: %i\n",err);
        return(-1);
    }

    cl_kernel corr_kernel = clCreateKernel( program, "corr", &err );
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return (-1);
    }

    cl_kernel offsetAccumulate_kernel = clCreateKernel( program, "offsetAccumulateElements", &err );
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return -1;
    }

    cl_kernel preseed_kernel = clCreateKernel( program, "preseed", &err );
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return -1;
    }

    for (int i =0; i < NUM_CL_FILES; i++){
        free(cl_programBuffer[i]);
    }

    // 5. set up arrays and initilize if required
    unsigned char *host_PrimaryInput    [N_STAGES]; //where things are brought from, ultimately. Code runs fastest when we create the aligned memory and then pin it to the device  	
    //unsigned char *host_CLinput_data    [N_STAGES]; //for the pointer that is created when you map the host to the cl memory  
    int *host_PrimaryOutput             [N_STAGES];
    //int *host_CLoutput_data             [N_STAGES];
    cl_mem device_CLinput_pinnedBuffer  [N_STAGES];
    cl_mem device_CLoutput_pinnedBuffer [N_STAGES];
    cl_mem device_CLinput_kernelData    [N_STAGES];
    cl_mem device_CLoutput_kernelData   [N_STAGES];

    cl_mem device_CLoutputAccum         [N_STAGES];
    //cl_mem device_CLoutputAccum_pinnedBuffer;
    //cl_int *host_outputAccum; //to do: clean up after pointers


    int len=NUM_FREQ*num_blocks*(size1_block*size1_block)*2*NUM_DATA_SETS;// *2 real and imag
    printf("Num_blocks %d ", num_blocks);
    printf("Output Length %d and size %ld B\n", len, len*sizeof(cl_int));
    cl_int *zeros=calloc(len,sizeof(cl_int)); //for the output buffers

    //posix_memalign ((void **)&host_CLinput_data, 4096, NUM_TIMESAMPLES*NUM_ELEM); //online it said that memalign was obsolete, so am using posix_memalign instead.  This should allow for pinning if desired. 
    //they use getpagesize() for the demo's alignment size for Linux for Firepro, rather than the 4096 which was mentioned in the BufferBandwidth demo.
    //memalign taken care of by CL?

    printf("Size of Data block(s) = %i B\n", NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ);
    printf("Number of Data Sets = %i\n", NUM_DATA_SETS);
    printf("Total Data size = %d B\n", NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ*NUM_DATA_SETS);

    // Set up arrays so that they can be used later on
    for (int i = 0; i < N_STAGES; i++){
        //preallocate memory for pinned buffers
        err = posix_memalign ((void **)&host_PrimaryInput[i], PAGESIZE_MEM, NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ*NUM_DATA_SETS); 
        //check if an extra command is needed to pre pin this--this might just make sure it is
        //aligned in memory space.
        if (err){
            printf("error in creating memory buffers: Inputa, stage: %i, err: %i. Exiting program.\n",i, err);
            return (err);
        }
        err = mlock(host_PrimaryInput[i], NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ*NUM_DATA_SETS);
        if (err){
            printf("error in creating memory buffers: Inputb, stage: %i, err: %i. Exiting program.\n",i, err);
            printf("%s",strerror(errno));
            return (err);
        }

        device_CLinput_pinnedBuffer[i] = clCreateBuffer ( context,
                                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,//
                                    NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ*NUM_DATA_SETS,
                                    host_PrimaryInput[i],
                                    &err); //create the clBuffer, using pre-pinned host memory

        if (err){
            printf("error in mapping pin pointers. Exiting program.\n");
            return (err);
        }

        err = posix_memalign ((void **)&host_PrimaryOutput[i], PAGESIZE_MEM, len*sizeof(cl_int));
        err |= mlock(host_PrimaryOutput[i],len*sizeof(cl_int));
        if (err){
            printf("error in creating memory buffers: Output, stage: %i. Exiting program.\n",i);
            return (err);
        }

        device_CLoutput_pinnedBuffer[i] = clCreateBuffer (context,
                                    CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                    len*sizeof(cl_int),
                                    host_PrimaryOutput[i],
                                    &err); //create the output buffer and allow cl to allocate host memory

        if (err){
            printf("error in mapping pin pointers. Exiting program.\n");
            return (err);
        }

        if (XOR_MANUAL){
            device_CLinput_kernelData[i] = clCreateBuffer (context,
                                        CL_MEM_READ_WRITE,// | CL_MEM_USE_PERSISTENT_MEM_AMD, //ran out of memory when I tried to use this
                                        NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ*NUM_DATA_SETS,
                                        0,
                                        &err); //cl memory that can only be read by kernel

            if (err){
                printf("error in allocating memory. Exiting program.\n");
                return (err);
            }
        }
        else{
            device_CLinput_kernelData[i] = clCreateBuffer (context,
                                        CL_MEM_READ_ONLY,// | CL_MEM_USE_PERSISTENT_MEM_AMD, //ran out of memory when I tried to use this
                                        NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ*NUM_DATA_SETS,
                                        0,
                                        &err); //cl memory that can only be read by kernel

            if (err){
                printf("error in allocating memory. Exiting program.\n");
                return (err);
            }
        }

        device_CLoutput_kernelData[i] = clCreateBuffer (context,
                                    CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
                                    len*sizeof(cl_int),
                                    zeros,
                                    &err); //cl memory that can only be written to by kernel--preset to 0s everywhere

        if (err){
            printf("error in allocating memory. Exiting program.\n");
            return (err);
        }

    } //end for
    free(zeros);

    //initialize an array for the accumulator of offsets (borrowed this buffer from an old version of code--check this)
///////
    zeros=calloc(NUM_FREQ*NUM_ELEM*2*NUM_DATA_SETS,sizeof(cl_uint)); // <--this was missed! Was causing most of the problems!
//////
    device_CLoutputAccum[0] = clCreateBuffer(context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          NUM_FREQ*NUM_ELEM*2*NUM_DATA_SETS*sizeof(cl_uint),
                                          zeros,
                                          &err);
    if (err){
            printf("error in allocating memory. Exiting program.\n");
            return (err);
    }

    device_CLoutputAccum[1] = clCreateBuffer(context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          NUM_FREQ*NUM_ELEM*2*NUM_DATA_SETS*sizeof(cl_uint),
                                          zeros,
                                          &err);
    if (err){
            printf("error in allocating memory. Exiting program.\n");
            return (err);
    }

    //arrays have been allocated

    //--------------------------------------------------------------
    //Generate Data Set!

    generate_char_data_set(GEN_TYPE,
                           GEN_DEFAULT_SEED, //random seed
                           GEN_DEFAULT_RE,//default_real,
                           GEN_DEFAULT_IM,//default_imaginary,
                           GEN_INITIAL_RE,//initial_real,
                           GEN_INITIAL_IM,//initial_imaginary,
                           GEN_FREQ,//int single_frequency,
                           NUM_TIMESAMPLES,//int num_timesteps,
                           ACTUAL_NUM_FREQ,//int num_frequencies,
                           ACTUAL_NUM_ELEM,//int num_elements,
                           NUM_DATA_SETS,
                           host_PrimaryInput[0]);

    if (NUM_ELEM <=32){
        if (INTERLEAVED){
            reorder_data_interleave_2_frequencies(NUM_TIMESAMPLES, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS,host_PrimaryInput[0]);
        }
    }
    //print_element_data(1, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, ALL_FREQUENCIES, host_PrimaryInput[0]);
    //reorder_data_phaseB_breakData(NUM_TIMESAMPLES, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, host_PrimaryInput[0]);

    memcpy(host_PrimaryInput[1], host_PrimaryInput[0], NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ*NUM_DATA_SETS);

    //--------------------------------------------------------------


    // 6. Set up Kernel parameters

    //upper triangular address mapping --converting 1d addresses to 2d addresses
    unsigned int global_id_x_map[num_blocks];
    unsigned int global_id_y_map[num_blocks];

//     for (int i=0; i<num_blocks; i++){
//         int t = (int)(sqrt(1 + 8*(num_blocks-i-1))-1)/2; /*t is number of the current row, counting/increasing row numbers from the bottom, up, and starting at 0 --note it uses the property that converting to int uses a floor/truncates at the decimal*/
//         int y = NUM_ELEM/size1_block-t-1;
//         int x = (t+1)*(t+2)/2 + (i - num_blocks)+y;
//         global_id_x_map[i] = x;
//         global_id_y_map[i] = y;
//         printf("i = %d: t = %d, y = %d, x = %d \n", i, t, y, x);
//     }

    //TODO: p260 OpenCL in Action has a clever while loop that changes 1 D addresses to X & Y indices for an upper triangle.  Time Test kernels using 
    //them compared to the lookup tables for NUM_ELEM = 256
    int largest_num_blocks_1D = NUM_ELEM/size1_block;
    int index_1D = 0;
    for (int j = 0; j < largest_num_blocks_1D; j++){
        for (int i = j; i < largest_num_blocks_1D; i++){
            global_id_x_map[index_1D] = i;
            global_id_y_map[index_1D] = j;
            index_1D++;
        }
    }

    cl_mem id_x_map = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    num_blocks * sizeof(cl_uint), global_id_x_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    cl_mem id_y_map = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    //set other parameters that will be fixed for the kernels (changeable parameters will be set in run loops)
    clSetKernelArg(corr_kernel, 2, sizeof(id_x_map), (void*) &id_x_map); //this should maybe be sizeof(void *)?
    clSetKernelArg(corr_kernel, 3, sizeof(id_y_map), (void*) &id_y_map);
    clSetKernelArg(corr_kernel, 4, 8*8*4 * sizeof(cl_uint), NULL);

    clSetKernelArg(preseed_kernel, 2, sizeof(id_x_map), (void*) &id_x_map); //this should maybe be sizeof(void *)?
    clSetKernelArg(preseed_kernel, 3, sizeof(id_y_map), (void*) &id_y_map);
    clSetKernelArg(preseed_kernel, 4, 64* sizeof(cl_uint), NULL);
    clSetKernelArg(preseed_kernel, 5, 64* sizeof(cl_uint), NULL);
    
    //uint data_input_length = ACTUAL_NUM_ELEM*ACTUAL_NUM_FREQ*NUM_TIMESAMPLES/4;
    //clSetKernelArg(offsetAccumulate_kernel, 2, sizeof(data_input_length),&data_input_length);

// Number of repetitions used for timing ------------->

// <---------------------------------------------------
    unsigned int n_cAccum=NUM_TIMESAMPLES/256u; //n_cAccum == number_of_compressedAccum
    size_t gws_corr[3]={8*NUM_DATA_SETS,8*NUM_FREQ,num_blocks*n_cAccum}; //global work size array
    size_t lws_corr[3]={8,8,1}; //local work size array

    size_t gws_accum[3]={64*NUM_DATA_SETS, (int)ceil(NUM_ELEM*NUM_FREQ/256.0),NUM_TIMESAMPLES/1024}; //1024 is the number of iterations performed in the kernel--hardcoded, so if the number in the kernel changes, change here as well
    size_t lws_accum[3]={64, 1, 1};

    size_t gws_preseed[3]={8*NUM_DATA_SETS, 8*NUM_FREQ, num_blocks};
    size_t lws_preseed[3]={8, 8, 1};

    //int *corr_ptr;

    //setup and start loop to process data in parallel
    int spinCount = 0; //we rotate through values to launch processes in order for the command queues. This helps keep track of what position, and can run indefinitely without overflow
    int writeToDevStageIndex;
    int kernelStageIndex;
    //int readFromDevStageIndex;

    cl_int numWaitEventWrite = 0;
    cl_event* eventWaitPtr = NULL;

    cl_event lastWriteEvent[N_STAGES]  = { 0 }; // All entries initialized to 0, since unspecified entries are set to 0
    cl_event lastKernelEvent[N_STAGES] = { 0 };
    //cl_event lastReadEvent[N_STAGES]   = { 0 };
    cl_event copyInputDataEvent;
    cl_event offsetAccumulateEvent;
    cl_event preseedEvent;

    if (TIMER_FOR_PROCESSING_ONLY){
        for (int i = 0; i < N_STAGES; i++){
             err = clEnqueueWriteBuffer(queue[0],
                                    device_CLinput_kernelData[i], //to here
                                    CL_TRUE,
                                    0, //offset
                                    NUM_TIMESAMPLES * NUM_ELEM*NUM_FREQ * NUM_DATA_SETS, //8 for multifreq interleaving
                                    host_PrimaryInput[i], //from here
                                    0,
                                    NULL,
                                    NULL);
            if (err){
                printf("Error in transfer to device memory. Error in loop %d, error: %s\n",i,oclGetOpenCLErrorCodeStr(err));
                exit(err);
            }
        }
        clFinish(queue[0]);
        printf("copy complete\n");
    }

    ///////////////////////////////////////////////////////////////////////////////
    cputime = e_time();
    for (int i=0; i<=nkern; i++){//if we were truly streaming data, for each correlation, we would need to change what arrays are used for input/output
        //printf("spinCount %d\n",spinCount);
        writeToDevStageIndex =  (spinCount ); // + 0) % N_STAGES;
        kernelStageIndex =      (spinCount + 1 ) % N_STAGES; //had been + 2 when it was 3 stages
        //readFromDevStageIndex = (spinCount + 1 ) % N_STAGES;

        //transfer section
        if (i < nkern){ //Start at 0, Stop before the last loop
            //check if it needs to wait on anything
            if(lastKernelEvent[writeToDevStageIndex] != 0){ //only equals 0 when it hasn't yet been defined i.e. the first run through the loop with N_STAGES == 2
                numWaitEventWrite = 1;
                eventWaitPtr = &lastKernelEvent[writeToDevStageIndex]; //writes must wait on the last kernel operation since
            }
            else {
                numWaitEventWrite = 0;
                eventWaitPtr = NULL;
                }

            //copy necessary buffers to device memory
            if (TIMER_FOR_PROCESSING_ONLY){
                err = clEnqueueWriteBuffer(queue[0],
                                        device_CLoutputAccum[writeToDevStageIndex],
                                        CL_FALSE,
                                        0,
                                        NUM_FREQ*NUM_ELEM*2*NUM_DATA_SETS*sizeof(cl_int),
                                        zeros,
                                        numWaitEventWrite,
                                        eventWaitPtr,
                                        &lastWriteEvent[writeToDevStageIndex]);


                err = clFlush(queue[0]);
                if (err){
                    printf("Error in flushing transfer to device memory. Error in loop %d\n",i);
                    exit(err);
                }
//                 if (i%100 == 0)
//                     printf(".");
            }
            else{
                err = clEnqueueWriteBuffer(queue[0],
                                        device_CLinput_kernelData[writeToDevStageIndex], //to here
                                        CL_FALSE,
                                        0, //offset
                                        NUM_TIMESAMPLES * NUM_ELEM*NUM_FREQ*NUM_DATA_SETS, //8 for multifreq interleaving
                                        host_PrimaryInput[writeToDevStageIndex], //from here
                                        numWaitEventWrite,
                                        eventWaitPtr,
                                        &copyInputDataEvent);
                if (err){
                    printf("Error in transfer to device memory. Error in loop %d, error: %s\n",i,oclGetOpenCLErrorCodeStr(err));
                    exit(err);
                }

                err = clEnqueueWriteBuffer(queue[0],
                                        device_CLoutputAccum[writeToDevStageIndex],
                                        CL_FALSE,
                                        0,
                                        NUM_FREQ*NUM_ELEM*2*NUM_DATA_SETS*sizeof(cl_int),
                                        zeros,
                                        1,
                                        &copyInputDataEvent,
                                        &lastWriteEvent[writeToDevStageIndex]);


                err = clFlush(queue[0]);
                if (err){
                    printf("Error in flushing transfer to device memory. Error in loop %d\n",i);
                    exit(err);
                }
            }
        }
        //printf("hello");

        //processing section
        if (lastWriteEvent[kernelStageIndex] !=0 && i <= nkern){//insert additional steps for processing here
            //required steps include: offset accumulator (order(Num_elements*Num_frequencies*Num_timesteps))
            //then pre-seed output array
            //then perform the correlation

            //accumulateFeeds_kernel--set 2 arguments--input array and zeroed output array
            err = clSetKernelArg(offsetAccumulate_kernel,
                                 0,
                                 sizeof(void*),
                                 (void*) &device_CLinput_kernelData[kernelStageIndex]);

            err |= clSetKernelArg(offsetAccumulate_kernel,
                                  1,
                                  sizeof(void *),
                                  (void *) &device_CLoutputAccum[kernelStageIndex]); //make sure this array is zeroed initially!
            if (err){
                printf("Error setting the kernel 0 arguments in loop %d\n", i);
                exit(err);
            }


            err = clEnqueueNDRangeKernel(queue[1],
                                         offsetAccumulate_kernel,
                                         3,
                                         NULL,
                                         gws_accum,
                                         lws_accum,
                                         1,
                                         &lastWriteEvent[kernelStageIndex], /* make sure data is present, first*/
                                         &offsetAccumulateEvent);
            if (err){
                printf("Error accumulating in loop %d\n", i);
                exit(err);
            }

            //preseed_kernel--set only 2 of the 6 arguments (the other 4 stay the same)
            err = clSetKernelArg(preseed_kernel,
                                 0,
                                 sizeof(void *),
                                 (void *) &device_CLoutputAccum[kernelStageIndex]);//assign the accumulated data as input

            err = clSetKernelArg(preseed_kernel,
                                 1,
                                 sizeof(void *),
                                 (void *) &device_CLoutput_kernelData[kernelStageIndex]); //set the output for preseeding the correlator array

            err = clEnqueueNDRangeKernel(queue[1],
                                         preseed_kernel,
                                         3, //3d global dimension, also worksize
                                         NULL, //no offsets
                                         gws_preseed,
                                         lws_preseed,
                                         1,
                                         &offsetAccumulateEvent,/*dependent on previous step so don't use &lastWriteEvent[kernelStageIndex],*/
                                         &preseedEvent);
            if (err){
                printf("Error performing preseed kernel operation in loop %d: error %d\n", i,err);
                exit(err);
            }

            //corr_kernel--set the input and output buffers (the other parameters stay the same).
            err =  clSetKernelArg(corr_kernel,
                                    0,
                                    sizeof(device_CLinput_kernelData[kernelStageIndex]), //sizeof(void *)
                                    (void*) &device_CLinput_kernelData[kernelStageIndex]);
            if (err){
                printf("Error setting the kernel 0 arguments in loop %d\n", i);
                exit(err);
            }
            err = clSetKernelArg(corr_kernel,
                                    1,
                                    sizeof(void *),//sizeof(void *)
                                    (void*) &device_CLoutput_kernelData[kernelStageIndex]);
            if (err){
                printf("Error setting the kernel 1 arguments in loop %d\n", i);
                exit(err);
            }

            err = clEnqueueNDRangeKernel(queue[1],
                                         corr_kernel,
                                         3, //3d global dimension, also worksize
                                         NULL, //no offsets
                                         gws_corr,
                                         lws_corr,
                                         1,
                                         &preseedEvent,/*dependent on previous step so don't use &lastWriteEvent[kernelStageIndex],*/
                                         &lastKernelEvent[kernelStageIndex]);
            if (err){
                printf("Error performing corr kernel operation in loop %d, err: %d\n", i,err);
                exit(err);
            }
            //printf("kernelStageIndex %i\n", kernelStageIndex);
            //if (kernelStageIndex == 1)
            //  err = clFinish(queue[1]);
            //else
                err = clFlush(queue[1]);
            if (err){
                printf("Error in flushing kernel run. Error in loop %d\n",i);
                exit(err);
            }

        }

        //since the kernel accumulates results, it isn't necessary/wanted to have it pull out results each time
        //processing of the results would need to be done, arrays reset (or kernel changed); the transfers could also slow things down, too....
        if (i%10==0){
            err =  clFinish(queue[0]);
            err |= clFinish(queue[1]);

            if (err){
                printf("Error while finishing up the queue after the loops.\n");
                return (err);
            }
        }
        spinCount++;
        spinCount = (spinCount < N_STAGES) ? spinCount : 0; //keeps the value of spinCount small, always, and then saves 1 remainder calculation earlier in the loop. 
    }

    //since there are only 2, simplify things (i.e. no need for a loop).
    err =  clFinish(queue[0]);
    err |= clFinish(queue[1]);

    if (err){
        printf("Error while finishing up the queue after the loops.\n");
        return (err);
    }
 //printf("HELLO\n");
    clReleaseEvent(preseedEvent);
    clReleaseEvent(copyInputDataEvent);
    clReleaseEvent(offsetAccumulateEvent);
    clReleaseEvent(lastKernelEvent[0]);
    clReleaseEvent(lastKernelEvent[1]);
    clReleaseEvent(lastWriteEvent[0]);
    clReleaseEvent(lastWriteEvent[1]);
    // 7. Look at the results via synchronous buffer map.
    //int *corr_ptr;
    //printf(".");
    err = clEnqueueReadBuffer(queue[0], device_CLoutput_kernelData[0], CL_TRUE, 0, len*sizeof(cl_int), host_PrimaryOutput[0], 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue[0], device_CLoutput_kernelData[1], CL_TRUE, 0, len*sizeof(cl_int), host_PrimaryOutput[1], 0, NULL, NULL);

    if (err){
        printf("Error reading data back to host.\n");
        //return (err);
    }

    err = clFinish(queue[0]);

    if (err){
        printf("Error while finishing up the queue after the loops.\n");
        //return (err);
    }

    //printf("to transfer part 1\n");
    //accumulate results into one array
    //unmap output?

    printf("Running %i iterations of full corr (%i time samples (%i Ki time samples), %i elements, %i frequencies, %i data set", nkern, NUM_TIMESAMPLES, NUM_TIMESAMPLES/1024, ACTUAL_NUM_ELEM, ACTUAL_NUM_FREQ, NUM_DATA_SETS);

    if (NUM_DATA_SETS == 1)
        printf(")\n");
    else
        printf("s)\n");

    cputime = e_time()-cputime;
    if (nkern > 1){
        for (int i = 0; i < len; i++){
            //dump out results
            if (DEBUG){
                printf("%d ",host_PrimaryOutput[0][i]);
                if ((i+1) % (32 * 32* 2) == 0)
                    printf("\n");
            }
            //host_CLoutput_data[0][i] += host_CLoutput_data[1][i];
            host_PrimaryOutput[0][i] += host_PrimaryOutput[1][i];
            host_PrimaryOutput[0][i] /=2; //the results in output 0 and 1 should be identical--this is just to check (in a rough way) that they are.
            //if the average of the two arrays is the correct answer, and one expects both of them to have an answer, then the answers of both should be correct
        }
    }
    //printf("to transfer part 2\n");
    //-------------------------------------------------------------- 
    //-------------------------------------------------------------- 
    double unpack_Rate =(1.0*NUM_TIMESAMPLES*ACTUAL_NUM_FREQ*nkern*NUM_DATA_SETS) /cputime;// /cputime/1000.0;
    printf("Unpacking rate: %6.4fs on GPU (%.1f kHz)\n",cputime,unpack_Rate/1000.0);
    printf("    [Theoretical max: @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n", card_tflops,
                                    (1.*card_tflops)*1e12 / (ACTUAL_NUM_ELEM/2.*(ACTUAL_NUM_ELEM+1.) * 2. * 2.) / 1e3,
                                    100.*unpack_Rate / ((card_tflops*1e12) / (ACTUAL_NUM_ELEM/2.*(ACTUAL_NUM_ELEM+1.) * 2. * 2.)));
    if (ACTUAL_NUM_ELEM == 16){
        printf("    [Algorithm max:   @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n", card_tflops,
                                    (1.*card_tflops)*1e12 / (num_blocks * 16 * 16 * 2. * 2.) / 1e3,
                                    100.*unpack_Rate / ((card_tflops*1e12) / (num_blocks * 16 * 16 * 2. * 2.)));
    }
    else{
        printf("    [Algorithm max:   @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n", card_tflops,
                                    (1.*card_tflops)*1e12 / (num_blocks * size1_block * size1_block * 2. * 2.) / 1e3,
                                    100.*unpack_Rate / ((card_tflops*1e12) / (num_blocks * size1_block * size1_block * 2. * 2.)));

    }
    // start using calls to do the comparisons
    cputime = e_time();
    int *correlated_CPU = calloc((ACTUAL_NUM_ELEM*(ACTUAL_NUM_ELEM))*ACTUAL_NUM_FREQ*2*NUM_DATA_SETS,sizeof(int)); //made for the largest possible size (one size fits all)
    //int *correlated_CPU = calloc((ACTUAL_NUM_ELEM*(ACTUAL_NUM_ELEM+1))/2*ACTUAL_NUM_FREQ*2,sizeof(int));
    if (correlated_CPU == NULL){
        printf("failed to allocate memory\n");
        return(-1);
    }
//    err = cpu_data_generate_and_correlate(NUM_TIMESAMPLES, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, correlated_CPU);
    if (UPPER_TRIANGLE){
        err = cpu_data_generate_and_correlate_upper_triangle_only(NUM_TIMESAMPLES, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS, correlated_CPU);
    }
    else{
        err = cpu_data_generate_and_correlate(NUM_TIMESAMPLES, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS, correlated_CPU);
    }

    int *correlated_GPU = (int *)malloc((ACTUAL_NUM_ELEM*(ACTUAL_NUM_ELEM))*HDF5_FREQ*2*NUM_DATA_SETS*sizeof(int));
    //int *correlated_GPU = (int *)malloc((ACTUAL_NUM_ELEM*(ACTUAL_NUM_ELEM+1))/2*ACTUAL_NUM_FREQ*2*sizeof(int));
    if (correlated_GPU == NULL){
        printf("failed to allocate memory\n");
        return(-1);
    }

    if (ACTUAL_NUM_ELEM == 16){
        //printf("Hey!\n");
        if (INTERLEAVED){
            //printf("I was interleaved!\n");
            reorganize_32_to_16_feed_GPU_Correlated_Data_Interleaved(ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS, host_PrimaryOutput[0]);
        }
        else{
            reorganize_32_to_16_feed_GPU_Correlated_Data(ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS, host_PrimaryOutput[0]); //needed for comparison of outputs.
        }

        if (UPPER_TRIANGLE){
            reorganize_data_16_element_with_triangle_conversion(HDF5_FREQ, ACTUAL_NUM_FREQ,NUM_DATA_SETS,host_PrimaryOutput[0],correlated_GPU);
        }
    }
    else{
        if (UPPER_TRIANGLE){
            reorganize_GPU_to_upper_triangle(size1_block, num_blocks, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS, host_PrimaryOutput[0], correlated_GPU);
        }
        else{
            reorganize_GPU_to_full_Matrix_for_comparison(size1_block, num_blocks, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS, host_PrimaryOutput[0], correlated_GPU);
        }
    }
    //correct_GPU_correlation_results (NUM_TIMESAMPLES, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, host_PrimaryOutput[0], host_outputAccum); //host side correction not needed now.
    int number_errors = 0;
    int64_t errors_squared;
    double *amp2_ratio_GPU_div_CPU = (double *)malloc(ACTUAL_NUM_ELEM*ACTUAL_NUM_ELEM*ACTUAL_NUM_FREQ*NUM_DATA_SETS*sizeof(double));
    if (amp2_ratio_GPU_div_CPU == NULL){
        printf("ran out of memory\n");
        return (-1);
    }
    double *phaseAngleDiff_GPU_m_CPU = (double *)malloc(ACTUAL_NUM_ELEM*ACTUAL_NUM_ELEM*ACTUAL_NUM_FREQ*NUM_DATA_SETS*sizeof(double));
    if (phaseAngleDiff_GPU_m_CPU == NULL){
        printf("2ran out of memory\n");
        return (-1);
    }

    if (UPPER_TRIANGLE){
        compare_NSquared_correlator_results_data_has_upper_triangle_only ( &number_errors, &errors_squared, HDF5_FREQ, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS, correlated_GPU, correlated_CPU, amp2_ratio_GPU_div_CPU, phaseAngleDiff_GPU_m_CPU, CHECKING_VERBOSE);
    }
    else{
        compare_NSquared_correlator_results ( &number_errors, &errors_squared, ACTUAL_NUM_FREQ, ACTUAL_NUM_ELEM, NUM_DATA_SETS, correlated_GPU, correlated_CPU, amp2_ratio_GPU_div_CPU, phaseAngleDiff_GPU_m_CPU, CHECKING_VERBOSE);
    }

    if (number_errors > 0)
        printf("Error with correlation/accumulation! Num Err: %d and length of correlated data: %d\n",number_errors, ACTUAL_NUM_ELEM*ACTUAL_NUM_ELEM*ACTUAL_NUM_FREQ);
    else
        printf("Correlation/accumulation successful! CPU matches GPU.\n");
    //printf ("idx = %d\n", idx);
    cputime=e_time()-cputime;
    printf("Full Corr (1 iteration): %4.2fs on CPU (%.2f kHz)\n",cputime,(1.0*NUM_TIMESAMPLES*ACTUAL_NUM_FREQ*NUM_DATA_SETS) /cputime/1e3);

    err = munlockall();


    for (int ns=0; ns < N_STAGES; ns++){
        err = clReleaseMemObject(device_CLinput_pinnedBuffer[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
        err = clReleaseMemObject( device_CLoutput_pinnedBuffer[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
        err = clReleaseMemObject(device_CLinput_kernelData[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
        err = clReleaseMemObject(device_CLoutput_kernelData[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
        assert(host_PrimaryInput[ns]!=NULL);
        free(host_PrimaryInput[ns]);
        free(host_PrimaryOutput[ns]);

        //err = clReleaseMemObject(device_CLoutputAccum_pinnedBuffer[ns]);
        //if (err != SDK_SUCCESS) {
        //    printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
        //    printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
        //    exit(err);
        //}
        err = clReleaseMemObject(device_CLoutputAccum[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
    }


    //free(corr_re);
    //free(corr_im);
    free(correlated_CPU);
    free(correlated_GPU);
    //free(host_outputAccum);
    free(amp2_ratio_GPU_div_CPU);
    free(phaseAngleDiff_GPU_m_CPU);
    //free(accum_re);
    //free(accum_im);

    free(zeros);

    //-------------------------------------------------------------- 
    //-------------------------------------------------------------- 


    //free(data_block);
    //free(input_data_2);
    //free(output_data_1);
    //free(output_data_2);

    //err = clEnqueueUnmapMemObject(queue,corr_buffer,corr_ptr,0,NULL,NULL);
    //if (err) printf("Error in clEnqueueUnmapMemObject!\n");
    //clFinish(queue);

    clReleaseKernel(corr_kernel);
    clReleaseProgram(program);
    //clReleaseMemObject(input_buffer);
    //clReleaseMemObject(input_buffer2);
    //clReleaseMemObject(copy_buffer);
    //clReleaseMemObject(copy_buffer2);
    //clReleaseMemObject(corr_buffer);
    clReleaseMemObject(id_x_map);
    clReleaseMemObject(id_y_map);
    clReleaseCommandQueue(queue[0]);
    clReleaseCommandQueue(queue[1]);
    clReleaseContext(context);
    return 0;
}
