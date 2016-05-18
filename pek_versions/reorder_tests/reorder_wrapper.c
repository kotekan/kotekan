// This testing code for the AMD correlator kernel is being released under the MIT License

// For this to work, you will need an AMD GPU and the AMD APP SDK for OpenCL installed: Note this code was written
// with OpenCL 1.2 in mind, not 2.0

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


#define NUM_CL_FILES                    1
#define OPENCL_FILENAME_1               "reorder_with_timeshift.cl"
//#define OPENCL_FILENAME_2               "offset_accumulator.cl"
//#define OPENCL_FILENAME_3               "preseed_multifreq.cl"

#define HI_NIBBLE(b)                    (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b)                    ((b) & 0x0F)

#define N_STAGES                        2 //write to CL_Mem, Kernel (Read is done after many runs since answers are accumulated)
#define N_QUEUES                        2 //have 2 separate queues so transfer and process paths can be queued nicely
#define PAGESIZE_MEM                    4096u
#define TIME_ACCUM                      256u


//enumerations/definitions: don't change
#define GENERATE_DATASET_CONSTANT       1u
#define GENERATE_DATASET_RAMP_UP        2u
#define GENERATE_DATASET_RAMP_DOWN      3u
#define GENERATE_DATASET_RANDOM_SEEDED  4u
#define ALL_FREQUENCIES                -1
#define SDK_SUCCESS                     0u

// ////////////////////// Run Parameters: change these as desired /////////////////////////////////////////
#define NUM_ELEM                        2048u//256u //minimum needs to be 32 for the kernels even if you want 16 elements, in general a multiple of 32
#define NUM_FREQ                        1u
#define ACTUAL_NUM_ELEM                 2048u//for ACTUAL_NUM_ELEM  16, have NUM_ELEM 32 above, and put half NUM_FREQ for ACTUAL_NUM_FREQ
#define ACTUAL_NUM_FREQ                 1u//8u
#define HDF5_FREQ                       1
#define UPPER_TRIANGLE                  1
#define TIMER_FOR_PROCESSING_ONLY       1

#define NUM_TIMESAMPLES                 64u*1024u//1024u //32
#define NUM_REPEATS_GPU                 100000u

#define GEN_TYPE                        GENERATE_DATASET_RANDOM_SEEDED
#define GEN_DEFAULT_SEED                42u
#define GEN_DEFAULT_RE                  0u
#define GEN_DEFAULT_IM                  0u
#define GEN_INITIAL_RE                  0u //-8
#define GEN_INITIAL_IM                  0u //7
#define GEN_FREQ                        ALL_FREQUENCIES
#define GEN_REPEAT_RANDOM               1u

#define CHECK_RESULTS_CPU               1
#define CHECKING_VERBOSE                0 //if 1, this writes out all the correlation products as calculated by the GPU and CPU
// ////////////////////////////////////////////////////////////////////////////////////


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
                            unsigned char *packed_data_set){

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

    if (generation_Type == GENERATE_DATASET_RANDOM_SEEDED){
        srand(random_seed);
    }

    for (int k = 0; k < num_timesteps; k++){
        //printf("k: %d\n",k);
        if (generation_Type == GENERATE_DATASET_RANDOM_SEEDED && GEN_REPEAT_RANDOM){
            srand(random_seed);
        }
        for (int j = 0; j < num_frequencies; j++){
            //printf("j: %d\n",j);
            for (int i = 0; i < num_elements; i++){
                int currentAddress = k*num_frequencies*num_elements + j*num_elements + i;
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
                        new_real = (unsigned char)(rand()%16); //to put the pseudorandom value in the range 0-15
                        new_imaginary = (unsigned char)(rand()%16);
                        break;
                    default: //shouldn't happen, but in case it does, just assign the default values everywhere
                        new_real = clipped_offset_default_real;
                        new_imaginary = clipped_offset_default_imaginary;
                        break;
                }

                if (single_frequency == ALL_FREQUENCIES){
                    temp_output = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                    packed_data_set[currentAddress] = temp_output;
                }
                else{
                    if (j == single_frequency){
                        temp_output = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                    }
                    else{
                        temp_output = ((clipped_offset_default_real<<4) & 0xF0) + (clipped_offset_default_imaginary & 0x0F);
                    }
                    packed_data_set[currentAddress] = temp_output;
                }
                //printf("%d ",data_set[currentAddress]);
            }
        }
    }

    return;
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
                    printf("%4d %4d ",(int)(HI_NIBBLE(temp))-8,(int)(LO_NIBBLE(temp))-8);
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}

int cpu_data_generate_and_correlate(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data){
    //correlatedData will be returned as num_frequencies blocks, each num_elements x num_elements x 2

    //generate a dataset that should be the same as what the gpu is testing
    //dataset will be num_timesteps x num_frequencies x num_elements large
    unsigned char *generated = (unsigned char *)malloc(num_timesteps*num_frequencies*num_elements*sizeof(unsigned char));
    //check the array was allocated properly
    if (generated == NULL){
        printf ("Error allocating memory: cpu_data_generate_and_correlate\n");
        return (-1);
    }

    generate_char_data_set(GEN_TYPE,GEN_DEFAULT_SEED,GEN_DEFAULT_RE, GEN_DEFAULT_IM,GEN_INITIAL_RE,GEN_INITIAL_IM,GEN_FREQ, num_timesteps, num_frequencies, num_elements, generated);

    if (CHECKING_VERBOSE){
        print_element_data(1, num_frequencies, num_elements, ALL_FREQUENCIES, generated);
    }

    unsigned char temp_char;
    //correlate based on generated data
    for (int k = 0; k < num_timesteps; k++){
        for (int j = 0; j < num_frequencies; j++){
            for (int element_y = 0; element_y < num_elements; element_y++){
                temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_y];
                int element_y_re = (int)(HI_NIBBLE(temp_char)) - 8; //-8 is to put the number back in the range -8 to 7 from 0 to 15
                int element_y_im = (int)(LO_NIBBLE(temp_char)) - 8;
                for (int element_x = 0; element_x < num_elements; element_x++){
                    temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_x];
                    int element_x_re = (int)(HI_NIBBLE(temp_char)) - 8;
                    int element_x_im = (int)(LO_NIBBLE(temp_char)) - 8;
                    if (k != 0){
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2]   += element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2+1] += element_x_im*element_y_re - element_x_re*element_y_im;
                    }
                    else{
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2]   = element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2+1] = element_x_im*element_y_re - element_x_re*element_y_im;
                    }
                }
            }
        }
    }

    //clean up parameters as needed
    free(generated);
    return (0);
}

int cpu_data_generate_and_correlate_upper_triangle_only(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data_triangle){
    //correlatedData will be returned as num_frequencies blocks, each num_elements x num_elements x 2

    //generate a dataset that should be the same as what the gpu is testing
    //dataset will be num_timesteps x num_frequencies x num_elements large
    unsigned char *generated = (unsigned char *)malloc(num_timesteps*num_frequencies*num_elements*sizeof(unsigned char));
    //check the array was allocated properly
    if (generated == NULL){
        printf ("Error allocating memory: cpu_data_generate_and_correlate\n");
        return (-1);
    }

    generate_char_data_set(GEN_TYPE,GEN_DEFAULT_SEED,GEN_DEFAULT_RE, GEN_DEFAULT_IM,GEN_INITIAL_RE,GEN_INITIAL_IM,GEN_FREQ, num_timesteps, num_frequencies, num_elements, generated);

    if (CHECKING_VERBOSE){
        print_element_data(1, num_frequencies, num_elements, ALL_FREQUENCIES, generated);
    }

    unsigned char temp_char;
    //correlate based on generated data
    for (int k = 0; k < num_timesteps; k++){
        int output_counter = 0;
        for (int j = 0; j < num_frequencies; j++){
            for (int element_y = 0; element_y < num_elements; element_y++){
                temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_y];
                int element_y_re = (int)(HI_NIBBLE(temp_char)) - 8; //-8 is to put the number back in the range -8 to 7 from 0 to 15
                int element_y_im = (int)(LO_NIBBLE(temp_char)) - 8;
                for (int element_x = element_y; element_x < num_elements; element_x++){
                    temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_x];
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

    //clean up parameters as needed
    free(generated);
    return (0);
}

void reorganize_32_to_16_feed_GPU_Correlated_Data(int actual_num_frequencies, int actual_num_elements, int *correlated_data){
    //data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements exist.
    //This is equivalent to processing 2 elements at the same time, where the desired correlations live in the first and fourth quadrants
    //This function is to reorganize the data so that comparisons can be done more easily

    //The input dataset is larger than the output, so can reorganize in the same array

    int input_frequencies = actual_num_frequencies/2;
    int input_elements = actual_num_elements*2;
    int address = 0;
    int address_out = 0;
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
    return;
}

void reorganize_GPU_to_full_Matrix_for_comparison(int block_side_length, int num_blocks, int actual_num_frequencies, int actual_num_elements, int *gpu_data, int *final_matrix){
    //takes the output data, grouped in blocks of block_dim x block_dim x 2 (complex pairs (ReIm)of ints), and fills a num_elements x num_elements x 2
    //
    for (int frequency_bin = 0; frequency_bin < actual_num_frequencies; frequency_bin++ ){
        int block_x_ID = 0;
        int block_y_ID = 0;
        int num_blocks_x = actual_num_elements/block_side_length;
        int block_check = num_blocks_x;

        for (int block_ID = 0; block_ID < num_blocks; block_ID++){
            if (block_ID == block_check){
                num_blocks_x--;
                block_check += num_blocks_x;
                block_y_ID++;
                block_x_ID = block_y_ID;
            }
            for (int y_ID_local = 0; y_ID_local < block_side_length; y_ID_local++){
                int y_ID_global = block_y_ID * block_side_length + y_ID_local;
                for (int x_ID_local = 0; x_ID_local < block_side_length; x_ID_local++){
                    int GPU_address = frequency_bin*(num_blocks*block_side_length*block_side_length*2) + block_ID *(block_side_length*block_side_length*2) + y_ID_local*block_side_length*2+x_ID_local*2; ///TO DO :simplify this statement after getting everything working
                    int x_ID_global = block_x_ID * block_side_length + x_ID_local;
                    if (x_ID_global >= y_ID_global){
                        if (x_ID_global > y_ID_global){ //store the conjugate: x and y addresses get swapped and the imaginary value is the negative of the original value
                            final_matrix[(frequency_bin*actual_num_elements*actual_num_elements+x_ID_global*actual_num_elements+y_ID_global)*2]   =  gpu_data[GPU_address];
                            final_matrix[(frequency_bin*actual_num_elements*actual_num_elements+x_ID_global*actual_num_elements+y_ID_global)*2+1] = -gpu_data[GPU_address+1];
                        }
                        //store the
                        final_matrix[(frequency_bin*actual_num_elements*actual_num_elements+y_ID_global*actual_num_elements+x_ID_global)*2]   = gpu_data[GPU_address];
                        final_matrix[(frequency_bin*actual_num_elements*actual_num_elements+y_ID_global*actual_num_elements+x_ID_global)*2+1] = gpu_data[GPU_address+1];
                    }
                }
            }
            //printf("block_ID: %d, block_y_ID: %d, block_x_ID: %d\n", block_ID, block_y_ID, block_x_ID);
            //update block offset values
            block_x_ID++;
        }
    }
    return;
}

void reorganize_GPU_to_upper_triangle(int block_side_length, int num_blocks, int actual_num_frequencies, int actual_num_elements, int *gpu_data, int *final_matrix){
    int GPU_address = 0; //we go through the gpu data sequentially and map it to the proper locations in the output array
    for (int frequency_bin = 0; frequency_bin < actual_num_frequencies; frequency_bin++ ){
        int block_x_ID = 0;
        int block_y_ID = 0;
        int num_blocks_x = actual_num_elements/block_side_length;
        int block_check = num_blocks_x;
        int frequency_offset = frequency_bin * (actual_num_elements* (actual_num_elements+1))/2;// frequency_bin * number of items in an upper triangle

        for (int block_ID = 0; block_ID < num_blocks; block_ID++){
            if (block_ID == block_check){
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
    return;
}

void reorganize_data_16_element_with_triangle_conversion (int num_frequencies_final, int actual_num_frequencies, int *input_data, int *output_data){
    //input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of complex visibilities for frequencies
    //output array will be sparsely to moderately filled, so loop such that writing is done in sequential order
    //int num_complex_visibilities = 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_counter = 0;
    for (int freq_count = 0; freq_count < num_frequencies_final; freq_count++){
        for (int y = 0; y < 16; y++){
            for (int x = y; x < 16; x++){
                if (freq_count < actual_num_frequencies){
                    int input_index = (freq_count * 256 + y*16 + x)*2; //blocks of data are 16 x 16 = 256 and row_stride is 16
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
    return;
}


void compare_NSquared_correlator_results ( int *num_err, int64_t *err_2, int num_frequencies, int num_elements, int *data_set_GPU, int *data_set_CPU, double *ratio_GPU_div_CPU, double *phase_difference, int verbosity){
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
    printf("\nTotal number of errors: %d, Sum of Squared Differences: %lld \n",*num_err, (long long int) *err_2);
    printf("sqrt(sum of squared differences/numberElements): %f \n", sqrt((*err_2)*1.0/local_Address));//add some more data--find maximum error, figure out other statistical properties
    printf("Maximum amplitude squared error: %d\n", max_error);
    return;
}

void compare_NSquared_correlator_results_data_has_upper_triangle_only ( int *num_err, int64_t *err_2, int actual_num_frequencies, int actual_num_elements, int *data_set_GPU, int *data_set_CPU, double *ratio_GPU_div_CPU, double *phase_difference, int verbosity){
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
    for (int freq = 0; freq < actual_num_frequencies; freq++){
        for (int element_y = 0; element_y < actual_num_elements; element_y++){
            for (int element_x = element_y; element_x < actual_num_elements; element_x++){
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
    printf("\nTotal number of errors: %d, Sum of Squared Differences: %lld \n",*num_err, (long long int) *err_2);
    printf("sqrt(sum of squared differences/numberElements): %f \n", sqrt((*err_2)*1.0/local_Address));//add some more data--find maximum error, figure out other statistical properties
    printf("Maximum amplitude squared error: %d\n", max_error);
    return;
}

int remap_pathfinder_feed_natural_to_slot_based( int feed_number){
    //256 element based... does not extend well, really...
    int offset =feed_number/256;
    feed_number = feed_number%256;
    int east_bool = 0;
    if (feed_number >127)
        east_bool = 1;
    unsigned int south_half = (feed_number/32)&0x1;
    int pol_1_bool = 1-((feed_number/64)&0x1);
    int extra_north_offset = 1-((feed_number/16)&0x1);
    int exclusive_south_or_east = south_half^east_bool;
    unsigned int reordered_subset = (feed_number & 0xF)^0xC;

    int output = (south_half*8+exclusive_south_or_east*4+extra_north_offset*2+pol_1_bool) *16+reordered_subset + offset*256;
    return output;
}

int remap_pathfinder_feed_slot_based_to_natural( int feed_number){
    //256 element based mapping... other than that is a hack to give a mapping
    int offset = feed_number/256;
    feed_number = feed_number%256;
    int east_bool = 0;
    if (feed_number >=64 && feed_number <192)
        east_bool = 1;
    int feed_div_16 = feed_number/16;
    int pol_0_bool = 1 - (feed_div_16 & 0x1);
    int south_half = 1;
    if (feed_number < 128)
        south_half = 0;
    int extra_south_offset = 1-((feed_number/32)&0x1);
    unsigned int reordered_subset = (feed_number & 0xF)^0xC;

    int output = east_bool*128 + pol_0_bool*64 + south_half*32 + extra_south_offset*16 + reordered_subset + offset*256;
    return output;
}

int main(int argc, char ** argv) {
    double cputime=0;

    if (argc != 3){
        printf("This program expects the user to run the executable as \n $ ./<executable_name> GPU_card[0-3] num_repeats\n");
        printf("For example: ./correlator_test 0 100\n");
        return -1;
    }
    int dev_number = atoi(argv[1]);
    int nkern= atoi(argv[2]);//NUM_REPEATS_GPU;

    //basic setup of CL devices
    cl_int err;

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

    cl_uint mcl,mcm;
    clGetDeviceInfo(deviceID[dev_number], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &mcl, NULL);
    clGetDeviceInfo(deviceID[dev_number], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &mcm, NULL);
    //float card_tflops = mcl*1e6 * mcm*16*4*2 / 1e12;

    // 3. Create a context and command queues on that device.
    cl_context context = clCreateContext( NULL, 1, &deviceID[dev_number], NULL, NULL, NULL);
    cl_command_queue queue[N_QUEUES];
    for (int i = 0; i < N_QUEUES; i++){
        queue[i] = clCreateCommandQueue( context, deviceID[dev_number], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err );
        //queue[i] = clCreateCommandQueue( context, deviceID[dev_number], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE , &err );
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
//    sprintf(cl_fileNames[1],OPENCL_FILENAME_2);
//    sprintf(cl_fileNames[2],OPENCL_FILENAME_3);

    char cl_options[1024];
    sprintf(cl_options,"-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du", ACTUAL_NUM_ELEM, ACTUAL_NUM_FREQ, NUM_ELEM, NUM_FREQ, num_blocks, NUM_TIMESAMPLES);
    printf("Dynamic define statements for GPU OpenCL kernels\n");
    printf("-D ACTUAL_NUM_ELEMENTS=%du \n-D ACTUAL_NUM_FREQUENCIES=%du \n-D NUM_ELEMENTS=%du \n-D NUM_FREQUENCIES=%du \n-D NUM_BLOCKS=%du \n-D NUM_TIMESAMPLES=%du\n", ACTUAL_NUM_ELEM, ACTUAL_NUM_FREQ, NUM_ELEM, NUM_FREQ, num_blocks, NUM_TIMESAMPLES);

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

    err = clBuildProgram( program, 1, &deviceID[dev_number], cl_options, NULL, NULL );
    if (err){
        printf("Error in clBuildProgram: %i\n",err);
        size_t log_size;
        clGetProgramBuildInfo(program,deviceID[dev_number], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log;
        program_log = (char*)malloc(log_size+1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program,deviceID[dev_number], CL_PROGRAM_BUILD_LOG, log_size+1,program_log,NULL);
        printf("%s\n",program_log);
        free(program_log);
        return(-1);
    }

    cl_kernel reorder_kernel = clCreateKernel(program, "reorder_input_data_with_timeshift_cache_lookups", &err);
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return (-1);
    }

    for (int i =0; i < NUM_CL_FILES; i++){
        free(cl_programBuffer[i]);
    }

    // 5. set up arrays and initilize if required
    unsigned char *host_PrimaryInput    [N_STAGES]; //where things are brought from, ultimately. Code runs fastest when we create the aligned memory and then pin it to the device

    cl_mem device_CLinput_pinnedBuffer  [N_STAGES];

    cl_mem device_CLinput_kernelData    [N_STAGES];


    int len=NUM_FREQ*num_blocks*(size1_block*size1_block)*2.;// *2 real and imag
//    printf("Num_blocks %d ", num_blocks);
//    printf("Output Length %d and size %ld B\n", len, len*sizeof(cl_int));
    cl_int *zeros=calloc(num_blocks*NUM_FREQ,sizeof(cl_int)); //for the output buffers

    free(zeros);

    zeros=calloc(len,sizeof(cl_int)); //for the output buffers

    printf("Size of Data block = %i B\n", NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ);
    if (TIMER_FOR_PROCESSING_ONLY){
        printf("Setting up and transferring data to GPU...\n");
    }

    // Set up arrays so that they can be used later on
    for (int i = 0; i < N_STAGES; i++){
        //preallocate memory for pinned buffers
        err = posix_memalign ((void **)&host_PrimaryInput[i], PAGESIZE_MEM, NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ);
        //check if an extra command is needed to pre pin this--this might just make sure it is
        //aligned in memory space.
        if (err){
            printf("error in creating memory buffers: Inputa, stage: %i, err: %i. Exiting program.\n",i, err);
            return (err);
        }
        err = mlock(host_PrimaryInput[i], NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ);
        if (err){
            printf("error in creating memory buffers: Inputb, stage: %i, err: %i. Exiting program.\n",i, err);
            printf("%s",strerror(errno));
            return (err);
        }

        device_CLinput_pinnedBuffer[i] = clCreateBuffer ( context,
                                    CL_MEM_USE_HOST_PTR,//
                                    NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ,
                                    host_PrimaryInput[i],
                                    &err); //create the clBuffer, using pre-pinned host memory

        if (err){
            printf("error in mapping pin pointers. Exiting program.\n");
            return (err);
        }



        device_CLinput_kernelData[i] = clCreateBuffer (context,
                                    CL_MEM_READ_WRITE,
                                    NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ,
                                    0,
                                    &err); //cl memory that can only be read by kernel

        if (err){
            printf("error in allocating memory. Exiting program.\n");
            return (err);
        }



    } //end for
    free(zeros);



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
                           host_PrimaryInput[0]);

//    memcpy(host_PrimaryInput[1], host_PrimaryInput[0], NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ);

    //--------------------------------------------------------------


    // 6. Set up Kernel parameters


    size_t gws_reorder[3]={1*(NUM_ELEM/4), NUM_FREQ, NUM_TIMESAMPLES/4};
    size_t lws_reorder[3]={64, 1, 1};


    zeros=calloc(NUM_TIMESAMPLES*NUM_FREQ*NUM_ELEM/4,sizeof(cl_int));
    cl_mem device_reordered_data;
    device_reordered_data = clCreateBuffer (context, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR, NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ,zeros,&err);
    if (err){
        printf("error in allocating memory. Exiting program.\n");
        return (err);
    }
    free(zeros);
    cl_mem device_remap_table;
    cl_mem device_timeshift_map;

    //create map for antennas
    int host_remap_lookup[NUM_ELEM];
    //simple mapping to check...
    //int offset_remap = 1;
    for (int i = 0; i < NUM_ELEM; i++){
        host_remap_lookup[i] = remap_pathfinder_feed_slot_based_to_natural(i);
//        host_remap_lookup[i] = i;
//         if (i+offset_remap < NUM_ELEM)
//             host_remap_lookup[i] = i+offset_remap;//;
//         else
//             host_remap_lookup[i] = i+offset_remap-NUM_ELEM;
    }

    //create timeshift mapping
    int start = 4;
    int stop = 10;
    char host_timeshift_map[NUM_ELEM];
    for (int i = 0; i < NUM_ELEM; i++){
        host_timeshift_map[i] = 0;
//        host_timeshift_map[435] = -2;
//        host_timeshift_map[871] = -2;
        if (i >=start && i <= stop)
            host_timeshift_map[i] = -2;
    }

//     __kernel void reorder_input_data_with_timeshift(__global unsigned int  *input_data,
//                                                 __global unsigned char *output_data,
//                                                 __global const    int  *remap_lookup, //a mapping of elements from FPGA ordering to natural cylinder ordering (or other)
//                                                 __global const    char *timeshift_map //NOTE: the map for timeshifting is based on the remapped addressing
//                                                 ) {

    device_remap_table = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    NUM_ELEM * sizeof(cl_uint), host_remap_lookup, &err);
    err |= clSetKernelArg(reorder_kernel,
                            2,
                            sizeof(void*),
                            (void*) &device_remap_table);
    if (err){
        printf("Error setting the kernel arrays (1)\n");
        exit(err);
    }

    device_timeshift_map = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    NUM_ELEM, host_timeshift_map, &err);
    err |= clSetKernelArg(reorder_kernel,
                            3,
                            sizeof(void *),
                            (void *) &device_timeshift_map); //make sure this array is zeroed initially!
    if (err){
        printf("Error setting the kernel arrays (2)\n");
        exit(err);
    }
    printf("Setup complete\n");
    if (TIMER_FOR_PROCESSING_ONLY){

            err = clEnqueueWriteBuffer(queue[0],
                                device_CLinput_kernelData[0], //to here
                                CL_TRUE,
                                0, //offset
                                NUM_TIMESAMPLES * NUM_ELEM*NUM_FREQ, //8 for multifreq interleaving
                                host_PrimaryInput[0], //from here
                                0,
                                NULL,
                                NULL);
        if (err){
            printf("Error in transfer to device memory. Error: %s\n",oclGetOpenCLErrorCodeStr(err));
            exit(err);
        }

        clFinish(queue[0]);
        printf("Data transfer to GPU complete\n");
    }
    cputime = e_time();

    for (int i=0; i <nkern; i++){
        //printf("Here\n");
        err = clSetKernelArg(reorder_kernel,
                                0,
                                sizeof(void*),
                                (void*) &device_CLinput_kernelData[0]);
        if (err){
            printf("Error setting the kernel 0 arguments in loop %d\n", i);
            exit(err);
        }
//         else{
//             printf("?\n");
//         }
        err = clSetKernelArg(reorder_kernel,
                                1,
                                sizeof(cl_mem),
                                &device_reordered_data); //make sure this array is zeroed initially!
        if (err){
            printf("Error setting the kernel 0 arguments in loop %d\n", i);
            exit(err);
        }
        //printf("Still here\n");
        clEnqueueNDRangeKernel(queue[1],
                                reorder_kernel,
                                3, //3d global dimension, also worksize
                                NULL, //no offsets
                                gws_reorder,
                                lws_reorder,
                                0,
                                NULL,/*dependent on previous step so don't use &lastWriteEvent[kernelStageIndex],*/
                                NULL);
        //printf("i:%d\n",i);
    }

    err = clFinish(queue[1]);
    cputime = e_time()-cputime;

    // 7. Look at the results
    err = clEnqueueReadBuffer(queue[0], device_reordered_data, CL_TRUE, 0, NUM_ELEM*NUM_FREQ*NUM_TIMESAMPLES, host_PrimaryInput[1], 0, NULL, NULL);
//    err |= clEnqueueReadBuffer(queue[0], device_CLoutput_kernelData[1], CL_TRUE, 0, len*sizeof(cl_int), host_PrimaryOutput[1], 0, NULL, NULL);

    if (err){
        printf("Error reading data back to host.\n");
        //return (err);
    }

    err = clFinish(queue[0]);

    if (err){
        printf("Error while finishing up the queue after the loops.\n");
        //return (err);
    }

    printf("Elapsed time: %8.2fs on GPU\n", cputime);
    printf("%d B per timestep, %d timesteps per run, %d runs\n", NUM_FREQ*NUM_ELEM, NUM_TIMESAMPLES, nkern);
    printf("%8.0f B/s processed\n", nkern*NUM_FREQ/cputime*NUM_ELEM*NUM_TIMESAMPLES);
    printf("%d GB/s\n", (int)round(nkern*NUM_FREQ/cputime*NUM_ELEM*NUM_TIMESAMPLES/1024/1024/1024));


    if (CHECK_RESULTS_CPU){
        printf("Checking results. Please wait...\n");
        // start using calls to do the comparisons
        unsigned char * cpu_data = (unsigned char *)malloc(NUM_FREQ*NUM_ELEM*NUM_TIMESAMPLES);
        for (int i = 0; i < NUM_TIMESAMPLES*NUM_ELEM*NUM_FREQ; i++)
            cpu_data[i] =0;
        for (int k = 0; k < NUM_TIMESAMPLES; k++){
            for (int j = 0; j < NUM_FREQ; j++){
                for (int i = 0; i < NUM_ELEM; i++){
                    int element_remapped = host_remap_lookup[i];
                    int cpu_address = (k+host_timeshift_map[element_remapped])*NUM_ELEM*NUM_FREQ +
                                        j*NUM_ELEM +
                                        element_remapped;
                    if (cpu_address >= 0 && cpu_address < (NUM_TIMESAMPLES*NUM_FREQ*NUM_ELEM))
                        cpu_data[cpu_address] = host_PrimaryInput[0][k*NUM_FREQ*NUM_ELEM + j*NUM_ELEM+i];
                }
            }
        }

        err = 0;
        for (int k = 0; k < NUM_TIMESAMPLES; k++){
            if (CHECKING_VERBOSE && k < 5)
                printf("Timestep: %d\n", k);
            for (int j = 0; j < NUM_FREQ; j++){
                if (CHECKING_VERBOSE && k < 5)
                    printf("Freq: %d\n", j);
                for (int i = 0; i < NUM_ELEM; i++){
                    if (CHECKING_VERBOSE && k < 5)
                        printf("        Element %4d cpu: %2d %2d gpu %2d %2d ",i,HI_NIBBLE(cpu_data[k*NUM_ELEM*NUM_FREQ+j*NUM_ELEM+i]),LO_NIBBLE(cpu_data[k*NUM_ELEM*NUM_FREQ+j*NUM_ELEM+i]),HI_NIBBLE(host_PrimaryInput[1][k*NUM_ELEM*NUM_FREQ+j*NUM_ELEM+i]),LO_NIBBLE(host_PrimaryInput[1][k*NUM_ELEM*NUM_FREQ+j*NUM_ELEM+i]));
                    if (HI_NIBBLE(cpu_data[k*NUM_ELEM*NUM_FREQ+j*NUM_ELEM+i])!=HI_NIBBLE(host_PrimaryInput[1][k*NUM_ELEM*NUM_FREQ+j*NUM_ELEM+i]) || LO_NIBBLE(cpu_data[k*NUM_ELEM*NUM_FREQ+j*NUM_ELEM+i])!=LO_NIBBLE(host_PrimaryInput[1][k*NUM_ELEM*NUM_FREQ+j*NUM_ELEM+i])){
                        err++;
                        if (CHECKING_VERBOSE && k < 5)
                            printf("Err: %d", err);
                    }
                    if (CHECKING_VERBOSE && k < 5)
                        printf("\n");
                }
            }
        }
        printf("\nTotal errors: %d\n",err);
        free (cpu_data);
    }
    else{
        printf("GPU calculations have not been verified. If kernels have been changed, be careful regarding these results.\n\n");
    }

    printf("END OF LINE.\n");
    err = munlockall();


    if (err != SDK_SUCCESS) {
        printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
        printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
        exit(err);
    }

    for (int ns=0; ns < N_STAGES; ns++){
        err = clReleaseMemObject(device_CLinput_pinnedBuffer[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
//         err = clReleaseMemObject( device_CLoutput_pinnedBuffer[ns]);
//         if (err != SDK_SUCCESS) {
//             printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
//             printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
//             exit(err);
//         }
        err = clReleaseMemObject(device_CLinput_kernelData[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
//         err = clReleaseMemObject(device_CLoutput_kernelData[ns]);
//         if (err != SDK_SUCCESS) {
//             printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
//             printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
//             exit(err);
//         }
        assert(host_PrimaryInput[ns]!=NULL);
        free(host_PrimaryInput[ns]);
//         free(host_PrimaryOutput[ns]);

//         err = clReleaseMemObject(device_CLoutputAccum[ns]);
//         if (err != SDK_SUCCESS) {
//             printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
//             printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
//             exit(err);
//         }
    }

//    free(zeros);

    //--------------------------------------------------------------

    clReleaseKernel(reorder_kernel);
    clReleaseProgram(program);
//    clReleaseMemObject(device_block_lock);
//    clReleaseMemObject(id_x_map);
//    clReleaseMemObject(id_y_map);
    clReleaseCommandQueue(queue[0]);
    clReleaseCommandQueue(queue[1]);
    clReleaseContext(context);
    return 0;
}
