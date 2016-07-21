#include "device_interface.h"
#include "gpu_command.h"
#include "callbackdata.h"
#include "math.h"
#include <errno.h>

device_interface::device_interface(struct Buffer * param_In_Buf, struct Buffer * param_Out_Buf, Config * param_Config
, int param_GPU_ID, struct Buffer * param_beamforming_out_buf)
{
    cl_int err;

    in_buf = param_In_Buf;
    out_buf = param_Out_Buf;
    config = param_Config;
    gpu_id = param_GPU_ID;
    beamforming_out_buf = param_beamforming_out_buf;

      // TODO explain these numbers/formulas.
    num_blocks = (param_Config->processing.num_adjusted_elements / param_Config->gpu.block_size) *
    (param_Config->processing.num_adjusted_elements / param_Config->gpu.block_size + 1) / 2.;

    accumulate_len = config->processing.num_adjusted_local_freq *
    config->processing.num_adjusted_elements * 2 * config->processing.num_data_sets * sizeof(cl_int);
    aligned_accumulate_len = PAGESIZE_MEM * (ceil((double)accumulate_len / (double)PAGESIZE_MEM));
    assert(aligned_accumulate_len >= accumulate_len);

//    stream_info = malloc(in_buf->num_buffers * sizeof(struct StreamINFO));
//    CHECK_MEM(stream_info);

    // Get a platform.
    CHECK_CL_ERROR( clGetPlatformIDs( 1, &platform_id, NULL ) );

    // Find a GPU device..
    CHECK_CL_ERROR( clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, MAX_GPUS, device_id, NULL) );

    context = clCreateContext( NULL, 1, &device_id[gpu_id], NULL, NULL, &err);
    CHECK_CL_ERROR(err);
}

int device_interface::getNumBlocks()
{
    return num_blocks;
}

int device_interface::getGpuID()
{
    return gpu_id;
}

Buffer* device_interface::getInBuf()
{
    return in_buf;
}

Buffer* device_interface::getOutBuf()
{
    return out_buf;
}

Buffer* device_interface::get_beamforming_out_buf()
{
    return beamforming_out_buf;
}

cl_context device_interface::getContext()
{
    return context;
}

cl_device_id device_interface::getDeviceID(int param_GPUID)
{
    return device_id[param_GPUID];
}

cl_int* device_interface::getAccumulateZeros()
{
    return accumulate_zeros;
}

int device_interface::getAlignedAccumulateLen() const
{
    return aligned_accumulate_len;
}

void device_interface::set_stream_id(int param_buffer_id)
{
    // Set the stream ID for the link.
    int32_t local_stream_id = get_streamID(in_buf, param_buffer_id);
    assert(local_stream_id != -1);
    stream_id = extract_stream_id(local_stream_id);
}

stream_id_t device_interface::get_stream_id()
{    
    return stream_id;
}

void device_interface::set_device_phases(cl_mem param_device_phases)
{
    device_phases = param_device_phases;
}

void device_interface::prepareCommandQueue()
{
    cl_int err;

    // Create command queues
    for (int i = 0; i < NUM_QUEUES; ++i) {
        queue[i] = clCreateCommandQueue( context, device_id[gpu_id], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err );
        CHECK_CL_ERROR(err);
    }
}

void device_interface::allocateMemory()
{
    //IN THE FUTURE, ANDRE TALKED ABOUT WANTING MEMORY TO BE DYNAMICALLY ALLOCATE BASED ON KERNEL STATES AND ASK FOR MEMORY BY SIZE AND HAVE THAT MEMORY NAMED BY THE KERNEL.
    //KERNELS WOULD THEN BE PASSED INTO DEVICE_INTERFACE
    //TO BE GIVEN MEMORY AND DEVICE_INTERFACE WOULD LOOP THROUGH THE KERNELS MEMORY STATES TO DETERMINE ALL THE MEMORY THAT KERNEL NEEDS.


    cl_int err;
    // TODO create a struct to contain all of these (including events) to make this memory allocation cleaner.

    // Setup device input buffers
    device_input_buffer = (cl_mem *) malloc(in_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(device_input_buffer);
    for (int i = 0; i < in_buf->num_buffers; ++i) {
        device_input_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, in_buf->aligned_buffer_size, NULL, &err);
        CHECK_CL_ERROR(err);
    }

    // Array used to zero the output memory on the device.
    // TODO should this be in it's own function?
    err = posix_memalign((void **) &accumulate_zeros, PAGESIZE_MEM, aligned_accumulate_len);
    if ( err != 0 ) {
        ERROR("Error creating aligned memory for accumulate zeros");
        exit(err);
    }

    // Ask that all pages be kept in memory
    err = mlock((void *) accumulate_zeros, aligned_accumulate_len);
    if ( err == -1 ) {
        ERROR("Error locking memory - check ulimit -a to check memlock limits");
        exit(errno);
    }
    memset(accumulate_zeros, 0, aligned_accumulate_len );

    // Setup device accumulate buffers.
    device_accumulate_buffer = (cl_mem *) malloc(in_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(device_accumulate_buffer);
    for (int i = 0; i < in_buf->num_buffers; ++i) {
        device_accumulate_buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, aligned_accumulate_len, NULL, &err);
        CHECK_CL_ERROR(err);
    }

    // Setup device output buffers.
    device_output_buffer = (cl_mem *) malloc(out_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(device_output_buffer);
    for (int i = 0; i < out_buf->num_buffers; ++i) {
        device_output_buffer[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_buf->aligned_buffer_size, NULL, &err);
        CHECK_CL_ERROR(err);
    }
    
    

##OCCURS IN SETUP_OPEN_CL    
    // Setup beamforming output buffers.
    if (config->gpu.use_beamforming == 1) {
        device_beamform_output_buffer = (cl_mem *) malloc(beamforming_out_buf->num_buffers * sizeof(cl_mem));
        CHECK_MEM(device_beamform_output_buffer);
        for (int i = 0; i < beamforming_out_buf->num_buffers; ++i) {
            device_beamform_output_buffer[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY
                    , beamforming_out_buf->aligned_buffer_size, NULL, &err);
            CHECK_CL_ERROR(err);
        }
    }
##
    
}

cl_mem device_interface::getInputBuffer(int param_BufferID)
{
    return device_input_buffer[param_BufferID];
}

cl_mem device_interface::getOutputBuffer(int param_BufferID)
{
    return device_output_buffer[param_BufferID];
}

cl_mem device_interface::getAccumulateBuffer(int param_BufferID)
{
  return device_accumulate_buffer[param_BufferID];
}

cl_mem device_interface::get_device_beamform_output_buffer(int param_BufferID)
{
    return device_beamform_output_buffer[param_BufferID];
}

cl_mem device_interface::get_device_phases()
{
    return device_phases;
}

void device_interface::deallocateResources()
{
    cl_int err;

    for (int i = 0; i < NUM_QUEUES; ++i) {
        CHECK_CL_ERROR( clReleaseCommandQueue(queue[i]) );
    }

    for (int i = 0; i < in_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(device_input_buffer[i]) );
    }
    free(device_input_buffer);

    for (int i = 0; i < in_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(device_accumulate_buffer[i]) );
    }
    free(device_accumulate_buffer);

    for (int i = 0; i < out_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(device_output_buffer[i]) );
    }
    free(device_output_buffer);

    err = munlock((void *) accumulate_zeros, aligned_accumulate_len);
    if ( err == -1 ) {
        ERROR("Error unlocking memory");
        exit(errno);
    }
    free(accumulate_zeros);

    CHECK_CL_ERROR( clReleaseContext(context) );

}

cl_command_queue device_interface::getQueue(int param_Dim)
{
    return queue[param_Dim];
}


