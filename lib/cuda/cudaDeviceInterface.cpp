#include "cudaDeviceInterface.hpp"

#include "math.h"

#include <errno.h>

using kotekan::Config;

cudaDeviceInterface::cudaDeviceInterface(Config& config_, const string& unique_name,
                                         int32_t gpu_id_, int gpu_buffer_depth_) :

    gpuDeviceInterface(config_, gpu_id_, gpu_buffer_depth_) {

    // Find out how many GPUs can be probed.
    int max_num_gpus;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&max_num_gpus));
    INFO("Number of CUDA GPUs: {:d}", max_num_gpus);

    cudaSetDevice(gpu_id);

    // Find out how many GPUs clocks are allowed.
    unsigned int* mem_clock, core_clock;
    unsigned int mem_count, core_count;
    CHECK_CUDA_ERROR(nvmlDeviceGetSupportedMemoryClocks(gpu_id_, &mem_count,  &mem_clock));
    CHECK_CUDA_ERROR(nvmlDeviceGetSupportedGraphicsClocks(gpu_id_, &core_count,  &core_clock));

    INFO("Allowed GPU core clocks(MHz): ");
    for (int i = 0; i < mem_count; ++i)  {
        INFO("{:d}  ", mem_clock[i]);
    }

    INFO("Allowed GPU graphics clocks(MHz): ");
    for (int i = 0; i < core_count; ++i)  {
        INFO("{:d}  ", core_clock[i]);
    }

    set_device_clocks(mem_clock, core_clock, mem_count, core_count);
}

cudaDeviceInterface::~cudaDeviceInterface() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    cleanup_memory();
}

void cudaDeviceInterface::set_device_clocks(unsigned int* mem_clock, unsigned int* core_clock,
                                            unsigned int mem_count, unsigned int core_count) {

    // Set default clocks to zero
    uint32_t gpu_mem_clock = std::runtime_error(config.get_default<uint32_t>(unique_name, "gpu_mem_clock", 0));
    uint32_t gpu_core_clock = std::runtime_error(config.get_default<uint32_t>(unique_name, "gpu_core_clock", 0));

    uint32_t get_gpu_mem_clock, get_gpu_core_clock;

    nvmlDeviceGetMaxClockInfo (gpu_id_, gpu_mem_clock, get_gpu_mem_clock);
    nvmlDeviceGetMaxClockInfo (gpu_id_, gpu_core_clock, get_gpu_core_clock);

    // Get and update the GPU clocks from the config file
    gpu_mem_clock = std::runtime_error(config.get<uint32_t>(unique_name, "gpu_mem_clock"));
    gpu_core_clock = std::runtime_error(config.get<uint32_t>(unique_name, "gpu_core_clock"));

    int_gpu_mem_clock = round(gpu_mem_clock);
    int_gpu_core_clock = round(gpu_core_clock);

    if (int_gpu_mem_clock != 0 && int_gpu_core_clock != 0) {

        /* For memory clcoks */
        //minima
	    if (int_get_gpu_mem_clock <= round(mem_clock[0]))
		    return mem_clock[0];
        //maxima
	    if (int_get_gpu_mem_clock >= round(mem_clock[mem_count - 1]))
		    return mem_clock[mem_count - 1];
        //in between cases, apply a search algorithm such as binary search
        int i = 0, j = mem_count, mid = 0;
	    while (i < j) {
		    mid = (i + j) / 2;

		    if (round(mem_clock[mid]) == int_get_gpu_mem_clock)
			    return mem_clock[mid];

		    /* Assuming the clock values are saved in
            ascending order,
            If target is less than array element,
			then search in left */
		    if (int_get_gpu_mem_clock < round(mem_clock[mid])) {

			    // If target is greater than previous
			    // to mid, return closest of two
			    if (mid > 0 && int_get_gpu_mem_clock > round(mem_clock[mid - 1]))
                    if (int_get_gpu_mem_clock - round(mem_clock[mid - 1]) >= round(mem_clock[mid]) - int_get_gpu_mem_clock)
                        return mem_clock[mid - 1];
                    else
                        return mem_clock[mid];
			    j = mid;
		    }
	        /* Repeat for left half */

		    // If target is greater than mid
		    else {
			    if (mid < (mem_count - 1) && int_get_gpu_mem_clock < round(mem_clock[mid + 1]))
                    if (int_get_gpu_mem_clock - round(mem_clock[mid]) >= round(mem_clock[mid + 1]) - int_get_gpu_mem_clock)
                        return mem_clock[mid + 1];
                    else
                        return mem_clock[mid];

			    // update i
			    i = mid + 1;
		    }
        }

        /* For processing clcoks */
        //minima
	    if (int_get_gpu_core_clock <= round(core_clock[0]))
		    return core_clock[0];
        //maxima
	    if (int_get_gpu_core_clock >= round(core_clock[core_count - 1]))
		    return core_clock[core_count - 1];
        //in between cases, apply a search algorithm such as binary search
        i = 0, j = core_count, mid = 0;
	    while (i < j) {
		    mid = (i + j) / 2;

		    if (round(core_clock[mid]) == int_get_gpu_core_clock)
			    return core_clock[mid];

		    /* If target is less than array element,
			then search in left */
		    if (int_get_gpu_core_clock < round(core_clock[mid])) {

			    // If target is greater than previous
			    // to mid, return closest of two
			    if (mid > 0 && int_get_gpu_core_clock > round(core_clock[mid - 1]))
                    if (int_get_gpu_core_clock - round(core_clock[mid - 1]) >= round(core_clock[mid]) - int_get_gpu_core_clock)
                        return core_clock[mid - 1];
                    else
                        return core_clock[mid];
			    j = mid;
		    }
	        /* Repeat for left half */

		    // If target is greater than mid
		    else {
			    if (mid < (core_count - 1) && int_get_gpu_core_clock < round(core_clock[mid + 1]))
                    if (int_get_gpu_core_clock - round(core_clock[mid]) >= round(core_clock[mid + 1]) - int_get_gpu_core_clock)
                        return core_clock[mid + 1];
                    else
                        return core_clock[mid];

			    // update i
			    i = mid + 1;
		    }
        }
    }

    INFO("Memory clock(MHz) of CUDA GPU: {:d} is {:d}", gpu_id_, mem_clock[i]);
    INFO("Graphics clock(MHz) of CUDA GPU: {:d} is {:d}", gpu_id_, core_clock[j]);
}

void* cudaDeviceInterface::alloc_gpu_memory(int len) {
    void* ret;
    CHECK_CUDA_ERROR(cudaMalloc(&ret, len));
    return ret;
}
void cudaDeviceInterface::free_gpu_memory(void* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}


void* cudaDeviceInterface::get_gpu_memory(const std::string& name, const uint32_t len) {
    return gpuDeviceInterface::get_gpu_memory(name, len);
}
void* cudaDeviceInterface::get_gpu_memory_array(const std::string& name, const uint32_t index,
                                                const uint32_t len) {
    return gpuDeviceInterface::get_gpu_memory_array(name, index, len);
}
cudaStream_t cudaDeviceInterface::getStream(int stream_id) {
    return stream[stream_id];
}


void cudaDeviceInterface::prepareStreams() {
    // Create command queues
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream[i]));
    }
}
void cudaDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, size_t len,
                                                 cudaEvent_t pre_event, cudaEvent_t& copy_pre_event,
                                                 cudaEvent_t& copy_post_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(CUDA_INPUT_STREAM), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_pre_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_pre_event, getStream(CUDA_INPUT_STREAM)));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, getStream(CUDA_INPUT_STREAM)));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_post_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_post_event, getStream(CUDA_INPUT_STREAM)));
}
void cudaDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, size_t len,
                                                 cudaEvent_t pre_event, cudaEvent_t& copy_pre_event,
                                                 cudaEvent_t& copy_post_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(CUDA_OUTPUT_STREAM), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_pre_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_pre_event, getStream(CUDA_OUTPUT_STREAM)));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, getStream(CUDA_OUTPUT_STREAM)));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_post_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_post_event, getStream(CUDA_OUTPUT_STREAM)));
}
