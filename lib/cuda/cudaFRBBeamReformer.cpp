#include "cudaFRBBeamReformer.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "visUtil.hpp"

#include "fmt.hpp"

#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaFRBBeamReformer);

cudaFRBBeamReformer::cudaFRBBeamReformer(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers,
                                         cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "FRB_beamreformer", "") {
    _num_beams = config.get<int>(unique_name, "num_beams");
    _beam_grid_size = config.get<int>(unique_name, "beam_grid_size");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _time_downsampling = config.get<int>(unique_name, "time_downsampling");

    _gpu_mem_beamgrid = config.get<std::string>(unique_name, "gpu_mem_beamgrid");
    _gpu_mem_phase = config.get<std::string>(unique_name, "gpu_mem_phase");
    _gpu_mem_beamout = config.get<std::string>(unique_name, "gpu_mem_beamout");

    set_command_type(gpuCommandType::KERNEL);

    rho = _beam_grid_size * _beam_grid_size;
    Td = (_samples_per_data_set + _time_downsampling - 1) / _time_downsampling;

    beamgrid_len = (size_t)_num_local_freq * Td * rho * sizeof(float16_t);
    phase_len = (size_t)_num_local_freq * rho * _num_beams * sizeof(float16_t);
    beamout_len = (size_t)_num_local_freq * Td * _num_beams * sizeof(float16_t);

    cublasStatus_t err = cublasCreate(&(this->handle));
    if (err != CUBLAS_STATUS_SUCCESS) {
        ERROR("Error at {:s}:{:d}: cublasCreate: {:s}", __FILE__, __LINE__,
              cublasGetStatusString(err));
        std::abort();
    }
}

cudaFRBBeamReformer::~cudaFRBBeamReformer() {
    cublasDestroy(this->handle);
}

cudaEvent_t cudaFRBBeamReformer::execute(int gpu_frame_id,
                                         const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(gpu_frame_id);

    float16_t* beamgrid_memory =
        (float16_t*)device.get_gpu_memory_array(_gpu_mem_beamgrid, gpu_frame_id, beamgrid_len);
    float16_t* phase_memory =
        (float16_t*)device.get_gpu_memory_array(_gpu_mem_phase, gpu_frame_id, phase_len);
    float16_t* beamout_memory =
        (float16_t*)device.get_gpu_memory_array(_gpu_mem_beamout, gpu_frame_id, beamout_len);

    record_start_event(gpu_frame_id);

    DEBUG("Running CUDA FRB BeamReformer on GPU frame {:d}", gpu_frame_id);

    for (int f = 0; f < _num_local_freq; f++) {
        int T = Td;
        int B = _num_beams;
        float16_t* d_Iin = beamgrid_memory + f * T * rho;
        float16_t* d_W = phase_memory + f * rho * B;
        float16_t* d_Iout = beamout_memory + f * T * B;

        __half alpha = 1.;
        __half beta = 0.;

        // Multiply A and B^T on GPU
        cublasStatus_t stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, T, B, rho, &alpha,
                                          d_Iin, T, d_W, B, &beta, d_Iout, T);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            ERROR("Error at {:s}:{:d}: cublasHgemm: {:s}", __FILE__, __LINE__,
                  cublasGetStatusString(stat));
            std::abort();
        }
    }
    return record_end_event(gpu_frame_id);
}
