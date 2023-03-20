#include "cudaFRBBeamReformer.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "visUtil.hpp"

#include "fmt.hpp"

#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaFRBBeamReformer);

static float deg2radf(float d) {
    return M_PI * d / 180.;
}
static float cosdf(float d) {
    return cosf(deg2radf(d));
}
static float sindf(float d) {
    return sinf(deg2radf(d));
}
static float Ufunc(int p, int M, float theta) {
    float acc = 0;
    for (int s = 0; s <= M; s++) {
        float A = 1.;
        if (s == 0 || s == M)
            A = 0.5;
        acc += A * cos((M_PI * (2. * theta - p) * s) / M);
    }
    return acc;
}

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

    // HACK -- for Minimum-Viable-Product purposes, we're going to
    // hard-code the FRB beamformer beam locations on the sky.  In
    // turn, this requires knowing the actual frequencies we're
    // processing!
    float* beam_dra = (float*)malloc(_num_beams * sizeof(float));
    assert(beam_dra);
    float* beam_ddec = (float*)malloc(_num_beams * sizeof(float));
    assert(beam_ddec);
    float* freqs = (float*)malloc(_num_local_freq * sizeof(float));
    assert(freqs);

    // FIXME -- in the real thing this should be an updatable config
    // or somesuch.  Tracking beams would update single elements of
    // these arrays.
    // We'll place beams on an RA,Dec grid, expressed in *degrees*
    // from -1.0 to +1.0
    float dra_min = -1.0;
    float dra_max = +1.0;
    float ddec_min = -1.0;
    float ddec_max = +1.0;
    int Nra = 50;
    int Ndec = 100;
    assert(Nra * Ndec == _num_beams);
    assert(_num_beams == 5000);
    for (int i = 0; i < Nra; i++) {
        for (int j = 0; j < Ndec; j++) {
            beam_dra [i*Ndec+j] = dra_min  + (dra_max  - dra_min ) * (float)i / (Nra-1);
            beam_ddec[i*Ndec+j] = ddec_min + (ddec_max - ddec_min) * (float)j / (Ndec-1);
        }
    }

    // FIXME -- the real frequencies should be passed in with the
    // metadata, probably!
    for (int i = 0; i < _num_local_freq; i++) {
        freqs[i] = 600e6 + i*(1.2e9 / 65536.);
    }

    // FIXME -- the dish spacings, in meters, should be passed in as config parameters!
    float dish_spacing_ew = 6.3;
    float dish_spacing_ns = 8.5;

    // FIXME -- the zenith distance of the dishes, in degrees.  We'll
    // assume the dishes remain pointed at HA=0, ie, they're always on
    // the meridian.
    float bore_zd = 30.;

    const float c = 3.0e8;

    const int M = _beam_grid_size/2;
    const int N = M;

    float16_t* host_phase = (float16_t*)malloc(phase_len);
    assert(host_phase);

    INFO("Computing beam-reformer phase matrix...");
    for (int fi = 0; fi < _num_local_freq; fi++) {
        float wavelength = c / freqs[fi];

        for (int bi = 0; bi < _num_beams; bi++) {
            // Kendrick's FRB beamforming note, equation 7:
            //   theta = M (nhat . sigma) / lambda
            // where nhat is the unit vector in the direction of the sky location
            // sigma is the dish displacement in meters East-West.
            //
            // We'll assume that the dishes are pointed along the meridian, so
            // the boresight lies in the y,z plane (x=0)
            //   nhat_0_x = 0  (x is the direction of RA = EW, y of Dec = NS)
            //   nhat_0_y = cos(zd)
            //   nhat_0_z = sin(zd)
            // And nhat for each beam will be
            //   nhat_z ~ sin(zd - ddec)
            //   nhat_x ~ cos(zd - ddec) * sin(dra)
            //   nhat_y ~ cos(zd - ddec) * cos(dra)
            // We could probably get away with small-angle
            // approximations of beam_dra,beam_ddec,
            //   nhat_z ~ sin(zd)
            //   nhat_y ~ cos(zd) - ddec * sin(zd)    (cos(a-b) ~ cos(a) + b sin(a) when b->0); cos(dra)~1
            //   nhat_x ~ cos(zd) * dra
            // (but here we don't use the small-angle approx)
            float theta1 = cosdf(bore_zd - beam_ddec[bi]) * sindf(beam_dra[bi]) * M * dish_spacing_ew / wavelength;
            float theta2 = cosdf(bore_zd - beam_ddec[bi]) * cosdf(beam_dra[bi]) * N * dish_spacing_ns / wavelength;

            float Up[2*M];
            float Uq[2*N];
            for (int i = 0; i < 2*M; i++)
                Up[i] = Ufunc(i, M, theta1);
            for (int i = 0; i < 2*N; i++)
                Uq[i] = Ufunc(i, N, theta2);

            const int B = _num_beams;
            const int Q  = 2*N;
            const int PQ = 2*M * 2*N;

            for (int p = 0; p < 2*M; p++) {
                for (int q = 0; q < 2*N; q++) {
                    host_phase[(size_t)fi * (B * PQ) + bi * PQ + p * Q + q] = (float16_t)(Up[p] * Uq[q]);
                }
            }
        }
    }
    INFO("Computed beam-reformer phase matrix");

    float16_t* phase_memory =
        (float16_t*)device.get_gpu_memory(_gpu_mem_phase, phase_len);

    CHECK_CUDA_ERROR(cudaMemcpy(phase_memory, host_phase, phase_len, cudaMemcpyHostToDevice));

    free(host_phase);
    free(beam_dra);
    free(beam_ddec);
    free(freqs);
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
        (float16_t*)device.get_gpu_memory(_gpu_mem_phase, phase_len);
    float16_t* beamout_memory =
        (float16_t*)device.get_gpu_memory_array(_gpu_mem_beamout, gpu_frame_id, beamout_len);

    record_start_event(gpu_frame_id);

    DEBUG("Running CUDA FRB BeamReformer on GPU frame {:d}", gpu_frame_id);

    for (int f = 0; f < _num_local_freq; f++) {
        int T = Td;
        int B = _num_beams;
        float16_t* d_Iin = beamgrid_memory + (size_t)f * T * rho;
        float16_t* d_W = phase_memory + (size_t)f * rho * B;
        float16_t* d_Iout = beamout_memory + (size_t)f * T * B;

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
