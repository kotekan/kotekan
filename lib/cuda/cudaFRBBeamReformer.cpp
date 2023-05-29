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
    return d * (float)(M_PI / 180.);
}
static float cosdf(float d) {
    return cosf(deg2radf(d));
}
static float sindf(float d) {
    return sinf(deg2radf(d));
}

double gettime() {
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return tp.tv_sec + tp.tv_usec / 1.0e+6;
}

// template<int M>
// static float Ufunc(int p, float theta) {
static float Ufunc(int p, int M, float theta) {
    float acc = 0;
    for (int s = 0; s <= M; s++) {
        float A = 1.;
        if (s == 0 || s == M)
            A = 0.5;
        acc += A * cosf(((float)M_PI * (2.0f * theta - p) * s) / M);
    }
    return acc;
}
// template<int M, int N>
void compute_beam_reformer_phase(int M, int N, float16_t* host_phase, int _num_local_freq,
                                 int _num_beams, float* beam_dra, float* beam_ddec, float* freqs,
                                 float dish_spacing_ew, float dish_spacing_ns, float bore_zd) {
    const float c = 3.0e8;
    const int B = _num_beams;
    const int Q = 2 * N;
    const int PQ = 2 * M * 2 * N;

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
            //   nhat_y ~ cos(zd) - ddec * sin(zd)    (cos(a-b) ~ cos(a) + b sin(a) when b->0);
            //   cos(dra)~1 nhat_x ~ cos(zd) * dra
            // (but here we don't use the small-angle approx)
            float theta1 = cosdf(bore_zd - beam_ddec[bi]) * sindf(beam_dra[bi]) * M
                           * dish_spacing_ew / wavelength;
            float theta2 = cosdf(bore_zd - beam_ddec[bi]) * cosdf(beam_dra[bi]) * N
                           * dish_spacing_ns / wavelength;

            float Up[2 * M];
            float Uq[2 * N];
            for (int i = 0; i < 2 * M; i++)
                // Up[i] = Ufunc<M>(i, theta1);
                Up[i] = Ufunc(i, M, theta1);
            for (int i = 0; i < 2 * N; i++)
                // Uq[i] = Ufunc<N>(i, theta2);
                Uq[i] = Ufunc(i, N, theta2);

            for (int p = 0; p < 2 * M; p++) {
                for (int q = 0; q < 2 * N; q++) {
                    host_phase[(size_t)fi * (B * PQ) + bi * PQ + p * Q + q] =
                        (float16_t)(Up[p] * Uq[q]);
                }
            }
        }
    }
}

cudaFRBBeamReformer::cudaFRBBeamReformer(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers,
                                         cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "FRB_beamreformer", "") {
    _num_beams = config.get<int>(unique_name, "num_beams");
    _beam_grid_size = config.get<int>(unique_name, "beam_grid_size");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _Td = config.get<int>(unique_name, "samples_per_data_set");
    _batched = config.get_default<bool>(unique_name, "batched", true);
    _gpu_mem_beamgrid = config.get<std::string>(unique_name, "gpu_mem_beamgrid");
    _gpu_mem_phase = config.get<std::string>(unique_name, "gpu_mem_phase");
    _gpu_mem_beamout = config.get<std::string>(unique_name, "gpu_mem_beamout");
    _cuda_streams =
        config.get_default<std::vector<int>>(unique_name, "cuda_streams", std::vector<int>());
    for (size_t i = 0; i < _cuda_streams.size(); i++) {
        if (_cuda_streams[i] >= device.get_num_streams()) {
            ERROR("Error: cudaFRBBeamReformer's config setting cuda_streams must have all elements "
                  "< number of streams on the device = {:d}",
                  device.get_num_streams());
        }
    }

    sync_events.resize(_gpu_buffer_depth);
    if (_cuda_streams.size()) {
        for (int i = 0; i < _gpu_buffer_depth; i++) {
            sync_events[i].resize(_cuda_streams.size());
        }
    }

    set_command_type(gpuCommandType::KERNEL);

    rho = _beam_grid_size * _beam_grid_size;

    beamgrid_len = (size_t)_num_local_freq * _Td * rho * sizeof(float16_t);
    phase_len = (size_t)_num_local_freq * rho * _num_beams * sizeof(float16_t);
    beamout_len = (size_t)_num_local_freq * _Td * _num_beams * sizeof(float16_t);

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
            beam_dra[i * Ndec + j] = dra_min + (dra_max - dra_min) * (float)i / (Nra - 1);
            beam_ddec[i * Ndec + j] = ddec_min + (ddec_max - ddec_min) * (float)j / (Ndec - 1);
        }
    }

    // FIXME -- the real frequencies should be passed in with the
    // metadata, probably!
    for (int i = 0; i < _num_local_freq; i++) {
        freqs[i] = 600e6 + i * (1.2e9 / 65536.);
    }

    // FIXME -- the dish spacings, in meters, should be passed in as config parameters!
    float dish_spacing_ew = 6.3;
    float dish_spacing_ns = 8.5;

    // FIXME -- the zenith distance of the dishes, in degrees.  We'll
    // assume the dishes remain pointed at HA=0, ie, they're always on
    // the meridian.
    float bore_zd = 30.;

    float16_t* host_phase = (float16_t*)malloc(phase_len);
    assert(host_phase);

    const char* beamphase_cache_fn = "beamphase-cache.bin";
    FILE* f = fopen(beamphase_cache_fn, "r");
    bool gotit = false;
    if (f) {
        INFO("Trying to read beamformer phase matrix from {:s}...", beamphase_cache_fn);
        double t0 = gettime();
        size_t nr = fread(host_phase, 1, phase_len, f);
        if (nr != phase_len) {
            INFO("Reading file {:s}: got {:d} bytes, but expected {:d}", beamphase_cache_fn, nr,
                 phase_len);
        } else {
            gotit = true;
        }
        fclose(f);
        INFO("That took {:g} sec", gettime() - t0);
    }
    if (!gotit) {
        int M = _beam_grid_size / 2;
        int N = M;
        INFO("Computing beam-reformer phase matrix...");
        double t0 = gettime();
        compute_beam_reformer_phase(M, N, host_phase, _num_local_freq, _num_beams, beam_dra,
                                    beam_ddec, freqs, dish_spacing_ew, dish_spacing_ns, bore_zd);
        INFO("That took {:g} sec", gettime() - t0);
        INFO("Computed beam-reformer phase matrix");
        f = fopen(beamphase_cache_fn, "w+");
        if (f) {
            size_t nw = fwrite(host_phase, 1, phase_len, f);
            if (nw != phase_len) {
                INFO("Failed to write beamformer phase matrix to {:s}: {:s}", beamphase_cache_fn,
                     strerror(errno));
                fclose(f);
                unlink(beamphase_cache_fn);
            } else {
                if (fclose(f)) {
                    INFO("Failed to close beamformer phase matrix in {:s}: {:s}",
                         beamphase_cache_fn, strerror(errno));
                    unlink(beamphase_cache_fn);
                } else {
                    INFO("Wrote beamformer phase matrix to {:s}", beamphase_cache_fn);
                }
            }
        }
    }

    float16_t* phase_memory = (float16_t*)device.get_gpu_memory(_gpu_mem_phase, phase_len);

    CHECK_CUDA_ERROR(cudaMemcpy(phase_memory, host_phase, phase_len, cudaMemcpyHostToDevice));

    free(host_phase);
    free(beam_dra);
    free(beam_ddec);
    free(freqs);

    int nstreams = std::max((int)_cuda_streams.size(), 1);
    if (_num_local_freq % nstreams != 0) {
        ERROR("Number of CUDA streams must evenly divide number of frequencies!");
    }
    if (nstreams == 1) {
        // We MUST set the stream -- otherwise it uses the CUDA
        // default stream, which != our default compute stream!
        DEBUG("Set cublas stream {:d}", cuda_stream_id);
        cublasSetStream(handle, device.getStream(cuda_stream_id));
    }

    // for cublas_hgemmBatched, pre-compute the GPU array pointers.
    if (_batched) {
        // GPU-memory arrays of pointers to the input & output
        // matrices (in GPU memory)
        __half** in;
        __half** out;
        __half** ph;
        in = (__half**)device.get_gpu_memory(unique_name + "_batch_in",
                                             _gpu_buffer_depth * _num_local_freq * sizeof(__half*));
        out = (__half**)device.get_gpu_memory(
            unique_name + "_batch_out", _gpu_buffer_depth * _num_local_freq * sizeof(__half*));
        ph = (__half**)device.get_gpu_memory(unique_name + "_batch_ph",
                                             _num_local_freq * sizeof(__half*));

        // Temporary CPU-memory arrays of pointers to the input &
        // output matrices (which live in GPU memory); these will get
        // copied to in/out/ph.
        __half* host_in[_gpu_buffer_depth * _num_local_freq];
        __half* host_out[_gpu_buffer_depth * _num_local_freq];
        __half* host_ph[_gpu_buffer_depth * _num_local_freq];

        // Final CPU-side pointers to GPU memory
        _gpu_in_pointers.resize(_gpu_buffer_depth);
        _gpu_out_pointers.resize(_gpu_buffer_depth);

        int freqs_per_stream = _num_local_freq / nstreams;

        // loop over gpu frames
        for (int gpu_frame_id = 0; gpu_frame_id < _gpu_buffer_depth; gpu_frame_id++) {
            // GPU input & output memory buffers for this gpu frame #.
            float16_t* gpu_in_base = (float16_t*)device.get_gpu_memory_array(
                _gpu_mem_beamgrid, gpu_frame_id, beamgrid_len);
            float16_t* gpu_out_base = (float16_t*)device.get_gpu_memory_array(
                _gpu_mem_beamout, gpu_frame_id, beamout_len);
            // Compute the per-frequency matrix offsets.
            for (int f = 0; f < _num_local_freq; f++) {
                host_in[gpu_frame_id * _num_local_freq + f] = gpu_in_base + (size_t)f * _Td * rho;
                host_out[gpu_frame_id * _num_local_freq + f] =
                    gpu_out_base + (size_t)f * _Td * _num_beams;
                if (gpu_frame_id == 0)
                    host_ph[f] = phase_memory + (size_t)f * rho * _num_beams;
            }
            // loop over streams and save the final GPU memory pointers (which we haven't yet filled
            // with data!)
            for (int i = 0; i < nstreams; i++) {
                _gpu_in_pointers[gpu_frame_id].push_back(
                    in + (gpu_frame_id * nstreams + i) * freqs_per_stream);
                _gpu_out_pointers[gpu_frame_id].push_back(
                    out + (gpu_frame_id * nstreams + i) * freqs_per_stream);
                if (gpu_frame_id == 0)
                    _gpu_phase_pointers.push_back(ph + i * freqs_per_stream);
            }
        }
        // Now copy the GPU pointer offsets (that we computed on the CPU) over to the GPU!
        CHECK_CUDA_ERROR(cudaMemcpy(in, host_in,
                                    _gpu_buffer_depth * _num_local_freq * sizeof(__half*),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(out, host_out,
                                    _gpu_buffer_depth * _num_local_freq * sizeof(__half*),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(
            cudaMemcpy(ph, host_ph, _num_local_freq * sizeof(__half*), cudaMemcpyHostToDevice));
    }
}

cudaFRBBeamReformer::~cudaFRBBeamReformer() {
    cublasDestroy(this->handle);
}

cudaEvent_t cudaFRBBeamReformer::execute(cudaPipelineState& pipestate,
                                         const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    float16_t* beamgrid_memory =
        (float16_t*)device.get_gpu_memory_array(_gpu_mem_beamgrid, pipestate.gpu_frame_id, beamgrid_len);
    float16_t* phase_memory = (float16_t*)device.get_gpu_memory(_gpu_mem_phase, phase_len);
    float16_t* beamout_memory =
        (float16_t*)device.get_gpu_memory_array(_gpu_mem_beamout, pipestate.gpu_frame_id, beamout_len);

    record_start_event(pipestate.gpu_frame_id);

    DEBUG("Running CUDA FRB BeamReformer on GPU frame {:d}: F={:d}, T={:d}, B={:d}, rho={:d}",
          pipestate.gpu_frame_id, _num_local_freq, _Td, _num_beams, rho);

    int calls_per_stream = (_cuda_streams.size() > 0) ? _num_local_freq / _cuda_streams.size() : 0;

    if (_batched) {
        int nstreams = std::max((int)_cuda_streams.size(), 1);
        int freqs_per_stream = _num_local_freq / nstreams;

        for (int i = 0; i < nstreams; i++) {
            if (nstreams > 1) {
                DEBUG("Set cublas stream {:d}", _cuda_streams[i]);
                cublasSetStream(handle, device.getStream(_cuda_streams[i]));
            }
            __half alpha = 1.;
            __half beta = 0.;
            cublasStatus_t stat = cublasHgemmBatched(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, _Td, _num_beams, rho, &alpha,
                _gpu_in_pointers[pipestate.gpu_frame_id][i], rho, _gpu_phase_pointers[i], rho, &beta,
                _gpu_out_pointers[pipestate.gpu_frame_id][i], _Td, freqs_per_stream);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                ERROR("Error at {:s}:{:d}: cublasHgemmBatched: {:s}", __FILE__, __LINE__,
                      cublasGetStatusString(stat));
                std::abort();
            }
        }

    } else {
        for (int f = 0; f < _num_local_freq; f++) {
            int B = _num_beams;
            float16_t* d_Iin = beamgrid_memory + (size_t)f * _Td * rho;
            float16_t* d_W = phase_memory + (size_t)f * rho * B;
            float16_t* d_Iout = beamout_memory + (size_t)f * _Td * B;

            __half alpha = 1.;
            __half beta = 0.;

            if ((calls_per_stream > 0) && (f % calls_per_stream == 0)) {
                DEBUG("Freq {:d}: set Cuda stream {:d}", f, _cuda_streams[f / calls_per_stream]);
                cublasSetStream(handle, device.getStream(_cuda_streams[f / calls_per_stream]));
            }
            /*
            // Multiply A and B^T on GPU
            cublasStatus_t stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, _Td, B, rho, &alpha,
            d_Iin, _Td, d_W, B, &beta, d_Iout, _Td);
            */

            // Multiply A^T and B on GPU, where the transposes are
            // according to cublas, ie, in the Fortran column-major view
            // of things.  That is, Transpose is the regular C row-major
            // ordering.
            cublasStatus_t stat = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, _Td, B, rho, &alpha,
                                              d_Iin, rho, d_W, rho, &beta, d_Iout, _Td);

            if (stat != CUBLAS_STATUS_SUCCESS) {
                ERROR("Error at {:s}:{:d}: cublasHgemm: {:s}", __FILE__, __LINE__,
                      cublasGetStatusString(stat));
                std::abort();
            }
        }
    }

    if (calls_per_stream > 0) {
        for (size_t i = 0; i < _cuda_streams.size(); i++) {
            if (_cuda_streams[i] != cuda_stream_id) {
                // Create an event on each compute stream that we "forked"
                CHECK_CUDA_ERROR(cudaEventCreate(&sync_events[pipestate.gpu_frame_id][i]));
                CHECK_CUDA_ERROR(cudaEventRecord(sync_events[pipestate.gpu_frame_id][i],
                                                 device.getStream(_cuda_streams[i])));
                // Now wait for that event on the main compute stream.
                CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(cuda_stream_id),
                                                     sync_events[pipestate.gpu_frame_id][i]));
            }
        }
    }

    return record_end_event(pipestate.gpu_frame_id);
}

void cudaFRBBeamReformer::finalize_frame(int frame_id) {

    float exec_time;
    for (size_t i = 0; i < _cuda_streams.size(); i++) {
        if (sync_events[frame_id][i] && sync_events[frame_id][i]) {
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&exec_time, start_events[frame_id],
                                                  synced_events[frame_id][i]));
            INFO("Sync for stream {:d} took {:.3f} ms", _cuda_streams[i], exec_time);
        }
    }
    if (start_events[frame_id] && end_events[frame_id]) {
        CHECK_CUDA_ERROR(
            cudaEventElapsedTime(&exec_time, start_events[frame_id], end_events[frame_id]));
        INFO("Start to end took {:.3f} ms", exec_time);
    }


    cudaCommand::finalize_frame(frame_id);
    for (size_t i = 0; i < _cuda_streams.size(); i++) {
        if (sync_events[frame_id][i])
            CHECK_CUDA_ERROR(cudaEventDestroy(sync_events[frame_id][i]));
        sync_events[frame_id][i] = nullptr;
    }
}
