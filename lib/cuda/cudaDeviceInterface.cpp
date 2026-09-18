#include "cudaDeviceInterface.hpp"

#include "cudaUtils.hpp"      // for CHECK_CUDA_ERROR
#include "cuda_runtime_api.h" // for cudaEventCreate, cudaEventRecord, cudaMemcpyAsync, cudaStr...
#include "kotekanLogging.hpp" // for FATAL_ERROR, INFO, WARN, DEBUG2

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>        // for assert
#include <chrono>          // for steady_clock
#include <cuda.h>          // for cuGetErrorString, cuModuleGetFunction, cuModuleLoadDataEx
#include <mutex>           // for mutex, lock_guard
#include <nvPTXCompiler.h> // for NVPTXCOMPILE_SUCCESS, nvPTXCompilerCompile, nvPTXCompilerC...
#include <nvrtc.h>         // for nvrtcGetErrorString, NVRTC_SUCCESS, nvrtcCompileProgram
#include <stdexcept>       // for runtime_error
#include <stdio.h>         // for fclose, fopen, fread, fseek, ftell, rewind, FILE, SEEK_END
#include <stdlib.h>        // for free, malloc
#include <thread>          // for this_thread::sleep_for
#include <utility>         // for pair

using kotekan::Config;

std::map<int32_t, std::weak_ptr<cudaDeviceInterface>> cudaDeviceInterface::inst_map;

// Protects access to inst_map
static std::mutex cuda_inst_map_mutex;

std::shared_ptr<cudaDeviceInterface>
cudaDeviceInterface::get(int32_t gpu_id, const std::string& name, Config& config) {
    std::lock_guard<std::mutex> lock(cuda_inst_map_mutex);
    auto it = inst_map.find(gpu_id);
    if (it != inst_map.end()) {
        if (auto existing = it->second.lock())
            // it->second is a std::weak_ptr. lock() attempts to create a new
            // shared_ptr that shares ownership with any existing shared owners.
            // If the weak_ptr has not expired, 'existing' becomes a valid shared_ptr.
            return existing;
    }
    auto dev = std::make_shared<cudaDeviceInterface>(config, name,
                                                     gpu_id); // creates an owning std::shared_ptr
    inst_map[gpu_id] = dev; // store weak reference (implicit conversion)
    return dev;
}

cudaDeviceInterface::cudaDeviceInterface(Config& config, const std::string& unique_name,
                                         int32_t gpu_id) :
    gpuDeviceInterface(config, unique_name, gpu_id) {

    // Find out how many GPUs can be probed.
    int max_num_gpus;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&max_num_gpus));
    INFO("Number of CUDA GPUs: {:d}", max_num_gpus);

    if (gpu_id > max_num_gpus) {
        throw std::runtime_error(
            "Asked for a GPU ID which is higher than the maximum number of GPUs in the system");
    }

    set_thread_device();
}

cudaDeviceInterface::~cudaDeviceInterface() {
    watchdog_stop = true;
    if (watchdog_thread.joinable())
        watchdog_thread.join();
    for (auto& stream : streams) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }
    cleanup_memory();
}

// ---------------------------------------------------------------------- the wedge probe

static int64_t probe_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::shared_ptr<cudaDeviceInterface::InFlight> cudaDeviceInterface::register_in_flight() {
    auto rec = std::make_shared<InFlight>();
    std::lock_guard<std::mutex> lock(in_flight_mutex);
    in_flight.push_back(rec);
    return rec;
}

void cudaDeviceInterface::start_wedge_watchdog(double period_s, double stuck_s) {
    // Every cudaProcess on the device calls this; exactly one thread must result. Stage
    // construction is serial today, but a probe that silently starts twice would report the
    // same stall twice and cost an hour of confusion.
    static std::mutex start_mutex;
    std::lock_guard<std::mutex> guard(start_mutex);
    if (period_s <= 0.0 || watchdog_thread.joinable())
        return;
    watchdog_period_s = period_s;
    watchdog_stuck_s = stuck_s;
    watchdog_thread = std::thread(&cudaDeviceInterface::watchdog_loop, this);
    // WARN, not INFO: nodes run at log_level WARN, and a probe whose own startup line is
    // invisible cannot be shown to have started. This IS the falsifier for the run.
    WARN("GPU[{:d}] WEDGE WATCHDOG ARMED: polling every {:.1f} s, reporting any command or "
         "lock wait held longer than {:.1f} s",
         gpu_id, period_s, stuck_s);
}

void cudaDeviceInterface::watchdog_loop() {
    // The watchdog must never take a queuing lock and never make a blocking CUDA call, or it
    // becomes another victim of the fault it exists to describe. cudaStreamQuery returns
    // immediately; set_thread_device is needed because every thread touching this device
    // must select it.
    set_thread_device();
    const auto tick = std::chrono::duration<double>(watchdog_period_s);
    while (!watchdog_stop) {
        std::this_thread::sleep_for(tick);
        if (watchdog_stop)
            break;
        const int64_t now = probe_now_us();

        // 1. Any pipeline that entered a command and never came out.
        std::vector<std::shared_ptr<InFlight>> recs;
        {
            std::lock_guard<std::mutex> lock(in_flight_mutex);
            recs = in_flight;
        }
        int n_stuck = 0, n_waiting = 0;
        std::vector<std::pair<std::string, double>> starved;
        for (const auto& r : recs) {
            std::string stage, command;
            int32_t stream;
            int64_t frame, age_us;
            bool holds, waits;
            {
                std::lock_guard<std::mutex> lock(r->m);
                if (!r->active) {
                    // Not in a command. If it has not finished one recently either, it is
                    // starved: blocked on its inputs, which is what a victim of the wedge
                    // looks like from here.
                    if (r->frames_done > 0
                        && now - r->last_done_us > (int64_t)(watchdog_stuck_s * 1e6))
                        starved.emplace_back(r->stage, (now - r->last_done_us) / 1e6);
                    continue;
                }
                stage = r->stage;
                command = r->command;
                stream = r->stream;
                frame = r->frame;
                age_us = now - r->entered_us;
                holds = r->holds_lock;
                waits = r->waiting;
            }
            if (age_us < (int64_t)(watchdog_stuck_s * 1e6))
                continue;
            n_stuck++;
            n_waiting += waits ? 1 : 0;
            WARN("*** GPU[{:d}] STUCK {:.1f} s: {:s} in {:s}, stream {:d}, frame {:d}{:s}{:s}",
                 gpu_id, age_us / 1e6, stage, command, stream, frame,
                 holds ? " -- HOLDS THE QUEUING LOCK" : "",
                 waits ? " -- WAITING FOR THE QUEUING LOCK" : "");
        }
        if (n_stuck == 0 && starved.empty())
            continue;
        for (const auto& sv : starved)
            WARN("*** GPU[{:d}] STARVED {:.1f} s: {:s} has finished no frame -- blocked on its "
                 "inputs, not on the GPU",
                 gpu_id, sv.second, sv.first);

        // 2. Is the GPU itself draining? This is the bit that splits the hypothesis: every
        //    stream idle while a thread sits in the driver means the block is host-side.
        std::string busy;
        int n_busy = 0;
        for (size_t i = 0; i < streams.size(); ++i) {
            const cudaError_t q = cudaStreamQuery(streams[i]);
            if (q == cudaErrorNotReady) {
                n_busy++;
                busy += (busy.empty() ? "" : ",") + std::to_string(i);
            } else if (q != cudaSuccess) {
                // Clear it: a sticky error here would be reported by the next CHECK elsewhere
                // and we must not turn a diagnostic into a fault of its own.
                cudaGetLastError();
                busy += (busy.empty() ? "" : ",") + std::to_string(i) + "=ERR";
            }
        }
        WARN("*** GPU[{:d}] WEDGE STATE: {:d} stuck ({:d} of them merely waiting), {:d} of "
             "{:d} streams still busy [{:s}], {:d} CUDA events outstanding",
             gpu_id, n_stuck, n_waiting, n_busy, (int)streams.size(), busy.empty() ? "none" : busy,
             (long long)events_outstanding.load());
    }
}

void cudaDeviceInterface::set_thread_device() {
    CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
}

void* cudaDeviceInterface::alloc_gpu_memory(size_t len) {
    void* ret;
    CHECK_CUDA_ERROR(cudaMalloc(&ret, len));
    return ret;
}
void cudaDeviceInterface::free_gpu_memory(void* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

cudaStream_t cudaDeviceInterface::getStream(int32_t cuda_stream_id) {
    return streams[cuda_stream_id];
}

int32_t cudaDeviceInterface::get_num_streams() {
    return streams.size();
}

std::recursive_mutex& cudaDeviceInterface::stream_mutex(int32_t stream_id) {
    if (stream_id < 0 || stream_id >= MAX_CUDA_STREAMS)
        throw std::runtime_error(fmt::format("stream_mutex: stream {:d} outside [0, {:d})",
                                             stream_id, MAX_CUDA_STREAMS));
    return stream_mutexes[stream_id];
}

void cudaDeviceInterface::prepareStreams(uint32_t num_streams) {
    if (num_streams > (uint32_t)MAX_CUDA_STREAMS)
        throw std::runtime_error(
            fmt::format("prepareStreams: asked for {:d} streams, the per-stream mutex array "
                        "holds {:d} -- raise MAX_CUDA_STREAMS",
                        num_streams, MAX_CUDA_STREAMS));
    // Create GPU command queues
    for (uint32_t i = streams.size(); i < num_streams; ++i) {
        cudaStream_t stream = nullptr;
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
}

void cudaDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t* copy_start_event,
                                                 cudaEvent_t* copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    if (copy_start_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_start_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_start_event, getStream(cuda_stream_id)));
    }
    // Data transfer to GPU
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, getStream(cuda_stream_id)));
    if (copy_end_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_end_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_end_event, getStream(cuda_stream_id)));
    }
}

void cudaDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t* copy_start_event,
                                                 cudaEvent_t* copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    if (copy_start_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_start_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_start_event, getStream(cuda_stream_id)));
    }
    // Data transfer from GPU
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, getStream(cuda_stream_id)));
    if (copy_end_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_end_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_end_event, getStream(cuda_stream_id)));
    }
}

void cudaDeviceInterface::build(const std::string& kernel_filename,
                                const std::vector<std::string>& kernel_names,
                                const std::vector<std::string>& opts) {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvrtcResult res;

    for (auto& kernel_name : kernel_names)
        if (runtime_kernels.count(kernel_name))
            FATAL_ERROR("Building CUDA kernels in file {:s}: kernel \"{:s}\" already exists.",
                        kernel_filename, kernel_name);

    // DEBUG("Building! {:s}", kernel_command)
    //  Load the kernel file contents into `program_buffer`
    fp = fopen(kernel_filename.c_str(), "r");
    if (fp == nullptr) {
        FATAL_ERROR("error loading file: {:s}", kernel_filename.c_str());
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);

    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
    if (sizeRead < (int32_t)program_size)
        FATAL_ERROR("Error reading the file: {:s}", kernel_filename);
    fclose(fp);

    // Create the program object
    nvrtcProgram prog;
    res = nvrtcCreateProgram(&prog, program_buffer, nullptr, 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        INFO("ERROR IN nvrtcCreateProgram: {}", error_str);
    }

    free(program_buffer);

    // Convert compiler options to a c-style array.
    std::vector<const char*> cstrings;
    cstrings.reserve(opts.size());

    for (auto& s : opts)
        cstrings.push_back(s.c_str());

    // Compile the kernel
    res = nvrtcCompileProgram(prog, cstrings.size(), cstrings.data());
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcCompileProgram: {}", error_str);
        // Obtain compilation log from the program.
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        INFO("COMPILE LOG: {}", log);
    }

    // Obtain PTX from the program.
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    res = nvrtcGetPTX(prog, ptx);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcGetPTX: {}", error_str);
    }
    DEBUG2("PTX EXTRACTED");
    res = nvrtcDestroyProgram(&prog);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcDestroyProgram: {}", error_str);
    }

    CUresult err;
    CUmodule module;
    // Get the module with the kernels
    err = cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        FATAL_ERROR("ERROR IN cuModuleLoadDataEx: {}", errStr);
    }

    for (auto& kernel_name : kernel_names) {
        runtime_kernels.emplace(kernel_name, nullptr);
        err = cuModuleGetFunction(&runtime_kernels[kernel_name], module, kernel_name.c_str());
        if (err != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(err, &errStr);
            FATAL_ERROR("ERROR IN cuModuleGetFunction for correlate: {}", errStr);
        }
        if (runtime_kernels[kernel_name] == nullptr) {
            FATAL_ERROR("Failed to find kernel name \"{}\" in compiled PTX module", kernel_name);
        }
    }
}

void cudaDeviceInterface::build_ptx(const std::string& kernel_filename,
                                    const std::vector<std::string>& kernel_names,
                                    const std::vector<std::string>& opts,
                                    const std::string& kernel_name_prefix) {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvPTXCompileResult nv_res;
    CUresult cu_res;
    nvPTXCompilerHandle compiler = nullptr;
    size_t elf_size;
    char* elf;
    CUmodule module;

    for (auto& kernel_name : kernel_names)
        if (runtime_kernels.count(kernel_name_prefix + kernel_name))
            FATAL_ERROR("Building CUDA kernels in file {:s}: kernel \"{:s}\" already exists.",
                        kernel_filename, kernel_name_prefix + kernel_name);
    // DEBUG("Building! {:s}", kernel_command)

    // Load the kernel file contents into `program_buffer`
    fp = fopen(kernel_filename.c_str(), "r");
    if (fp == NULL) {
        FATAL_ERROR("error loading file: {:s}", kernel_filename.c_str());
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);

    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
    if (sizeRead < (int32_t)program_size)
        FATAL_ERROR("Error reading the file: {:s}", kernel_filename);
    fclose(fp);

    // Create the compiler
    nv_res = nvPTXCompilerCreate(&compiler, program_size, program_buffer);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        // TODO Report ENUM names.
        FATAL_ERROR("Could not create PTX compiler, error code: {:d}", (int)nv_res);
        return;
    }

    // Compile for the compute capability of the GPU that is actually present. Callers
    // pass the --gpu-name their kernel was generated for, but SASS is only compatible
    // within a major architecture (an sm_89 cubin fails to load on an sm_86 device with
    // "no kernel image is available for execution on the device"), so kernels would
    // otherwise only run on the specific GPU model they were generated for.
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, gpu_id));
    const std::string gpu_name = fmt::format("--gpu-name=sm_{:d}{:d}", prop.major, prop.minor);
    std::vector<std::string> compile_opts;
    compile_opts.reserve(opts.size() + 1);
    for (const std::string& opt : opts) {
        if (opt.rfind("--gpu-name", 0) == 0) {
            if (opt != gpu_name)
                INFO("Kernel file {:s} was generated for {:s}; compiling for the local GPU with "
                     "{:s} instead",
                     kernel_filename, opt, gpu_name);
        } else {
            compile_opts.push_back(opt);
        }
    }
    compile_opts.push_back(gpu_name);

    // Convert compiler options to a c-style array.
    std::vector<const char*> cstring_opts;
    cstring_opts.reserve(compile_opts.size());
    for (auto& s : compile_opts)
        cstring_opts.push_back(s.c_str());

    // Compile the code
    nv_res = nvPTXCompilerCompile(compiler, cstring_opts.size(), cstring_opts.data());
    // TODO Abstract error checking
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        size_t error_size;
        char* error_log = nullptr;
        nv_res = nvPTXCompilerGetErrorLogSize(compiler, &error_size);
        if (nv_res != NVPTXCOMPILE_SUCCESS) {
            FATAL_ERROR("Could not get error log size, error code: {:d}", (int)nv_res);
            return;
        }
        if (error_size != 0) {
            error_log = (char*)malloc(error_size + 1);
            assert(error_log != nullptr);
            nv_res = nvPTXCompilerGetErrorLog(compiler, error_log);
            if (nv_res != NVPTXCOMPILE_SUCCESS) {
                FATAL_ERROR("Could not get error log, error code: {:d}", (int)nv_res);
                free(error_log);
                return;
            }
        }
        FATAL_ERROR("Could not compile PTX: \n{:s}", error_log);
        free(error_log);
        return;
    }

    nv_res = nvPTXCompilerGetCompiledProgramSize(compiler, &elf_size);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get compiled PTX elf size, error code: {:d}", (int)nv_res);
        return;
    }

    elf = (char*)malloc(elf_size);
    assert(elf != nullptr);
    nv_res = nvPTXCompilerGetCompiledProgram(compiler, (void*)elf);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get compiled PTX elf data, error code: {:d}", (int)nv_res);
        return;
    }

    // Dump Logs
    size_t info_size;
    nv_res = nvPTXCompilerGetInfoLogSize(compiler, &info_size);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get info log size, error code: {:d}", (int)nv_res);
        return;
    }

    if (info_size != 0) {
        char* info_Log = (char*)malloc(info_size + 1);
        nv_res = nvPTXCompilerGetInfoLog(compiler, info_Log);
        if (nv_res != NVPTXCOMPILE_SUCCESS) {
            FATAL_ERROR("Could not get PTX compiler logs, error code: {:d}", (int)nv_res);
            free(info_Log);
            return;
        }
        INFO("PTX Compiler logs: \n{:s}", info_Log);
        free(info_Log);
    }

    // Cleanup compiler
    nv_res = nvPTXCompilerDestroy(&compiler);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not destroy compiler, error code: {:d}", (int)nv_res);
        return;
    }

    // Extract kernels
    cu_res = cuModuleLoadDataEx(&module, elf, 0, nullptr, nullptr);
    if (cu_res != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(cu_res, &errStr);
        FATAL_ERROR("Could not load module data from elf for kernel file {:s}: {:s}",
                    kernel_filename, errStr);
        return;
    }

    for (auto& kernel_name : kernel_names) {
        runtime_kernels.emplace(kernel_name_prefix + kernel_name, nullptr);
        cu_res = cuModuleGetFunction(&runtime_kernels[kernel_name_prefix + kernel_name], module,
                                     kernel_name.c_str());
        if (cu_res != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(cu_res, &errStr);
            FATAL_ERROR("ERROR IN cuModuleGetFunction for correlate: {:s}", errStr);
        }
    }

    free(elf);
}
