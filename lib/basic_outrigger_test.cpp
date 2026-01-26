#include <CL/cl.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


// temporary wrapper for testing kernel without kotekan, to be removed after later stages of testing

std::string loadKernel(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open kernel file");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    const int NTIME = 100;
    const int NFREQ = 8;
    const int NINPUT = 256;
    const int NPOINTING = 4;
    const int GROUP = 4; // no. work items

    // Test data
    std::vector<unsigned int> inputData(NTIME * NFREQ * NINPUT, 1.0);
    std::vector<float> phaseMap(NPOINTING * NFREQ * NINPUT, 1.0);
    std::vector<float> gpuOut(NTIME * NFREQ * NPOINTING * 2, 0.0); // e.g. X,Y per pointing per time
    std::vector<float> refOut(NTIME * NFREQ * NPOINTING * 2, 0.0); // CPU 

    for (int t = 0; t < NTIME; t++) {
        for (int f = 0; f < NFREQ; f++) {
            for (int i = 0; i < NINPUT; i++) {
                int idx = t * (NFREQ * NINPUT) + f * NINPUT + i;
                inputData[idx] = 1 + idx;  
            }
        }
    }

    for (int p = 0; p < NPOINTING; p++) {
        for (int f = 0; f < NFREQ; f++) {
            for (int i = 0; i < NINPUT; i++) {
                int idx = p * (NFREQ * NINPUT) + f * NINPUT + i;
                phaseMap[idx] = .42; // trivial phases
            }
        }
    }


    for (int p = 0; p < NPOINTING; p++) {
        for (int f = 0; f < NFREQ; f++) {
            for (int t = 0; t < NTIME; t++) {
                float refX = 0, refY = 0;
                for (int i = 0; i < NINPUT; i++) {
                    int inIdx = t * (NFREQ * NINPUT) + f * NINPUT + i;
                    int phIdx = p * (NFREQ * NINPUT) + f * NINPUT + i;
                    float v = inputData[inIdx] * phaseMap[phIdx];
                    if (i % 2 == 0) refX += v;
                    else            refY += v;
                }
                int outIdx = t * (NFREQ * NPOINTING*2) + f * NPOINTING*2 + p*2;
                refOut[outIdx + 0] = refX;
                refOut[outIdx + 1] = refY;
                //std::cout << "test" << std::endl;
                //std::cout << outIdx + 1 << std::endl;
                // if (f == 0 && p == 0) {
                //     std::cout << "test" << "\n";
                //     std::cout << outIdx + 1 << std::endl;
                //     std::cout << refY << std::endl;
                //     std::cout << refOut[outIdx + 1] << std::endl;
                // }
            }
        }
    }
    std::cout << refOut[1] << std::endl;

    // ---- OpenCL setup ----
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    std::string srcStr = loadKernel("/home/shiona/kotekan/lib/opencl/kernels/outrigger_beamformer.cl");

    if (srcStr.empty()) {
        std::cerr << "Kernel file is empty or not found!" << std::endl;
        return 1;
    } else {
        std::cout << "SRCSTR";
        // std::cout << srcStr;
    }

    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::cout << "Num platforms: " << numPlatforms << std::endl;

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    for (cl_uint i = 0; i < numPlatforms; ++i) {
        char name[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, name, nullptr);
        std::cout << "Platform " << i << ": " << name << std::endl;

        cl_uint numDevices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        std::cout << "  Num devices: " << numDevices << std::endl;

        std::vector<cl_device_id> devices(numDevices);
        if (numDevices > 0)
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);

        for (cl_uint j = 0; j < numDevices; ++j) {
            char devName[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, devName, nullptr);
            std::cout << "    Device " << j << ": " << devName << std::endl;
        }
    }


    const char* kernelSrc = srcStr.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSrc, nullptr, nullptr);

    cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize + 1, 0);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "OpenCL Build Log:\n" << log.data() << std::endl;
        std::cerr << "Error code: " << err << std::endl;

        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "gpu_beamforming", nullptr);

    cl_mem inBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(char) * NTIME * NFREQ * NINPUT, inputData.data(), nullptr);
    cl_mem phBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_float2) * NPOINTING * NFREQ * NINPUT, phaseMap.data(), nullptr);
    cl_mem outBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float2) * NTIME * NFREQ * NPOINTING * 2, nullptr, nullptr);

    // inputData, phasemap, outputData
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &phBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outBuf);

    //clSetKernelArg(kernel, 3, sizeof(unsigned int) * NINPUT, nullptr);
    //clSetKernelArg(kernel, 4, sizeof(float) * NINPUT, nullptr);
    
    // outputPartial X/Y
    clSetKernelArg(kernel, 3, sizeof(float) * GROUP, nullptr);
    clSetKernelArg(kernel, 4, sizeof(float) * GROUP, nullptr);

    unsigned int nPointingVal = NPOINTING;
    clSetKernelArg(kernel, 5, sizeof(unsigned int), &nPointingVal);

    // size-3 arrays for opencl dimensions
    // split first dimension into pointing and input. Local automatically splits the first dim, 
    // so that n_groups * size_group = npointing * size_group
    // and get_group_id automatically takes care of this
    size_t global[3] = {NPOINTING*GROUP,NFREQ,NTIME}; //set array of size 3 to have these 3 elements. This will set the dimensions of the groups
    size_t local[3] = {GROUP,1,1};

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
    clFinish(queue);
    // move data from gpu -> cpu
    clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, sizeof(float) * NTIME * NFREQ * NPOINTING * 2, gpuOut.data(), 0, nullptr,
                        nullptr);
    clFinish(queue); 
    for (int sidx=0; sidx<NTIME * NFREQ * NPOINTING * 2; sidx++){
        //int idx = sidx*2;
        //std::cout << "idx:" << idx << "\n";
        //std::cout << "CPU: X=" << refOut[idx] << " Y=" << refOut[idx+1] << "\n";
        //std::cout << "GPU: X=" << gpuOut[idx] << " Y=" << gpuOut[idx+1] << "\n";
        if (std::fabs(refOut[sidx] - gpuOut[sidx])/refOut[sidx] > 1e-5){
            std::cout << "FAILED\n";
            std::cout << "GPU=" << gpuOut[sidx] << " CPU=" << refOut[sidx] << "\n";
        }
    }

    int idx = 2;
    std::cout << "idx:" << idx << "\n";
    std::cout << "CPU: X=" << refOut[idx] << " Y=" << refOut[idx+1] << "\n";
    std::cout << "GPU: X=" << gpuOut[idx] << " Y=" << gpuOut[idx+1] << "\n";

    return 0;
}