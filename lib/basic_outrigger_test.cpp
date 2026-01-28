#include <CL/cl.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>

static inline float truncate_uint8(float val)
{
    val = (val > 15) ? 15 : val;
    val = (val < 0) ? 0 : val;
    return val; 
}

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
    const int NTIME = 1; // 100;
    const int NFREQ = 8; //8;
    const int NINPUT = 256;
    const int NPOINTING = 1; //4;
    const int GROUP = 64; // no. work items

    // Random number generator
    std::random_device rd;                        
    std::mt19937 gen(rd());                      
    std::uniform_int_distribution<int> dist(0, 255);  // range for uchar
    std::uniform_real_distribution<float> dist_phase(0.0f, 1.0f); // floats in [0,1)

    // Test data
    std::vector<unsigned char> inputData(NTIME * NFREQ * NINPUT, 0b00010001);
    std::vector<float> phaseMap(NPOINTING * NFREQ * NINPUT*2, 1.0);
    std::vector<unsigned char> gpuOut(NTIME * NFREQ * NPOINTING * 2, 0.0); // e.g. X,Y per pointing per time
    std::vector<unsigned char> refOut(NTIME * NFREQ * NPOINTING * 2, 0.0); // CPU 
    float scaling = 256; 
    int POLBLOCK = 16; 
    for (int t = 0; t < NTIME; t++) {
        for (int f = 0; f < NFREQ; f++) {
            for (int i = 0; i < NINPUT; i++) {
                int idx = t * (NFREQ * NINPUT) + f * NINPUT + i;
                inputData[idx] =  static_cast<unsigned char>(dist(gen)); //0b01010111; // (idx)%NINPUT;  
            }
        }
    }

    for (int p = 0; p < NPOINTING; p++) {
        for (int f = 0; f < NFREQ; f++) {
            for (int i = 0; i < NINPUT; i++) {
                int idx = p * (NFREQ * NINPUT*2) + f * NINPUT*2 + i*2;
                phaseMap[idx] = dist_phase(gen); //1.0; // .42; // trivial phases
                phaseMap[idx+1] = dist_phase(gen); //.5;// .8; // trivial phases
            }
        }
    }


    for (int p = 0; p < NPOINTING; p++) {
        for (int f = 0; f < NFREQ; f++) {
            for (int t = 0; t < NTIME; t++) {
                float refX_re = 0, refX_im = 0,refY_re = 0, refY_im = 0;
                for (int i = 0; i < NINPUT; i++) {
                    int inIdx = t * (NFREQ * NINPUT) + f * NINPUT + i;
                    int phIdx = p * (NFREQ * 2*NINPUT) + f * 2*NINPUT + i*2;
                    float ph_RE = phaseMap[phIdx];
                    float ph_IM = phaseMap[phIdx+1];
                    unsigned char data_temp = inputData[inIdx];
                    //float v = inputData[inIdx] * phaseMap[phIdx];
                    if ((i / POLBLOCK ) % 2 == 0) {
                        refX_re +=  ph_RE*((float)((data_temp & 0xf0)>> 4u)-8)
                                  - ph_IM*((float)((data_temp & 0x0f)>> 0u)-8);     
                        refX_im += ph_IM*((float)((data_temp & 0xf0)>> 4u)-8)
                                 + ph_RE*((float)((data_temp & 0x0f)>> 0u)-8);  
                    }
                    else{
                        refY_re += ph_RE*((float)((data_temp & 0xf0)>> 4u)-8)
                                  -ph_IM*((float)((data_temp & 0x0f)>> 0u)-8);   
                        
                        refY_im += ph_IM*((float)((data_temp & 0xf0)>> 4u)-8)
                                  +ph_RE*((float)((data_temp & 0x0f)>> 0u)-8); 
                    }
                }
                refX_re = refX_re / scaling + 8; 
                refX_im = refX_im / scaling + 8; 
                refX_re = truncate_uint8(refX_re);
                refX_im = truncate_uint8(refX_im);
                int out_idx = f * (NTIME * NPOINTING*2) + t * NPOINTING*2 + p*2;
                refOut[out_idx] = (((int)refX_re << 4) & 0xF0) + ((int)refX_im& 0x0F); 

                refY_re = refY_re / scaling + 8; 
                refY_im = refY_im / scaling + 8; 
                refY_re = truncate_uint8(refY_re);
                refY_im = truncate_uint8(refY_im);
                refOut[out_idx + 1] = (((int)refY_re << 4) & 0xF0) + ((int)refY_im& 0x0F);

            }
        }
    }

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

    //  std::vector<unsigned char> inputData(NTIME * NFREQ * NINPUT, 0b00010001);
    // std::vector<float> phaseMap(NPOINTING * NFREQ * NINPUT*2, 1.0);
    // std::vector<unsigned char> gpuOut(NTIME * NFREQ * NPOINTING * 2, 0.0); // e.g. X,Y per pointing per time
    // std::vector<unsigned char> refOut(NTIME * NFREQ * NPOINTING * 2, 0.0); // CPU 
    cl_mem inBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(u_char) * NTIME * NFREQ * NINPUT, inputData.data(), nullptr);
    cl_mem phBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_float2) * NPOINTING * NFREQ * NINPUT, phaseMap.data(), nullptr);
    cl_mem outBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(u_char) * NTIME * NFREQ * NPOINTING * 2, nullptr, nullptr);

    // inputData, phasemap, outputData
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &phBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outBuf);

    //clSetKernelArg(kernel, 3, sizeof(unsigned int) * NINPUT, nullptr);
    //clSetKernelArg(kernel, 4, sizeof(float) * NINPUT, nullptr);
    
    // outputPartial X/Y
    //clSetKernelArg(kernel, 3, sizeof(float) * GROUP, nullptr);
    //clSetKernelArg(kernel, 4, sizeof(float) * GROUP, nullptr);

    clSetKernelArg(kernel, 3, sizeof(float), &scaling);

    unsigned int nPointingVal = NPOINTING;
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &nPointingVal);

    // size-3 arrays for opencl dimensions
    // split first dimension into pointing and input. Local automatically splits the first dim, 
    // so that n_groups * size_group = npointing * size_group
    // and get_group_id automatically takes care of this
    size_t global[3] = {NPOINTING*GROUP,NFREQ,NTIME}; //set array of size 3 to have these 3 elements. This will set the dimensions of the groups
    size_t local[3] = {GROUP,1,1};

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
    clFinish(queue);
    // move data from gpu -> cpu
    clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, sizeof(u_char) * NTIME * NFREQ * NPOINTING * 2, gpuOut.data(), 0, nullptr,
                        nullptr);
    clFinish(queue); 
    for (int sidx=0; sidx<NTIME * NFREQ * NPOINTING; sidx++){
        //int idx = sidx*2;
        //std::cout << "idx:" << idx << "\n";
        //std::cout << "CPU: X=" << refOut[idx] << " Y=" << refOut[idx+1] << "\n";
        //std::cout << "GPU: X=" << gpuOut[idx] << " Y=" << gpuOut[idx+1] << "\n";
        // if (std::fabs(refOut[sidx] - gpuOut[sidx])/refOut[sidx] > 1){
        int idx = sidx * 2;

        u_char refOutVal = refOut[idx];
        int refOutRe = ((refOutVal & 0xf0)>> 4u)-8;
        int refOutIm = ((refOutVal & 0x0f)>> 0u)-8;
        u_char gpuOutVal = gpuOut[idx];
        int gpuOutRe = ((gpuOutVal & 0xf0)>> 4u)-8;
        int gpuOutIm = ((gpuOutVal & 0x0f)>> 0u)-8;
        
        std::cout << "X Re GPU=" << gpuOutRe << " CPU=" << refOutRe << std::endl;
        std::cout << "X Im GPU=" << gpuOutIm << " CPU=" << refOutIm << std::endl;
        if (refOut[idx] != gpuOut[idx]) {
            std::cout << "FAILED";
        }
        std::cout << "\n";

        idx = idx+1;

        refOutVal = refOut[idx];
        refOutRe = ((refOutVal & 0xf0)>> 4u)-8;
        refOutIm = ((refOutVal & 0x0f)>> 0u)-8;
        gpuOutVal = gpuOut[idx];
        gpuOutRe = ((gpuOutVal & 0xf0)>> 4u)-8;
        gpuOutIm = ((gpuOutVal & 0x0f)>> 0u)-8;
        
        std::cout << "Y Re GPU=" << gpuOutRe << " CPU=" << refOutRe << std::endl;
        std::cout << "Y Im GPU=" << gpuOutIm << " CPU=" << refOutIm << std::endl;

        if (refOut[idx] != gpuOut[idx]) {
            std::cout << "FAILED";
        }
        std::cout << "\n";
    }

    int idx = 0;
    std::cout << "idx:" << idx << "\n";
    std::cout << "CPU: X=" << (int)refOut[idx] << " Y=" << (int)refOut[idx+1] << "\n";
    std::cout << "GPU: X=" << (int)gpuOut[idx] << " Y=" << (int)gpuOut[idx+1] << "\n";

    return 0;
}