#include "cudaUpchannelize16k.hpp"

#include "cudaUtils.hpp"
#include "math.h"

#include "fmt.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

const int sizeof_float16_t = 2;

REGISTER_CUDA_COMMAND(cudaUpchannelize16k);

cudaUpchannelize16k::cudaUpchannelize16k(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers, cudaDeviceInterface& device,
                                         int inst) :
    cudaUpchannelize(config, unique_name, host_buffers, device, inst, "upchannelize16",
                     "upchan-U16.ptx", 16384,
                     "_Z17julia_upchan_404813CuDeviceArrayI9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_"
                     "IS1_Li1ELi1EES_I5Int32Li1ELi1EE") {}

cudaUpchannelize16k::~cudaUpchannelize16k() {}
