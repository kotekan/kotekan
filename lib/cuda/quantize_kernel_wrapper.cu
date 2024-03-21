#include "quantize_kernel.cu"

void launch_quantize_kernel(cudaStream_t stream, int nframes, const __half2* in_base,
                            __half2* outf_base, unsigned int* outi_base, const int* index_array) {
    dim3 nblocks;
    nblocks.x = nframes;
    nblocks.y = 1;
    nblocks.z = 1;
    int nthreads = 32;
    int shmem_nbytes = 0;
    quantize<<<nblocks, nthreads, shmem_nbytes, stream>>>(in_base, outf_base, outi_base,
                                                          index_array);
}
