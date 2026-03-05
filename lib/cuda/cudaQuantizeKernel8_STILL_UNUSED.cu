#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void cpu_quantize8_chunk(const __half* __restrict__ const input, __half* __restrict__ const outputf,
                         uint8_t* __restrict__ const outputi, const int chunk_size) {
    assert(chunk_size >= 0);
    assert(chunk_size % 4 == 0);

    float sum = 0, sum2 = 0;
    for (int i = 0; i < chunk_size; i += 2) {
        const __half2 x = ((const __half2*)input)[i / 2];
        sum += hsum2(x);
        sum2 += hsum2(__hmul2(x, x));
    }
    const float mean = sum / chunk_size;
    const float scale = fmax(1.0f, sqrt(scale / (chunk_size - 1)) / 2.3f);

    const __half2 meanh_scaleh = __floats2half2_rn(mean, scale);
    ((__half2*)outputf)[0] = meanh_scaleh;

    const __half meanh = __low2half(meanh_scaleh);
    const __half scaleh = __high2half(meanh_scaleh);
    const __half offseth = __float2half(128.5f);
    const __half minh = __float2half(1.0f);
    const __half maxh = __float2half(255.0f);
    for (int i = 0; i < chunk_size; i += 4) {
        uint32_t outi = 0;

        const __half2 x01 = ((const __half2*)input)[(i + 0) / 2];
        const __half2 y01 = __hmax2(minh, __hmin2(maxh, (x01 - meanh) / scaleh + offseth));
        const uint32_t i0 = __half2int_rd(__low2half(y01));
        const uint32_t i1 = __half2int_rd(__high2half(y01));
        outi |= (i0 << 0) | (i1 << 8);

        const __half2 x23 = ((const __half2*)input)[(i + 2) / 2];
        const __half2 y23 = __hmax2(minh, __hmin2(maxh, (x23 - meanh) / scaleh + offseth));
        const uint32_t i2 = __half2int_rd(__low2half(y23));
        const uint32_t i3 = __half2int_rd(__high2half(y23));
        outi |= (i2 << 16) | (i3 << 24);

        ((uint32_t*)outputi)[i / 4] = outi;
    }
}

void cpu_quantize8(const __half* __restrict__ const input, __half* __restrict__ const outputf,
                   uint8_t* __restrict__ const outputi, const int input_size1,
                   const int input_size2, const int input_size3, const int input_stride2,
                   const int input_stride3, const int outputf_size1, const int outputf_size2,
                   const int outputf_size3, const int outputf_stride2, const int outputf_stride3,
                   const int outputi_size1, const int outputi_size2, const int outputi_size3,
                   const int outputi_stride2, outputi_stride3, const int chunk_size) {
    assert(input);
    assert(outputf);
    assert(outputi);

    assert(input_size1 >= 0);
    assert(input_size2 >= 0);
    assert(input_size3 >= 0);
    assert(input_stride2 >= 0);
    assert(input_stride3 >= 0);
    assert(outputf_size1 >= 0);
    assert(outputf_size2 >= 0);
    assert(outputf_size3 >= 0);
    assert(outputf_stride2 >= 0);
    assert(outputf_stride3 >= 0);
    assert(outputi_size1 >= 0);
    assert(outputi_size2 >= 0);
    assert(outputi_size3 >= 0);
    assert(outputi_stride2 >= 0);
    assert(outputi_stride3 >= 0);
    assert(chunk_size > 0);

    assert(chunk_size % 4 == 0);
    assert(ninputs1 % chunk_size == 0);

    assert(outputf_size1 == 2 * input_size1 / chunk_size);
    assert(outputf_size2 == input_size2);
    assert(outputf_size3 == input_size3);
    assert(outputi_size1 == input_size1);
    assert(outputi_size2 == input_size2);
    assert(outputi_size3 == input_size3);

    const int nchunks = ninputs1 / chunk_size;

    for (int dim3 = 0; dim3 < ninputs3; ++dim3) {
        for (int dim2 = 0; dim2 < ninputs2; ++dim2) {
            for (int chunk = 0; chunk < nchunks; ++chunk) {
                const int input_dim1 = chunk * chunk_size;
                const int outputf_dim1 = chunk * 2;
                const int outputi_dim1 = chunk * chunk_size;
                const int input_offset = input_dim1 + input_stride2 * dim2 + input_stride3 * dim3;
                const int outputf_offset =
                    outputf_dim1 + outputf_stride2 * dim2 + outputf_stride3 * dim3;
                const int outputi_offset =
                    outputi_dim1 + outputi_stride2 * dim2 + outputi_stride3 * dim3;
                cpu_quantize8_chunk(input + input_offset, outputf + outputf_offset,
                                    outputi + outputi_offset, chunk_size);
            }
        }
    }
}

void launch_quantize8_kernel(cudaStream_t stream, int nframes, const __half2* in_base,
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
