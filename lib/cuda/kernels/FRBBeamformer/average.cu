__global__ void FRBBeamformer_average(const int* __restrict__ const E_global,
                                      float* __restrict__ const Esum_global) {
    asm("	ld.param.u64 	%%rd4, [E_global];\n"
        "	ld.param.u64 	%%rd5, [Esum_global];\n");
}
