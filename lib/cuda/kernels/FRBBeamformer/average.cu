__global__ void FRBBeamformer_average(const int* __restrict__ const E_global,
                                      float* __restrict__ const Esum_global) {
    asm("	mov.u64 	%%rd4, %0;\n"
        "	mov.u64 	%%rd5, %1;\n"
        "	mov.u32 	%%r4, %%ctaid.x;\n"
        "	shl.b32 	%%r5, %%r4, 11;\n"
        "	and.b32  	%%r6, %%r5, 522240;\n"
        "	mov.u32 	%%r7, %%tid.y;\n"
        "	shl.b32 	%%r8, %%r7, 8;\n"
        "	and.b32  	%%r9, %%r8, 1792;\n"
        "	mov.u32 	%%r10, %%tid.x;\n"
        "	and.b32  	%%r11, %%r10, 31;\n"
        "	or.b32  	%%r12, %%r9, %%r6;\n"
        "	or.b32  	%%r13, %%r12, %%r11;\n"
        "	mul.wide.u32 	%%rd6, %%r13, 4;\n"
        "	add.s64 	%%rd7, %%rd5, %%rd6;\n"
        "	mov.u32 	%%r26, 0;\n"
        "	st.global.u32 	[%%rd7], %%r26;\n"
        "	st.global.u32 	[%%rd7+128], %%r26;\n"
        "	st.global.u32 	[%%rd7+256], %%r26;\n"
        "	st.global.u32 	[%%rd7+384], %%r26;\n"
        "	st.global.u32 	[%%rd7+512], %%r26;\n"
        "	st.global.u32 	[%%rd7+640], %%r26;\n"
        "	st.global.u32 	[%%rd7+768], %%r26;\n"
        "	st.global.u32 	[%%rd7+896], %%r26;\n"
        //
        ::"l"(E_global),
        "l"(Esum_global));
}
