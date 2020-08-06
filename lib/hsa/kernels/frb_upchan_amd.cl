#define RE    x
#define IM    y
//INVERSE FFT??
#define WN   0.04908738521234052f //-2*pi/128
//#define WN -0.0078125 //-1/128. -> AMD sincos angles normalized to ±1
#define L get_local_id(0)
#define N_TIMES 3 //number of subsequent 128-sample blocks to sum together

#define flip(sel, mask, ra, rb,t) \
        __asm__ ("V_CMP_EQ_U32_E32 vcc %[sel], %[mask] \n" \
                             "V_CNDMASK_B32_E32 %[bi], %[rbi], %[rai], vcc \n" \
                             "V_CNDMASK_B32_E32 %[br], %[rbr], %[rar], vcc \n" \
                             : [bi] "=v" (t.IM), \
                               [br] "=v" (t.RE) \
                             : [sel] "v" (sel), \
                               [mask] "v" (mask), \
                               [rai] "v" (ra.IM), \
                               [rar] "v" (ra.RE), \
                               [rbi] "v" (rb.IM), \
                               [rbr] "v" (rb.RE) \
                             : "vcc"); \

#define flop(sel, mask, ra, rb,t) \
        __asm__ ("V_CMP_EQ_U32_E32 vcc %[sel], %[mask] \n" \
                             "V_CNDMASK_B32_E32 %[rai], %[rai], %[bi], vcc \n" \
                             "V_CNDMASK_B32_E32 %[rar], %[rar], %[br], vcc \n" \
                             "V_CNDMASK_B32_E32 %[rbi], %[bi], %[rbi], vcc \n" \
                             "V_CNDMASK_B32_E32 %[rbr], %[br], %[rbr], vcc " \
                             : [rai] "+&v" (ra.IM), \
                               [rar] "+&v" (ra.RE), \
                               [rbi] "+&v" (rb.IM), \
                               [rbr] "+&v" (rb.RE) \
                             : [sel] "v" (sel), \
                               [mask] "v" (mask), \
                               [bi] "v" (t.IM), \
                               [br] "v" (t.RE) \
                             : "vcc");

#define butterfly(ra, rb, sincos, t) \
        t = ra - rb; \
        ra = ra + rb; \
        rb.IM = t.RE * sincos.IM + t.IM * sincos.RE; \
        rb.RE = t.RE * sincos.RE - t.IM * sincos.IM;

#define twiddle(sincos, W, m) \
        sincos.IM = native_sin(W * m); \
        sincos.RE = native_cos(W * m);

__constant float CBP[16] = { 0.52225748f, 0.58330915f, 0.6868705f, 0.80121821f, 0.89386546f, 0.95477358f, 0.98662733f, 0.99942558f,
                            0.99988676f, 0.98905127f, 0.95874124f, 0.90094667f, 0.81113021f, 0.6999944f, 0.59367968f, 0.52614263f};
__constant float HFB_BP[16] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};


__kernel void frb_upchan_amd (__global float *data, __global float *results_array, __global float *hfb_output_array){
    float2 res[N_TIMES][2];
    float2 sc, b;
    float twiddle_angle;
    uint mask, sel;
    float pow[2]={0,0};

    __local float BP[16];
    if (L<16) BP[L] = CBP[L];

    uint nbeam = get_global_size(1);
    uint nsamp = get_global_size(0) * 6 + 32;
    uint nsamp_out = get_global_size(0) / 64;

    #pragma unroll
    for (uint p=0; p<2; p++) { //sum over two pols
        #pragma unroll
        for (int i=0; i<N_TIMES; i++) {
            uint addr = p * nbeam * nsamp + //polarization, slowest varying
                        get_group_id(1) * nsamp + //beam, separated by nsamp
                        get_group_id(0) * 384 + //output timesteps
                        i * 128 + // FFT'd timesteps
                        L; //idx within FFT
            //top of butterfly
            res[i][0].x = as_float(as_uint(data[addr]) & 0xffff);
            res[i][0].y = as_float(as_uint(data[addr]) >> 16);
            __asm__ ("V_CVT_F32_F16 %0, %1" : "=v"(res[i][0].x) : "v"(res[i][0].x)); //flop back into 32b
            __asm__ ("V_CVT_F32_F16 %0, %1" : "=v"(res[i][0].y) : "v"(res[i][0].y)); //flop back into 32b
            //bottom of butterfly
            res[i][1].x = as_float(as_uint(data[addr + 64]) & 0xffff);
            res[i][1].y = as_float(as_uint(data[addr + 64]) >> 16);
            __asm__ ("V_CVT_F32_F16 %0, %1" : "=v"(res[i][1].x) : "v"(res[i][1].x)); //flop back into 32b
            __asm__ ("V_CVT_F32_F16 %0, %1" : "=v"(res[i][1].y) : "v"(res[i][1].y)); //flop back into 32b
            twiddle(sc,WN,amd_bfe((uint)L,0,6));
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 1 -> bpermute across all 64
        mask = 0x20;
        sel = L & mask;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, mask, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.IM)));
            b.RE = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.RE)));
            twiddle(sc, WN*2, amd_bfe((uint)L,0,5));
            flop(sel, mask, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 2 -> swizzle across 32 WI
        mask = 0x10;
        sel = L&mask;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, mask, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0100000000011111));
            b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0100000000011111));
            twiddle(sc, WN*4, amd_bfe((uint)L,0,4));
            flop(sel, mask, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 3 -> dpp across 16
        mask = 0x08;
        sel = L&mask;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, mask, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0x128, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0x128, 0xf, 0xf, 0));
            twiddle(sc, WN*8, amd_bfe((uint)L,0,3));
            flop(sel, mask, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 4 -> swizzle across 8
        mask = 0x04;
        sel = L&mask;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, mask, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0001000000011111));
            b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0001000000011111));
            twiddle(sc, WN*16, amd_bfe((uint)L,0,2));
            flop(sel, mask, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 5 -> dpp across 4
        mask = 0x02;
        sel = L&mask;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, mask, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b01001110, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b01001110, 0xf, 0xf, 0));
            twiddle(sc, WN*32, amd_bfe((uint)L,0,1));
            flop(sel, mask, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 6 -> dpp across 2
        mask = 0x01;
        sel = L&mask;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++) {
            flip(sel, mask, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b10110001, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b10110001, 0xf, 0xf, 0));
            twiddle(sc, WN*64, 0);
            flop(sel, mask, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //add powers
        pow[0] += res[0][0].RE * res[0][0].RE + res[0][0].IM * res[0][0].IM +
                  res[1][0].RE * res[1][0].RE + res[1][0].IM * res[1][0].IM +
                  res[2][0].RE * res[2][0].RE + res[2][0].IM * res[2][0].IM;
        pow[1] += res[0][1].RE * res[0][1].RE + res[0][1].IM * res[0][1].IM +
                  res[1][1].RE * res[1][1].RE + res[1][1].IM * res[1][1].IM +
                  res[2][1].RE * res[2][1].RE + res[2][1].IM * res[2][1].IM;
    }

    uint l=L*2;
    __asm__ ("V_BFREV_B32 %0, %1" : "=v"(l) : "v"(l)); //32b bit-reverse

    //write HFB
    {
        uint addr = get_group_id(0) * 1024 * 128 + \
                    get_group_id(1) * 128 + \
                    (l>>25); //7b reverse
        // JSW TODO: Bandpass filter to be applied in the post-processing stage
        hfb_output_array[addr   ] = pow[0] / 6.f;// / HFB_BP[(l+8)%16]; //lower 64 freqs
        hfb_output_array[addr+64] = pow[1] / 6.f;// / HFB_BP[(l+8)%16]; //upper 64 freqs
    }

    //sum 8 adjacent => separated by 16, first indexed as irev(0..15)
    pow[0] += as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(pow[0])));
    pow[1] += as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(pow[1])));
    //swap among 32
    pow[0] += as_float(__builtin_amdgcn_ds_swizzle(as_uint(pow[0]), 0b0100000000011111));
    pow[1] += as_float(__builtin_amdgcn_ds_swizzle(as_uint(pow[1]), 0b0100000000011111));
    //swap among 16
    pow[0] += as_float(__builtin_amdgcn_mov_dpp(as_uint(pow[0]), 0x128, 0xf, 0xf, 0));
    pow[1] += as_float(__builtin_amdgcn_mov_dpp(as_uint(pow[1]), 0x128, 0xf, 0xf, 0));

    if (L<16) {
        l = l>>28; //4b reverse
        uint addr = get_group_id(1) * nsamp_out * 16 + \
                    get_group_id(0) * 16 + l;
        //use 16 WIs to output, need to select appropriate pow & offset to write
        uint off = (L&0x8)?8:0; //swap lower & upper 8
        float out = (L&0x8)?pow[0]:pow[1];
        results_array[addr+off] = out / 48.f / BP[l+off];
    }
}
