#define RE    x
#define IM    y
//INVERSE FFT: Freq decreases w/ index
#define WN 2.f * M_PI_F / 128.f
#define L get_local_id(1)
#define N_TIMES 3 //number of subsequent 128-sample blocks to sum together

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define T get_group_id(1)
#define B get_group_id(0)

#define flip(sel, ra, rb,t) \
        t.IM = select(rb.IM,ra.IM,sel); \
        t.RE = select(rb.RE,ra.RE,sel);

#define flop(sel, ra, rb,t) \
        ra.IM = select(ra.IM,t.IM,sel); \
        ra.RE = select(ra.RE,t.RE,sel); \
        rb.IM = select(t.IM,rb.IM,sel); \
        rb.RE = select(t.RE,rb.RE,sel);

#define butterfly(ra, rb, sincos, t) \
        t = ra - rb; \
        ra = ra + rb; \
        rb.IM = t.RE * sincos.IM + t.IM * sincos.RE; \
        rb.RE = t.RE * sincos.RE - t.IM * sincos.IM;

#define twiddle(sincos, W, m) \
        sincos.IM = native_sin(W * m); \
        sincos.RE = native_cos(W * m);

__constant float CBP[16] = { 0.52225748f, 0.58330915f, 0.6868705f,  0.80121821f,
                             0.89386546f, 0.95477358f, 0.98662733f, 0.99942558f,
                             0.99988676f, 0.98905127f, 0.95874124f, 0.90094667f,
                             0.81113021f, 0.6999944f,  0.59367968f, 0.52614263f};
__constant float HFB_BP[16] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                                1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};


__kernel void frb_upchan_amd (__global half2 *data, __global float *results_array, __global float *hfb_output_array){
    float2 res[N_TIMES][2];
    float2 sc, b;
    float twiddle_angle;
    uint sel;
    float pow[2]={0,0};

    __local float BP[16];
    if (L<16) BP[L] = CBP[L];

    uint nbeam = get_global_size(0);
    uint nsamp = get_global_size(1) * 6 + 64;
    uint nsamp_out = get_global_size(1) / 64;

    #pragma unroll
    for (uint p=0; p<2; p++) { //sum over two pols
        #pragma unroll
        for (int i=0; i<N_TIMES; i++) {
            uint addr = p * nbeam * nsamp + //polarization, slowest varying
                        B * nsamp + //beam, separated by nsamp
                        T * 384 + //output timesteps
                        i * 128 + // FFT'd timesteps
                        L; //idx within FFT
            res[i][0] = (float2){data[addr].RE, data[addr].IM};
            res[i][1] = (float2){data[addr+64].RE, data[addr+64].IM};
            twiddle(sc,WN,amd_bfe((uint)L,0,6));
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 1 -> bpermute across all 64
        sel = L & 0x20;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.IM)));
            b.RE = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.RE)));
            twiddle(sc, WN*2, amd_bfe((uint)L,0,5));
            flop(sel, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 2 -> swizzle across 32 WI
        sel = L & 0x10;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0100000000011111));
            b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0100000000011111));
            twiddle(sc, WN*4, amd_bfe((uint)L,0,4));
            flop(sel, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 3 -> dpp across 16
        sel = L & 0x08;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0x128, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0x128, 0xf, 0xf, 0));
            twiddle(sc, WN*8, amd_bfe((uint)L,0,3));
            flop(sel, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 4 -> swizzle across 8
        sel = L & 0x04;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0001000000011111));
            b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0001000000011111));
            twiddle(sc, WN*16, amd_bfe((uint)L,0,2));
            flop(sel, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 5 -> dpp across 4
        sel = L & 0x02;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++){
            flip(sel, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b01001110, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b01001110, 0xf, 0xf, 0));
            twiddle(sc, WN*32, amd_bfe((uint)L,0,1));
            flop(sel, res[i][0], res[i][1],b);
            butterfly(res[i][0], res[i][1], sc, b);
        }

        //shuffle 6 -> dpp across 2
        sel = L & 0x01;
        #pragma unroll
        for (int i=0; i<N_TIMES; i++) {
            flip(sel, res[i][0], res[i][1], b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b10110001, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b10110001, 0xf, 0xf, 0));
            twiddle(sc, WN*64, 0);
            flop(sel, res[i][0], res[i][1],b);
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
        uint addr = T * 1024 * 128 + \
                    B * 128 + \
                    (l>>25); //7b reverse
        // JSW TODO: Bandpass filter to be applied in the post-processing stage
        hfb_output_array[addr+64] = pow[0] / 6.f;// / HFB_BP[(l+8)%16]; //lower 64 freqs
        hfb_output_array[addr   ] = pow[1] / 6.f;// / HFB_BP[(l+8)%16]; //upper 64 freqs
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
        uint addr = B * nsamp_out * 16 + \
                    T * 16 + l;
        //use 16 WIs to output, need to select appropriate pow & offset to write
        sel = L&0x8;
        uint off = select(0u,8u,sel);
        float out = select(pow[1],pow[0],sel) / 48.f / BP[l+off];
        results_array[addr+off] = out;
    }
}