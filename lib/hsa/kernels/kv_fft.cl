// llvm-objdump -disassemble -mcpu=fiji ../lib/hsa/kernels/kv_fft.hsaco

#define IM    x
#define RE    y
#define PI2 -1.5707963267948966f
#define PI4 -0.7853981633974483f
#define PI8 -0.39269908169872414f
#define PI16 -0.19634954084936207f
#define PI32 -0.09817477042468103f
#define PI64 -0.04908738521234052f
#define PI128 -0.02454369260617026f
#define PI256 -0.01227184630308513f

#define L get_local_id(0)

#define flip(sel, mask, ra, rb,t) \
        __asm__ __volatile__("V_CMP_EQ_U32 %[sel], %[mask] \n" \
                             "V_CNDMASK_B32 %[bi], %[rbi], %[rai] \n" \
                             "V_CNDMASK_B32 %[br], %[rbr], %[rar] \n" \
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
        __asm__ __volatile__("V_CMP_EQ_U32 %[sel], %[mask] \n" \
                             "V_CNDMASK_B32 %[rai], %[raii], %[bi] \n" \
                             "V_CNDMASK_B32 %[rar], %[rari], %[br] \n" \
                             "V_CNDMASK_B32 %[rbi], %[bi], %[rbii] \n" \
                             "V_CNDMASK_B32 %[rbr], %[br], %[rbri] " \
                             : [rai] "=&v" (ra.IM), \
                               [rar] "=&v" (ra.RE), \
                               [rbi] "=&v" (rb.IM), \
                               [rbr] "=&v" (rb.RE) \
                             : [raii] "0" (ra.IM), \
                               [rari] "1" (ra.RE), \
                               [rbii] "2" (rb.IM), \
                               [rbri] "3" (rb.RE), \
                               [sel] "v" (sel), \
                               [mask] "v" (mask), \
                               [bi] "v" (t.IM), \
                               [br] "v" (t.RE) \
                             : "vcc");

#define butterfly(ra, rb, sincos, t) \
        t = ra - rb; \
        ra = ra + rb; \
        rb.IM = t.RE * sincos.IM + t.IM * sincos.RE; \
        rb.RE = t.RE * sincos.RE - t.IM * sincos.IM;

#define twiddle(sincos, W, m,idx) \
        sincos.IM = native_sin(W * ((L&m)*4+idx)); \
        sincos.RE = native_cos(W * ((L&m)*4+idx));


__kernel void kv_fft (__global uint *inputData,
                      __global float2 *outputData){
    float2 res[4][8];
    float2 sc, b;
    float twiddle_angle;
    uint mask, sel;

    uint data_temp[4];

    #pragma unroll
    for (int ew=0; ew<4; ew++){
        data_temp[ew] = inputData[L +                      //offset within 256 NS feeds
                                  ew * 512/4 +             //cylinder
                                  get_group_id(0) * 256 +  //EW vs NS pol
                                  get_group_id(1) * 2048/4 //timesteps
                                 ];
    }
    #pragma unroll
    for (int i=0; i<4; i++) {
        #pragma unroll
        for (int ew=0; ew<4; ew++){
            res[ew][2*i  ].IM = ((int)amd_bfe(data_temp[ew],i*8+0,4))-8;
            res[ew][2*i  ].RE = ((int)amd_bfe(data_temp[ew],i*8+4,4))-8;
            twiddle(sc,PI256,0xffffffff,i);
            res[ew][2*i+1].IM = res[ew][2*i].RE * sc.IM + res[ew][2*i].IM * sc.RE;
            res[ew][2*i+1].RE = res[ew][2*i].RE * sc.RE - res[ew][2*i].IM * sc.IM;
        }
    }

    #pragma unroll
    for (int ew=0; ew<4; ew++){

    //shuffle 1 -> bpermute across all 64
    mask = 0x20;
    sel = L & mask;
    #pragma unroll
    for (int i=0; i<4; i++){
        flip(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        b.IM = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.IM)));
        b.RE = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.RE)));
        twiddle(sc, PI128, 0x1f, i);
        flop(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
    }

   //shuffle 2 -> swizzle across 32 WI
    mask = 0x10;
    sel = L&mask;
    #pragma unroll
    for (int i=0; i<4; i++){
        flip(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0100000000011111));
        b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0100000000011111));
        twiddle(sc, PI64, 0xf, i);
        flop(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
    }

    //shuffle 3 -> dpp across 16
    mask = 0x08;
    sel = L&mask;
    #pragma unroll
    for (int i=0; i<4; i++){
        flip(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0x128, 0xf, 0xf, 0));
        b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0x128, 0xf, 0xf, 0));
        twiddle(sc, PI32, 0x7, i);
        flop(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
    }

    //shuffle 4 -> swizzle across 8
    mask = 0x04;
    sel = L&mask;
    #pragma unroll
    for (int i=0; i<4; i++){
        flip(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0001000000011111));
        b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0001000000011111));
        twiddle(sc, PI16, 0x3, i);
        flop(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
    }

    //shuffle 5 -> dpp across 4
    mask = 0x02;
    sel = L&mask;
    #pragma unroll
    for (int i=0; i<4; i++){
        flip(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b01001110, 0xf, 0xf, 0));
        b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b01001110, 0xf, 0xf, 0));
        twiddle(sc, PI8, 0x1, i);
        flop(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
    }

    //shuffle 6 -> dpp across 2
    mask = 0x01;
    sel = L&mask;
    #pragma unroll
    for (int i=0; i<4; i++) {
        flip(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b10110001, 0xf, 0xf, 0));
        b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b10110001, 0xf, 0xf, 0));
        twiddle(sc, PI4, 0, i);
        flop(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
        butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
    }

    //shuffle 7 -> swap internally across 4 pairs
    {
        float2 a[2] = {res[ew][1], res[ew][3]};

        res[ew][1] = res[ew][0] - res[ew][4];
        res[ew][0] = res[ew][0] + res[ew][4];

        res[ew][3].IM = res[ew][6].RE - res[ew][2].RE;
        res[ew][3].RE = res[ew][2].IM - res[ew][6].IM;
        res[ew][2] = res[ew][2] + res[ew][6];

        res[ew][4] = a[0] + res[ew][5];
        res[ew][5] = a[0] - res[ew][5];

        a[0] = a[1] - res[ew][7];
        res[ew][6] = a[1] + res[ew][7];
        res[ew][7].IM = -a[0].RE;
        res[ew][7].RE = a[0].IM;
    }
    //shuffle 8 -> swap internally across 2 pairs
    {
        float2 a[2] = {res[ew][1], res[ew][5]};

        res[ew][1] = res[ew][0] - res[ew][2];
        res[ew][0] = res[ew][0] + res[ew][2];
        res[ew][2] = a[0] + res[ew][3];
        res[ew][3] = a[0] - res[ew][3];
        res[ew][5] = res[ew][4] - res[ew][6];
        res[ew][4] = res[ew][4] + res[ew][6];
        res[ew][6] = a[1] + res[ew][7];
        res[ew][7] = a[1] - res[ew][7];
    }
    }

    //output!
    #pragma unroll
    for (int ew=0; ew<4; ew++) {
        #pragma unroll
        for (int i=0; i<8; i++) {
            uint irev = (L*8+i);
            __asm__ __volatile__("V_BFREV_B32 %0, %1" : "=v"(irev) : "v"(irev)); //32b bit-reverse
            outputData[get_group_id(0) * 512 +
                       ew * 1024 +
                       get_group_id(1) * 1024 * 4 +
                       (irev>>23)] = res[ew][i];
        }
    }
}
