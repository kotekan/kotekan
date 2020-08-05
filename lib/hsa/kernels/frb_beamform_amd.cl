// llvm-objdump -disassemble -mcpu=fiji ../lib/hsa/kernels/kv_fft.hsaco

// Optimization notes:
// - beamforming needs to be written out long, not looped [HUGE, 6% Fiji]
// - map does better as a global load [0.1% Fiji]
// - co does better with a shared load [0.4%% Fiji]

#define RE    x
#define IM    y
#define WN -0.01227184630308513f //-2*pi/512
//#define WN -0.001953125f //-1/512. -> AMD sincos angles normalized to ±1
#define L get_local_id(0)

#define flip(sel, mask, ra, rb,t) \
        __asm__ ("V_CMP_EQ_U32 %[sel], %[mask] \n" \
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
        __asm__ ("V_CMP_EQ_U32 %[sel], %[mask] \n" \
                             "V_CNDMASK_B32 %[rai], %[rai], %[bi] \n" \
                             "V_CNDMASK_B32 %[rar], %[rar], %[br] \n" \
                             "V_CNDMASK_B32 %[rbi], %[bi], %[rbi] \n" \
                             "V_CNDMASK_B32 %[rbr], %[br], %[rbr] " \
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

#define twiddle(sincos, W, m,idx) \
        sincos.IM = native_sin(W * (m*4+idx)); \
        sincos.RE = native_cos(W * (m*4+idx));
//        twiddle_angle = W*(m*4+idx); \
//        __asm__ ("V_SIN_F32 %0, %1" : "=v"(sincos.IM) : "v"(twiddle_angle)); \
//        __asm__ ("V_COS_F32 %0, %1" : "=v"(sincos.RE) : "v"(twiddle_angle));



__kernel void frb_beamform_amd (__global uint *inputData, __global uint *map, global float *co, __global float *outputData, __constant float2 *gains){
    float2 res[4][8];
    float2 sc, b;
    float twiddle_angle;
    uint mask, sel;

    //pre-load the bf coeffs into local share
    __local float lcof[32];
    if (L<32) lcof[L] = co[L];
    float2 *lco = (float2*)lcof;

    uint data_temp[4];
    #pragma unroll
    for (int ew=0; ew<4; ew++){
        data_temp[ew] = inputData[L +                       //offset within 256 NS feeds
                                  ew * 256 /4 +              //cylinder,pol
                                  get_group_id(1) * 1024/4 + //EW vs NS pol
                                  get_group_id(2) * 2048/4   //timesteps
                                 ];
        #pragma unroll
        for (int i=0; i<4; i++) {
            float2 t;
            float2 gain = gains[get_global_id(1)*1024 + ew*256 + L*4 + i];
            t.IM = ((float)amd_bfe(data_temp[ew],i*8+0,4))-8;
            t.RE = ((float)amd_bfe(data_temp[ew],i*8+4,4))-8;
            //gains are conjugated?
            res[ew][2*i  ].RE = t.RE * gain.RE + t.IM * gain.IM;
            res[ew][2*i  ].IM = t.IM * gain.RE - t.RE * gain.IM;
            twiddle(sc,WN, amd_bfe((uint)L,0,6),i);
            res[ew][2*i+1].IM = res[ew][2*i].RE * sc.IM + res[ew][2*i].IM * sc.RE;
            res[ew][2*i+1].RE = res[ew][2*i].RE * sc.RE - res[ew][2*i].IM * sc.IM;
        }

        //shuffle 1 -> bpermute across all 64
        mask = 0x20;
        sel = L & mask;
        #pragma unroll
        for (int i=0; i<4; i++){
            flip(sel, mask, res[ew][2*i], res[ew][2*i+1], b);
            b.IM = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.IM)));
            b.RE = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.RE)));
            twiddle(sc, WN*2, amd_bfe((uint)L,0,5), i);
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
            twiddle(sc, WN*4, amd_bfe((uint)L,0,4), i);
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
            twiddle(sc, WN*8, amd_bfe((uint)L,0,3), i);
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
            twiddle(sc, WN*16, amd_bfe((uint)L,0,2), i);
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
            twiddle(sc, WN*32, amd_bfe((uint)L,0,1), i);
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
            twiddle(sc, WN*64, 0, i);
            flop(sel, mask, res[ew][2*i], res[ew][2*i+1],b);
            butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
        }

        //shuffle 7 -> swap internally across 4 pairs
        {
            float2 a[2] = {res[ew][1], res[ew][3]}; //replace sc,b

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
            float2 a[2] = {res[ew][1], res[ew][5]}; //replace sc,b

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

    //choose the right NS beams via local share, one cyl at a time
    __local float2 transfer[512];
    #pragma unroll
    for (int ew=0; ew<4; ew++) { //cylinder
        //de-mux cyl into local share
        #pragma unroll
        for (int i=0; i<8; i++){
            uint irev = L*8+i;
            __asm__ ("V_BFREV_B32 %0, %1" : "=v"(irev) : "v"(irev)); //32b bit-reverse
            transfer[irev>>23] = res[ew][i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //fetch the right ns beams for cyl
        #pragma unroll
        for (int ns=0; ns<4; ns++){
            uint addr = map[L*4+ns];
            res[ew][ns] = transfer[addr];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //form beams
    float2 beam[4][4];
    #pragma unroll
    for (int bidx=0; bidx<4; bidx++){ //4 beams EW
        #pragma unroll
        for (int ns=0; ns<4; ns++){
            beam[ns][bidx].RE = (res[0][ns].RE*lco[bidx*4+0].RE + res[0][ns].IM*lco[bidx*4+0].IM +
                                 res[1][ns].RE*lco[bidx*4+1].RE + res[1][ns].IM*lco[bidx*4+1].IM +
                                 res[2][ns].RE*lco[bidx*4+2].RE + res[2][ns].IM*lco[bidx*4+2].IM +
                                 res[3][ns].RE*lco[bidx*4+3].RE + res[3][ns].IM*lco[bidx*4+3].IM)/4.;
            beam[ns][bidx].IM = (res[0][ns].RE*lco[bidx*4+0].IM - res[0][ns].IM*lco[bidx*4+0].RE +
                                 res[1][ns].RE*lco[bidx*4+1].IM - res[1][ns].IM*lco[bidx*4+1].RE +
                                 res[2][ns].RE*lco[bidx*4+2].IM - res[2][ns].IM*lco[bidx*4+2].RE +
                                 res[3][ns].RE*lco[bidx*4+3].IM - res[3][ns].IM*lco[bidx*4+3].RE)/4.;
            float out;
            __asm__ ("V_CVT_PKRTZ_F16_F32 %0, %1, %2" : "=v"(out): "v"(beam[ns][bidx].RE), "v"(beam[ns][bidx].IM));
            uint addr = get_group_id(2) * 2048 + //time
                        get_group_id(1) * 1024 + //pol
                        bidx * 256 +             //EW
                        (255-(L*4+ns));          //position
            outputData[addr] = out;
        }
    }
}
