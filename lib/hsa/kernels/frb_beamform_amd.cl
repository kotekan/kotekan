// llvm-objdump -disassemble -mcpu=fiji ../lib/hsa/kernels/kv_fft.hsaco

// Optimization notes:
// - beamforming needs to be written out long, not looped [HUGE, 6% Fiji]
// - map does better as a global load [0.1% Fiji]
// - co does better with a shared load [0.4%% Fiji]

#define RE    x
#define IM    y
//#define WN -0.01227184630308513f //-2*pi/512
#define WN -2.f * M_PI_F / 512.f
//#define WN -0.001953125f //-1/512. -> AMD sincos angles normalized to ±1
#define L get_local_id(0) //FFT position
#define P get_group_id(1) //pol
#define T get_group_id(2) //time

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define flip(sel, ra, rb,t) \
        t.IM = select(rb.IM,ra.IM,sel); \
        t.RE = select(rb.RE,ra.RE,sel);

#define flop(sel, ra, rb,t) \
        __asm__ ("V_CMP_NE_U32_E32 vcc 0, %[sel] \n" \
                 "V_CNDMASK_B32_E32 %[rai], %[rai], %[bi] vcc \n" \
                 "V_CNDMASK_B32_E32 %[rar], %[rar], %[br] vcc \n" \
                 "V_CNDMASK_B32_E32 %[rbi], %[bi], %[rbi] vcc \n" \
                 "V_CNDMASK_B32_E32 %[rbr], %[br], %[rbr] vcc " \
                 : [rai] "+&v" (ra.IM), \
                   [rar] "+&v" (ra.RE), \
                   [rbi] "+&v" (rb.IM), \
                   [rbr] "+&v" (rb.RE) \
                 : [sel] "v" (sel), \
                   [mask] "v" (mask), \
                   [bi] "v" (t.IM), \
                   [br] "v" (t.RE) \
                 : "vcc");
/* For some reason, the compiler wants the unused "mask" variable above.
   Dropping it adds 40 vregs (!!) so we're stuck with the above assembly :(
        ra.IM = select(ra.IM,t.IM,sel); \
        ra.RE = select(ra.RE,t.RE,sel); \
        rb.IM = select(t.IM,rb.IM,sel); \
        rb.RE = select(t.RE,rb.RE,sel);
*/
#define butterfly(ra, rb, sincos, t) \
        t = ra - rb; \
        ra = ra + rb; \
        rb.IM = t.RE * sincos.IM + t.IM * sincos.RE; \
        rb.RE = t.RE * sincos.RE - t.IM * sincos.IM;

#define twiddle(sincos, W, m,idx) \
        sincos.IM = native_sin(W * (m*4+idx)); \
        sincos.RE = native_cos(W * (m*4+idx));

__kernel
//__attribute__((amdgpu_num_vgpr(80)))
//__attribute__((amdgpu_waves_per_eu(1,21)))
//__attribute__((reqd_work_group_size(64, 1, 1)))
void frb_beamform_amd (__global uint *inputData, __global uint *map, global float *co, __global half2 *outputData, __constant float2 *gains){
    float2 res[4][8];
    float2 sc, b;
    float twiddle_angle;
    uint mask=0x1, sel;

    //pre-load the bf coeffs into local share
    __local float lcof[32];
    if (L<32) lcof[L] = co[L];
    float2 *lco = (float2*)lcof;

    uint data_temp[4];
    #pragma unroll
    for (int ew=0; ew<4; ew++){
        data_temp[ew] = inputData[L +                       //offset within 256 NS feeds
                                  ew * 256/4 +              //cylinder,pol
                                  P * 1024/4 + //EW vs NS pol
                                  T * 2048/4   //timesteps
                                 ];
        #pragma unroll
        for (int i=0; i<4; i++) {
            float2 t;
            float2 gain = gains[P*1024 + ew*256 + L*4 + i];
            t.IM = ((float)amd_bfe(data_temp[ew],i*8+0,4))-8.f;
            t.RE = ((float)amd_bfe(data_temp[ew],i*8+4,4))-8.f;
            //gains are conjugated?
            res[ew][2*i  ].RE = t.RE * gain.RE + t.IM * gain.IM;
            res[ew][2*i  ].IM = t.IM * gain.RE - t.RE * gain.IM;
            twiddle(sc,WN, amd_bfe((uint)L,0,6),i);
            res[ew][2*i+1].IM = res[ew][2*i].RE * sc.IM + res[ew][2*i].IM * sc.RE;
            res[ew][2*i+1].RE = res[ew][2*i].RE * sc.RE - res[ew][2*i].IM * sc.IM;
        }

        //shuffle 1 -> bpermute across all 64
        sel = L & 0x20;
        #pragma unroll
        for (int i=0; i<4; i++){
            flip(sel, res[ew][2*i], res[ew][2*i+1], b);
            b.IM = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.IM)));
            b.RE = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.RE)));
            twiddle(sc, WN*2, amd_bfe((uint)L,0,5), i);
            flop(sel, res[ew][2*i], res[ew][2*i+1],b);
            butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
        }

        //shuffle 2 -> swizzle across 32 WI
        sel = L & 0x10;
        #pragma unroll
        for (int i=0; i<4; i++){
            flip(sel, res[ew][2*i], res[ew][2*i+1],b);
            b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0100000000011111));
            b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0100000000011111));
            twiddle(sc, WN*4, amd_bfe((uint)L,0,4), i);
            flop(sel, res[ew][2*i], res[ew][2*i+1],b);
            butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
        }

        //shuffle 3 -> dpp across 16
        sel = L & 0x8;
        #pragma unroll
        for (int i=0; i<4; i++){
            flip(sel, res[ew][2*i], res[ew][2*i+1],b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0x128, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0x128, 0xf, 0xf, 0));
            twiddle(sc, WN*8, amd_bfe((uint)L,0,3), i);
            flop(sel, res[ew][2*i], res[ew][2*i+1],b);
            butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
        }

        //shuffle 4 -> swizzle across 8
        sel = L & 0x4;
        #pragma unroll
        for (int i=0; i<4; i++){
            flip(sel, res[ew][2*i], res[ew][2*i+1],b);
            b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0001000000011111));
            b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0001000000011111));
            twiddle(sc, WN*16, amd_bfe((uint)L,0,2), i);
            flop(sel, res[ew][2*i], res[ew][2*i+1],b);
            butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
        }

        //shuffle 5 -> dpp across 4
        sel = L & 0x2;
        #pragma unroll
        for (int i=0; i<4; i++){
            flip(sel, res[ew][2*i], res[ew][2*i+1],b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b01001110, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b01001110, 0xf, 0xf, 0));
            twiddle(sc, WN*32, amd_bfe((uint)L,0,1), i);
            flop(sel, res[ew][2*i], res[ew][2*i+1],b);
            butterfly(res[ew][2*i], res[ew][2*i+1], sc, b);
        }

        //shuffle 6 -> dpp across 2
        sel = L & 0x1;
        #pragma unroll
        for (int i=0; i<4; i++) {
            flip(sel, res[ew][2*i], res[ew][2*i+1],b);
            b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b10110001, 0xf, 0xf, 0));
            b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b10110001, 0xf, 0xf, 0));
            twiddle(sc, WN*64, 0, i);
            flop(sel, res[ew][2*i], res[ew][2*i+1],b);
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
        uint irev = L*8;;
        __asm__ ("V_BFREV_B32 %0, %1" : "=v"(irev) : "v"(irev)); //32b bit-reverse
        irev = irev>>23; //9b reverse
        transfer[irev    ] = res[ew][0]; //hardcoded offsets to speed up bit flip
        transfer[irev+256] = res[ew][1];
        transfer[irev+128] = res[ew][2];
        transfer[irev+384] = res[ew][3];
        transfer[irev+ 64] = res[ew][4];
        transfer[irev+320] = res[ew][5];
        transfer[irev+192] = res[ew][6];
        transfer[irev+448] = res[ew][7];

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
                                 res[3][ns].RE*lco[bidx*4+3].RE + res[3][ns].IM*lco[bidx*4+3].IM)/4.f;
            beam[ns][bidx].IM = (res[0][ns].RE*lco[bidx*4+0].IM - res[0][ns].IM*lco[bidx*4+0].RE +
                                 res[1][ns].RE*lco[bidx*4+1].IM - res[1][ns].IM*lco[bidx*4+1].RE +
                                 res[2][ns].RE*lco[bidx*4+2].IM - res[2][ns].IM*lco[bidx*4+2].RE +
                                 res[3][ns].RE*lco[bidx*4+3].IM - res[3][ns].IM*lco[bidx*4+3].RE)/4.f;
            uint addr = T * 2048 + //time
                        P * 1024 + //pol
                        bidx * 256 +             //EW
                        (255-(L*4+ns));          //position
            outputData[addr] = (half2){beam[ns][bidx].RE,beam[ns][bidx].IM};
        }
    }
}
