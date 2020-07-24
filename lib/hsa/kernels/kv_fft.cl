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

#define WNang -0.01227184630308513 //2pi/512
#define CUSTOM_BIT_REVERSE_9_BITS(index) ((( ( (((index) * 0x0802) & 0x22110) | (((index) * 0x8020)&0x88440) ) * 0x10101 ) >> 15) & 0x1FE)

#define L get_local_id(0)

__kernel void kv_fft (__global uint *inputData,
                      __global float2 *outputData){
    float2 res[8];
    float2 sc;
    float twiddle_angle;

    uint data_temp = inputData[get_local_size(0) * get_group_id(0) + L];

    {
        res[0].IM = ((int)((data_temp & 0x0000000f) >>   0u)) - 8;
        res[0].RE = ((int)((data_temp & 0x000000f0) >>   4u)) - 8;
        twiddle_angle = WNang * (L*4+0);
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        res[1].IM = res[0].RE * sc.IM + res[0].IM * sc.RE;
        res[1].RE = res[0].RE * sc.RE - res[0].IM * sc.IM;

        res[2].IM = ((int)((data_temp & 0x00000f00) >>   8u)) - 8;
        res[2].RE = ((int)((data_temp & 0x0000f000) >>  12u)) - 8;
        twiddle_angle = WNang * (L*4+1);
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        res[3].IM = res[2].RE * sc.IM + res[2].IM * sc.RE;
        res[3].RE = res[2].RE * sc.RE - res[2].IM * sc.IM;

        res[4].IM = ((int)((data_temp & 0x000f0000) >>  16u)) - 8;
        res[4].RE = ((int)((data_temp & 0x00f00000) >>  20u)) - 8;
        twiddle_angle = WNang * (L*4+2);
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        res[5].IM = res[4].RE * sc.IM + res[4].IM * sc.RE;
        res[5].RE = res[4].RE * sc.RE - res[4].IM * sc.IM;

        res[6].IM = ((int)((data_temp & 0x0f000000) >>  24u)) - 8;
        res[6].RE = ((int)((data_temp & 0xf0000000) >>  28u)) - 8;
        twiddle_angle = WNang * (L*4+3);
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        res[7].IM = res[6].RE * sc.IM + res[6].IM * sc.RE;
        res[7].RE = res[6].RE * sc.RE - res[6].IM * sc.IM;
    }

    //shuffle 1 -> bpermute across all 64
    #pragma unroll
    for (int i=0; i<4; i++){
        float2 a,b;
        a = (L & 0x20) ? res[2*i+1] : res[2*i  ];
        b = (L & 0x20) ? res[2*i  ] : res[2*i+1];

        b.IM = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.IM)));
        b.RE = as_float(__builtin_amdgcn_ds_bpermute(4*(L+32), as_uint(b.RE)));

        res[2*i  ] = (L & 0x20) ? b : a;
        res[2*i+1] = (L & 0x20) ? a : b;

        twiddle_angle = WNang * ((L&31)*4+i)*2;
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        a = res[2*i] + res[2*i+1];
        b = res[2*i] - res[2*i+1];
        res[2*i  ] = a;
        res[2*i+1].IM = b.RE * sc.IM + b.IM * sc.RE;
        res[2*i+1].RE = b.RE * sc.RE - b.IM * sc.IM;
    }

   //shuffle 2 -> swizzle across 32 WI
    #pragma unroll
    for (int i=0; i<4; i++){
        float2 a,b;
        a = (L & 0x10) ? res[2*i+1] : res[2*i  ];
        b = (L & 0x10) ? res[2*i  ] : res[2*i+1];

        // pattern = 0b 0 10000 00000 11111
        b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0100000000011111));
        b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0100000000011111));

        res[2*i  ] = (L & 0x10) ? b : a;
        res[2*i+1] = (L & 0x10) ? a : b;

        twiddle_angle = WNang * ((L&15)*4+i)*4;
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        a = res[2*i] + res[2*i+1];
        b = res[2*i] - res[2*i+1];
        res[2*i  ] = a;
        res[2*i+1].IM = b.RE * sc.IM + b.IM * sc.RE;
        res[2*i+1].RE = b.RE * sc.RE - b.IM * sc.IM;
    }

    //shuffle 3 -> dpp across 16
    #pragma unroll
    for (int i=0; i<4; i++){
        float2 a,b;
        a = (L & 0x08) ? res[2*i+1] : res[2*i  ];
        b = (L & 0x08) ? res[2*i  ] : res[2*i+1];

        //use DPP to swap a full row (rotate by 8)
        b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0x128, 0xf, 0xf, 0));
        b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0x128, 0xf, 0xf, 0));
        // pattern = 0b 0 01000 00000 11111 (0 5xor 5or 5and)
        //b.IM = __builtin_amdgcn_ds_swizzle(b.IM, 0b0010000000011111);
        //b.RE = __builtin_amdgcn_ds_swizzle(b.RE, 0b0010000000011111);

        res[2*i  ] = (L & 0x08) ? b : a;
        res[2*i+1] = (L & 0x08) ? a : b;

        twiddle_angle = WNang * ((L&7)*4+i)*8;
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        a = res[2*i] + res[2*i+1];
        b = res[2*i] - res[2*i+1];
        res[2*i  ] = a;
        res[2*i+1].IM = b.RE * sc.IM + b.IM * sc.RE;
        res[2*i+1].RE = b.RE * sc.RE - b.IM * sc.IM;
    }

    //shuffle 4 -> swizzle across 8
    #pragma unroll
    for (int i=0; i<4; i++){
        float2 a,b;
        a = (L & 0x04) ? res[2*i+1] : res[2*i  ];
        b = (L & 0x04) ? res[2*i  ] : res[2*i+1];

        // pattern = 0b 0 00100 00000 11111 (0 5xor 5or 5and)
        b.IM = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.IM), 0b0001000000011111));
        b.RE = as_float(__builtin_amdgcn_ds_swizzle(as_uint(b.RE), 0b0001000000011111));

        res[2*i  ] = (L & 0x04) ? b : a;
        res[2*i+1] = (L & 0x04) ? a : b;

        twiddle_angle = WNang * ((L&3)*4+i)*16;
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        a = res[2*i] + res[2*i+1];
        b = res[2*i] - res[2*i+1];
        res[2*i  ] = a;
        res[2*i+1].IM = b.RE * sc.IM + b.IM * sc.RE;
        res[2*i+1].RE = b.RE * sc.RE - b.IM * sc.IM;
    }

    //shuffle 5 -> dpp across 4
    #pragma unroll
    for (int i=0; i<4; i++){
        float2 a,b;
        a = (L & 0x02) ? res[2*i+1] : res[2*i  ];
        b = (L & 0x02) ? res[2*i  ] : res[2*i+1];

        //use DPP to swap among 4
        // 0 <-> 2 ; 1 <-> 3 ==> 0b 01 00 11 10
        b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b01001110, 0xf, 0xf, 0));
        b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b01001110, 0xf, 0xf, 0));
        // pattern = 0b 0 00010 00000 11111 (0 5xor 5or 5and)
        //b.IM = __builtin_amdgcn_ds_swizzle(b.IM, 0b0000100000011111);
        //b.RE = __builtin_amdgcn_ds_swizzle(b.RE, 0b0000100000011111);

        res[2*i  ] = (L & 0x02) ? b : a;
        res[2*i+1] = (L & 0x02) ? a : b;

        twiddle_angle = WNang * ((L&1)*4+i)*32;
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        a = res[2*i] + res[2*i+1];
        b = res[2*i] - res[2*i+1];
        res[2*i  ] = a;
        res[2*i+1].IM = b.RE * sc.IM + b.IM * sc.RE;
        res[2*i+1].RE = b.RE * sc.RE - b.IM * sc.IM;
    }

    //shuffle 6 -> dpp across 2
    #pragma unroll
    for (int i=0; i<4; i++) {
        float2 a,b;
        a = (L & 0x01) ? res[2*i+1] : res[2*i  ];
        b = (L & 0x01) ? res[2*i  ] : res[2*i+1];

        //use DPP to swap adjacent
        // 0 <-> 1 ; 2 <-> 3  ==>  0b 10 11 00 01
        b.IM = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.IM), 0b10110001, 0xf, 0xf, 0));
        b.RE = as_float(__builtin_amdgcn_mov_dpp(as_uint(b.RE), 0b10110001, 0xf, 0xf, 0));
        // pattern = 0b 0 00001 00000 11111 (0 5xor 5or 5and)
        //b.IM = __builtin_amdgcn_ds_swizzle(b.IM, 0b0000010000011111);
        //b.RE = __builtin_amdgcn_ds_swizzle(b.RE, 0b0000010000011111);

        res[2*i  ] = (L & 0x01) ? b : a;
        res[2*i+1] = (L & 0x01) ? a : b;

        twiddle_angle = WNang * ((L&0)*4+i)*64;
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        a = res[2*i] + res[2*i+1];
        b = res[2*i] - res[2*i+1];
        res[2*i  ] = a;
        res[2*i+1].IM = b.RE * sc.IM + b.IM * sc.RE;
        res[2*i+1].RE = b.RE * sc.RE - b.IM * sc.IM;
    }

    //shuffle 7 -> swap internally across 4 pairs
    {
        float2 a[2] = {res[1], res[3]};
        float2 b;

        res[1] = res[0] - res[4];
        res[0] = res[0] + res[4];

        twiddle_angle = WNang * (1)*128;
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        b = res[2] - res[6];
        res[2] = res[2] + res[6];
        res[3].IM = b.RE * sc.IM + b.IM * sc.RE;
        res[3].RE = b.RE * sc.RE - b.IM * sc.IM;

        res[4] = a[0] + res[5];
        res[5] = a[0] - res[5];

        twiddle_angle = WNang * (1)*128;
        sc.IM = native_sin(twiddle_angle);
        sc.RE = native_cos(twiddle_angle);
        b = a[1] - res[7];
        res[6] = a[1] + res[7];
        res[7].IM = b.RE * sc.IM + b.IM * sc.RE;
        res[7].RE = b.RE * sc.RE - b.IM * sc.IM;
    }
    //shuffle 8 -> swap internally across 2 pairs
    {
        float2 a[2] = {res[1], res[5]};

        res[1] = res[0] - res[2];
        res[0] = res[0] + res[2];
        res[2] = a[0] + res[3];
        res[3] = a[0] - res[3];
        res[5] = res[4] - res[6];
        res[4] = res[4] + res[6];
        res[6] = a[1] + res[7];
        res[7] = a[1] - res[7];
    }

    //output!
    #pragma unroll
    for (int i=0; i<8; i++)
    {
        outputData[(CUSTOM_BIT_REVERSE_9_BITS(L*8 + i) | ((L&32)/32)) ] = res[i];
    /*
        outputData[L*8+i + //local group
                   get_group_id(0) * 512 + //other groups at same time
                   get_group_id(1) * 512 * get_num_groups(0) //time offset
                   ] = res[7-i];
   */
    }
}
