#define T 16 //(sgx/sgy)
#define BS T*gs

#define lx get_local_id(0)
#define ly get_local_id(1)

#define gx get_group_id(0)
#define gy get_group_id(1)

#define sgx get_local_size(0)
#define sgy get_local_size(1)

#define ggx get_global_size(0)
#define ggy get_global_size(1)

#define ngx get_num_groups(0)
#define ngy get_num_groups(1)

//input dimensions [n_samples, n_pol, n_beams] -> [N,2048]
//output dimensions [n_pol, n_beams, n_samples] -> [2048,N]

#define flip(sel, a, b, c) \
        c = select(b,a,sel);

#define flop(sel, a, b, c) \
        a = select(a,c,sel); \
        b = select(c,b,sel);


__kernel
//__attribute__((reqd_work_group_size(32, 2, 1)))
//__attribute__((amdgpu_num_vgpr(28)))
void transpose(__global uint *input, __global uint *output) {
    uint loc[T];
    uint sel;

    #pragma unroll
    for (uint i=0; i<T; i++) {
        uint addr = (sgy*gy*T + ly*T + i)*ggx + (gx*sgx + lx);
        loc[i] = input[addr];
        //loc[i] = ly*16*32 + i*32 + lx;
    }

    //stage 1, 32x32 -> 16x16 blocks
    #pragma unroll
    for (uint i=0; i<T; i++){
        uint dy = (lx & 16)/16;
        uint dx = (lx & 15) + ly*16;
        loc[i] = __builtin_amdgcn_ds_bpermute( (dy*32+dx)*4, loc[i]);
    }

    //stage 2
    sel = lx & 0x8;
    #pragma unroll
    for (uint i=0; i<T/2; i++){
        uint send = select(loc[i+T/2], loc[i], sel);
        send = __builtin_amdgcn_mov_dpp(send, 0x128, 0xf, 0xf, 0);
        loc[i    ] = select(loc[i    ], send, sel);
        loc[i+T/2] = select(send, loc[i+T/2], sel);
    }

    //stage 3
    sel = lx & 0x4;
    #pragma unroll
    for (uint i=0; i<T/4; i++){
        #pragma unroll
        for (uint j=0; j<2; j++){
            uint k = j*T/2 + i;
            uint send = select(loc[k+T/4], loc[k], sel);
            send = __builtin_amdgcn_ds_swizzle(send, 0b0001000000011111);
            loc[k    ] = select(loc[k    ], send, sel);
            loc[k+T/4] = select(send, loc[k+T/4], sel);
        }
    }

    //stage 4
    sel = lx & 0x2;
    #pragma unroll
    for (uint i=0; i<T/8; i++){
        #pragma unroll
        for (uint j=0; j<4; j++){
            uint k = j*T/4 + i;
            uint send = select(loc[k+T/8], loc[k], sel);
            send = __builtin_amdgcn_mov_dpp(send, 0b01001110, 0xf, 0xf, 0);
            loc[k    ] = select(loc[k    ], send, sel);
            loc[k+T/8] = select(send, loc[k+T/8], sel);
        }
    }

    //stage 5
    sel = lx & 0x1;
    #pragma unroll
    for (uint j=0; j<8; j++){
        uint k = j*2;
        uint send = select(loc[k+1], loc[k], sel);
        send = __builtin_amdgcn_mov_dpp(send, 0b10110001, 0xf, 0xf, 0);
        loc[k     ] = select(loc[k     ], send, sel);
        loc[k+T/16] = select(send, loc[k+T/16], sel);
    }

    //full 16x16 transpose...
    /*#pragma unroll
    dx = (lx ^ i) & 0xf;
    for (uint i=0; i<T; i++){
        loc[dx] = __builtin_amdgcn_ds_bpermute( (ly*32 + (lx & 0x10) +dx)*4, loc[dx]);
    }*/

    #pragma unroll
    for (uint i=0; i<T; i++){
//      uint addr = (sgy*gy*T + ly*T + i)*2048 + (gx*sgx + lx);
        uint addr = (gx*sgx + ly*T + i)*(ggy*T+64) + gy*sgy*T + lx;
        output[addr] = loc[i];
//        output[ly*16*32 + i*32 + lx] = loc[i];
    }
}

