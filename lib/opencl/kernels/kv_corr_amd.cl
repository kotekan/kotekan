//#define SAMPLES_PER_DATA_SET
//#define NUM_ELEMENTS

#define xl get_local_id(0)
#define yl get_local_id(1)
#define zl get_local_id(2)

#define xg get_global_id(0)
#define yg get_global_id(1)
#define zg get_global_id(2)

#define xgr get_group_id(0)
#define ygr get_group_id(1)
#define zgr get_group_id(2)

#define n_integrate SAMPLES_PER_DATA_SET

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable

__kernel __attribute__((reqd_work_group_size(16, 4, 1)))
void corr ( __global const uint *packed,
            __global int *presum,
            __global int *corr_buf,
            __global const uint *id_x_map,
            __global const uint *id_y_map)
//            __global int *block_lock)
{
    int ix = xl/2;
    int iy = yl*2 + (xl&0x1);

    //figure out where to load data from
    uint input_x = id_x_map[zg]*8 + ix;
    uint input_y = id_y_map[zg]*8 + iy;

    //pre-seed
    if (ygr == 0)
        for (int y=0; y<4; y++) for (int x=0; x<4; x++){
            corr_buf[ ((zg*1024 + (iy*4+y)*32 + ix*4+x)*2)+1 ] = 
                128 * SAMPLES_PER_DATA_SET - 8*(presum[(input_x*4+x)*2+0] + presum[(input_y*4+y)*2+0] +
                                                presum[(input_x*4+x)*2+1] + presum[(input_y*4+y)*2+1]);
            corr_buf[ ((zg*1024 + (iy*4+y)*32 + ix*4+x)*2)+0 ] =
                                             8*(presum[(input_x*4+x)*2+0] - presum[(input_y*4+y)*2+0] -
                                                presum[(input_x*4+x)*2+1] + presum[(input_y*4+y)*2+1]);
        }


    //seed the 8x8 workgroup with staggered time offsets
    uint T = ((ix + iy) % 8) + ygr*n_integrate;

    //find the address of the work items to hand off to
    //there's gotta be a better way to get this...
    uint dest_x = ((((iy-1)&0x6)<<3) + (xl^0x1)) *4;

    //temporary registers that hold the inputs; y is packed x is not
    uint y_ri[4], x_ir[4], x_ii[2], y_0r[4];

    local uint locoflow_r[4][16][4][2];
    local uint locoflow_i[4][16][4];
    for (; T<n_integrate + ygr*n_integrate; T+=16384){
        //zero the accumulation buffers
        uint corr_rr_ri[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        uint corr_1i_2i[4][2] = {{0,0},    {0,0},    {0,0},    {0,0}    };

        for (int k=0; k<4; k++) {
            locoflow_i[yl][xl][k]    = 0;
            locoflow_r[yl][xl][k][0] = 0;
            locoflow_r[yl][xl][k][1] = 0;
        }

        //big 'ol outer loop, to do arbitrarily long accumulations
        for (uint t=0; t<16384; t+=64){
            //move top bits to overflow to make room in accumulation buffers
            //had to move to before the loop; after, it jumbled the compilation
            for (int y=0; y<4; y++)
                atomic_add(&locoflow_i[yl][xl][y],
                              ((corr_1i_2i[y][0] & 0x80008000) >> 15) +
                              ((corr_1i_2i[y][1] & 0x80008000) >>  7));
            for (int y=0; y<4; y++) for (int x=0; x<2; x++)
                corr_1i_2i[y][x]=corr_1i_2i[y][x] & 0x7fff7fff;

            for (int y=0; y<4; y++)
                atomic_add(&locoflow_r[yl][xl][y][0],
                              ((corr_rr_ri[y][0] & 0x80008000) >> 15) +
                              ((corr_rr_ri[y][1] & 0x80008000) >>  7));
            for (int y=0; y<4; y++)
                atomic_add(&locoflow_r[yl][xl][y][1],
                              ((corr_rr_ri[y][2] & 0x80008000) >> 15) +
                              ((corr_rr_ri[y][3] & 0x80008000) >>  7));
            for (int y=0; y<4; y++) for (int x=0; x<4; x++)
                corr_rr_ri[y][x]=corr_rr_ri[y][x] & 0x7fff7fff;

            //accumulate 64 samples before unpacking
            for (int j=0; j<64; j+=8){
                //load up 4 inputs from each side of the 4x4 block
                uint xv = packed[(T+t+j)*NUM_ELEMENTS/4 + input_x];
                //x_ir == [000I000R]
                x_ir[0] = mad24(0x10000u,amd_bfe(xv, 0,4), amd_bfe(xv, 4,4));
                x_ir[1] = mad24(0x10000u,amd_bfe(xv, 8,4), amd_bfe(xv,12,4));
                x_ir[2] = mad24(0x10000u,amd_bfe(xv,16,4), amd_bfe(xv,20,4));
                x_ir[3] = mad24(0x10000u,amd_bfe(xv,24,4), amd_bfe(xv,28,4));
                //x_ii == [000I000I]
                x_ii[0] = mad24(0x10000u,amd_bfe(xv, 8,4), amd_bfe(xv, 0,4));
                x_ii[1] = mad24(0x10000u,amd_bfe(xv,24,4), amd_bfe(xv,16,4));

                uint yv = packed[(T+t+j)*NUM_ELEMENTS/4 + input_y];
                //y_ri == [000R000I]
                y_ri[0] = mad24(0x10000u,amd_bfe(yv, 4,4), amd_bfe(yv, 0,4));
                y_ri[1] = mad24(0x10000u,amd_bfe(yv,12,4), amd_bfe(yv, 8,4));
                y_ri[2] = mad24(0x10000u,amd_bfe(yv,20,4), amd_bfe(yv,16,4));
                y_ri[3] = mad24(0x10000u,amd_bfe(yv,28,4), amd_bfe(yv,24,4));

                //process 8 timesteps before reloading
                //#pragma unroll //comment this out if you want to read / debug the dump.isa...
                for (int i=0; i<8; i++){
                    //y_0r == [0000000R] -- same price to re-calculate, and it saves registers!
                    for (int k=0; k<4; k++) y_0r[k] = y_ri[k]>>16;

                    //multiples
                    for (int y=0; y<4; y++) for (int x=0; x<4; x++)
                        corr_rr_ri[y][x] = mad24(x_ir[x],y_ri[y],corr_rr_ri[y][x]);
                    for (int y=0; y<4; y++) for (int x=0; x<2; x++)
                        corr_1i_2i[y][x] = mad24(x_ii[x],y_0r[y],corr_1i_2i[y][x]);

                    //then pass data
                    for (int k=0; k<4; k++)
                        x_ir[k] = __builtin_amdgcn_ds_bpermute(dest_x,x_ir[k]);
                    for (int k=0; k<4; k++)
                        y_ri[k] = __builtin_amdgcn_mov_dpp(y_ri[k],0x122,0xf,0xf,0); //rotate right by 2

                    x_ii[0] = __builtin_amdgcn_ds_bpermute(dest_x,x_ii[0]);
                    x_ii[1] = __builtin_amdgcn_ds_bpermute(dest_x,x_ii[1]);
                }
            }
        }
        //unpacked into long-term real, imaginary accumulation buffer.
        global int *out=corr_buf + (zg*1024 + iy*32*4 + ix*4)*2;
        out+=y_ri[0]+73; //stopping pre-VGRP allocation
        #pragma unroll
        for (int y=0; y<4; y++){
            int r[4] = {
                (corr_rr_ri[y][0]>>16) + (amd_bfe(locoflow_r[yl][xl][y][0],16,8) << 15),
                (corr_rr_ri[y][1]>>16) + (amd_bfe(locoflow_r[yl][xl][y][0],24,8) << 15),
                (corr_rr_ri[y][2]>>16) + (amd_bfe(locoflow_r[yl][xl][y][1],16,8) << 15),
                (corr_rr_ri[y][3]>>16) + (amd_bfe(locoflow_r[yl][xl][y][1],24,8) << 15)
            };
            int i[4] = {
                (amd_bfe(locoflow_i[yl][xl][y],    0,8) << 15) -
                (amd_bfe(locoflow_r[yl][xl][y][0], 0,8) << 15) +
                 amd_bfe(corr_1i_2i[y][0], 0,16) - (corr_rr_ri[y][0]&0xffff),
                (amd_bfe(locoflow_i[yl][xl][y],   16,8) << 15) -
                (amd_bfe(locoflow_r[yl][xl][y][0], 8,8) << 15) +
                 amd_bfe(corr_1i_2i[y][0],16,16) - (corr_rr_ri[y][1]&0xffff),
                (amd_bfe(locoflow_i[yl][xl][y],    8,8) << 15) -
                (amd_bfe(locoflow_r[yl][xl][y][1], 0,8) << 15) +
                 amd_bfe(corr_1i_2i[y][1], 0,16) - (corr_rr_ri[y][2]&0xffff),
                (amd_bfe(locoflow_i[yl][xl][y],   24,8) << 15) -
                (amd_bfe(locoflow_r[yl][xl][y][1], 8,8) << 15) +
                 amd_bfe(corr_1i_2i[y][1],16,16) - (corr_rr_ri[y][3]&0xffff)
            };
            #pragma unroll
            for (int x=0; x<4; x++){
                atomic_add(out++ -y_ri[0]-73,i[x]);
                atomic_add(out++ -y_ri[0]-73,r[x]);
            }
            out+=56; //(32-4)*2;
        }
    }
}
