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

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void corr ( __global const uint *packed,
            __global int *presum,
            __global int *corr_buf,
            __global const uint *id_x_map,
            __global const uint *id_y_map,
            __global int *block_lock)
{
    //figure out where to load data from
    uint addr_x = id_x_map[zg]*8 + xl;
    uint addr_y = id_y_map[zg]*8 + yl;

    //pre-seed
    if (ygr == 0)
        for (int y=0; y<4; y++) for (int x=0; x<4; x++){
            corr_buf[ ((zg*1024 + (yl*4+y)*32 + xl*4+x)*2)+0 ] =
                128 * SAMPLES_PER_DATA_SET - 8*(presum[(addr_x*4+x)*2+0] + presum[(addr_y*4+y)*2+0] +
                                                presum[(addr_x*4+x)*2+1] + presum[(addr_y*4+y)*2+1]);
            corr_buf[ ((zg*1024 + (yl*4+y)*32 + xl*4+x)*2)+1 ] =
                                             8*(presum[(addr_x*4+x)*2+0] - presum[(addr_y*4+y)*2+0] -
                                                presum[(addr_x*4+x)*2+1] + presum[(addr_y*4+y)*2+1]);
        }


    //seed the 8x8 workgroup with staggered time offsets
    uint t = ((xl + yl) % 8);

    //find the address of the work items to hand off to
    uint dest_x = ((yl+1)%8)*8 + xl;
    uint dest_y = ((xl+1)%8)   + yl*8;

    //temporary registers that hold the inputs; y is packed x is not
    uint x_re[4], x_im[4], y_ir[4];

    //big 'ol outer loop, to do arbitrarily long accumulations
    for ( ; t<SAMPLES_PER_DATA_SET; t+=256){
        //zero the accumulation buffers
        uint corr_0r_ir[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        uint corr_0i_ir[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};

        //accumulate 256 samples before unpacking
        for (int j=0; j<256; j+=8){
            //load up 4 inputs from each side of the 4x4 block
            uint xv = packed[(j+t)*NUM_ELEMENTS/4 + addr_x];
            //uint xv = a_input(j+t,input_x);
            x_re[0] = ((xv & 0x0000000f) >>  0u);
            x_im[0] = ((xv & 0x000000f0) >>  4u);
            x_re[1] = ((xv & 0x00000f00) >>  8u);
            x_im[1] = ((xv & 0x0000f000) >> 12u);
            x_re[2] = ((xv & 0x000f0000) >> 16u);
            x_im[2] = ((xv & 0x00f00000) >> 20u);
            x_re[3] = ((xv & 0x0f000000) >> 24u);
            x_im[3] = ((xv & 0xf0000000) >> 28u);
            uint yv = packed[(j+t)*NUM_ELEMENTS/4 + addr_y];
            //uint yv = a_input(j+t,input_y);
            y_ir[0] = ((yv & 0x000000f0) << 12u) +  ((yv & 0x0000000f) >>  0u);
            y_ir[1] = ((yv & 0x0000f000) <<  4u) +  ((yv & 0x00000f00) >>  8u);
            y_ir[2] = ((yv & 0x00f00000) >>  4u) +  ((yv & 0x000f0000) >> 16u);
            y_ir[3] = ((yv & 0xf0000000) >> 12u) +  ((yv & 0x0f000000) >> 24u);

            //process 8 timesteps before reloading
            for (int i=0; i<8; i++){
                //16x umad24, all rolled up
                #pragma unroll
                for (int y=0; y<4; y++) for (int x=0; x<4; x++) {
                    corr_0r_ir[y][x] = mad24(x_re[x],y_ir[y],corr_0r_ir[y][x]);
                    corr_0i_ir[y][x] = mad24(x_im[x],y_ir[y],corr_0i_ir[y][x]);
                }
                //rotate data to the neighbour work items
                #pragma unroll
                for (int k=0; k<4; k++){
                    x_re[k] = (uint)__builtin_amdgcn_ds_bpermute(dest_x,x_re[k]);
                    x_im[k] = (uint)__builtin_amdgcn_ds_bpermute(dest_x,x_im[k]);
                    y_ir[k] = (uint)__builtin_amdgcn_ds_bpermute(dest_y,y_ir[k]);
                }
            }
        }
        global int *out=(corr_buf + (zg*1024 + yg*32*4 + xg*4)*2);
        #pragma unroll
        for (int y=0; y<4; y++){
            #pragma unroll
            for (int x=0; x<4; x++) {
                atomic_add(out++,(corr_0r_ir[y][x]&0xffff)+(corr_0i_ir[y][x]>>16));
                atomic_add(out++,(corr_0i_ir[y][x]&0xffff)-(corr_0r_ir[y][x]>>16));
            }
            out+=56; //(32-4)*2;
        }
    }
}
