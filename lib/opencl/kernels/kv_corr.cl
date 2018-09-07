//#define SAMPLES_PER_DATA_SET
//#define NUM_ELEMENTS

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel //__attribute__((reqd_work_group_size(LOCAL_SIZE, LOCAL_SIZE, 1)))
void corr ( __global const uint *packed,
            __global int *presum,
            __global int *corr_buf,
            __global const uint *id_x_map,
            __global const uint *id_y_map,
            __global int *block_lock)
{
    local uint x_re_buf[64][4];
    local uint x_im_buf[64][4];
    local uint y_ir_buf[64][4];

    //figure out where to load data from
    uint addr_x = id_x_map[get_global_id(0)]*8 + get_local_id(2);
    uint addr_y = id_y_map[get_global_id(0)]*8 + get_local_id(1);

    //pre-seed
    if (get_group_id(1) == 0)
        for (int y=0; y<4; y++) for (int x=0; x<4; x++){
            corr_buf[ ((get_global_id(0)*1024 + (get_local_id(1)*4+y)*32 + get_local_id(2)*4+x)*2)+0 ] =
                128 * SAMPLES_PER_DATA_SET - 8*(presum[(addr_x*4+x)*2+0] + presum[(addr_y*4+y)*2+0] +
                                                presum[(addr_x*4+x)*2+1] + presum[(addr_y*4+y)*2+1]);
            corr_buf[ ((get_global_id(0)*1024 + (get_local_id(1)*4+y)*32 + get_local_id(2)*4+x)*2)+1 ] =
                                             8*(presum[(addr_x*4+x)*2+0] - presum[(addr_y*4+y)*2+0] -
                                                presum[(addr_x*4+x)*2+1] + presum[(addr_y*4+y)*2+1]);
        }


    //seed the 8x8 workgroup with staggered time offsets
    uint t = ((get_local_id(2) + get_local_id(1)) % 8);

    //find the address of the work items to hand off to
    uint dest_x = ((get_local_id(1)+1)%8)*8 + get_local_id(2);
    uint dest_y = ((get_local_id(2)+1)%8)   + get_local_id(1)*8;

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
//                    x_re[k] = (uint)__builtin_amdgcn_ds_bpermute(x_re[k],dest_x);
//                    x_im[k] = (uint)__builtin_amdgcn_ds_bpermute(x_im[k],dest_x);
//                    y_ir[k] = (uint)__builtin_amdgcn_ds_bpermute(y_ir[k],dest_y);
                }
                barrier(CLK_GLOBAL_MEM_FENCE); //make sure everyone is done
                for (int k=0; k<4; k++) {
                    x_re_buf[dest_x][k]=x_re[k];
                    x_im_buf[dest_x][k]=x_im[k];
                    y_ir_buf[dest_x][k]=y_ir[k];
                }
                barrier(CLK_GLOBAL_MEM_FENCE); //make sure everyone is done
                for (int k=0; k<4; k++) {
                    x_re[k]=x_re_buf[dest_x][k];
                    x_im[k]=x_im_buf[dest_x][k];
                    y_ir[k]=y_ir_buf[dest_x][k];
                }
            }
        }
        __global int *out=(corr_buf + (get_global_id(0)*1024 + get_global_id(1)*32*4 + get_global_id(2)*4)*2);
        for (int y=0; y<4; y++){
            for (int x=0; x<4; x++) {
                atomic_add(out++,(corr_0r_ir[y][x]&0xffff)+(corr_0i_ir[y][x]>>16));
                atomic_add(out++,(corr_0i_ir[y][x]&0xffff)-(corr_0r_ir[y][x]>>16));
            }
            out+=56; //(32-4)*2;
        }
    }
}
