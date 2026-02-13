#define xl get_local_id(0)
#define yl get_local_id(1)
#define zl get_local_id(2)

#define xg get_global_id(0)
#define yg get_global_id(1)
#define zg get_global_id(2)

#define xgr get_group_id(0)
#define ygr get_group_id(1)
#define zgr get_group_id(2)

#define NUM_FREQS get_num_groups(0)
#define NUM_TIMES get_num_groups(1)
#define NUM_BLOCKS get_num_groups(2)

#define FREQ_ID xgr
#define TIME_ID ygr
#define BLOCK_ID zgr

// Requires the following define constants to be passed in at run time
// * SAMPLES_PER_DATA_SET
// * NUM_ELEMENTS
// * NUM_FREQS
// * BLOCK_SIZE
// * COARSE_BLOCK_SIZE

// This is the same as the kernel in kv_corr_sm.cl, but with the ifdef tests removed.
// Removing the ifdefs was needed to support very old versions of the OpenCL compiler
__kernel __attribute__((reqd_work_group_size(COARSE_BLOCK_SIZE, COARSE_BLOCK_SIZE, 1)))
void corr ( __global const uint *packed,
            __global int *presum,
            __global int *corr_buf,
            __global const uint *id_x_map,
            __global const uint *id_y_map)
{
    //figure out where to load data om
    uint addr_x = id_x_map[BLOCK_ID]*COARSE_BLOCK_SIZE + xl;
    uint addr_y = id_y_map[BLOCK_ID]*COARSE_BLOCK_SIZE + yl;

    //pre-seed
    if (TIME_ID==0)
        for (int y=0; y<4; y++) for (int x=0; x<4; x++){
            corr_buf[ (((FREQ_ID*NUM_BLOCKS+BLOCK_ID)*BLOCK_SIZE*BLOCK_SIZE + (yl*4+y)*BLOCK_SIZE + xl*4+x)*2)+0 ] =
                                             8*(presum[(FREQ_ID*NUM_ELEMENTS+addr_x*4+x)*2+0] - 
                                                presum[(FREQ_ID*NUM_ELEMENTS+addr_y*4+y)*2+0] -
                                                presum[(FREQ_ID*NUM_ELEMENTS+addr_x*4+x)*2+1] +
                                                presum[(FREQ_ID*NUM_ELEMENTS+addr_y*4+y)*2+1]);
            corr_buf[ (((FREQ_ID*NUM_BLOCKS+BLOCK_ID)*BLOCK_SIZE*BLOCK_SIZE + (yl*4+y)*BLOCK_SIZE + xl*4+x)*2)+1 ] =
                                                128 * SAMPLES_PER_DATA_SET*NUM_TIMES - 
                                             8*(presum[(FREQ_ID*NUM_ELEMENTS+addr_x*4+x)*2+0] +
                                                presum[(FREQ_ID*NUM_ELEMENTS+addr_y*4+y)*2+0] +
                                                presum[(FREQ_ID*NUM_ELEMENTS+addr_x*4+x)*2+1] +
                                                presum[(FREQ_ID*NUM_ELEMENTS+addr_y*4+y)*2+1]);
        }

    //seed the workgroup with staggered time offsets
    uint T = SAMPLES_PER_DATA_SET * TIME_ID;
    uint t = ((xl + yl) % COARSE_BLOCK_SIZE);

    //temporary registers that hold the inputs; y is packed x is not
    uint x_re[4], x_im[4], y_ir[4];

    //big 'ol outer loop, to do arbitrarily long accumulations
    for ( ; t<SAMPLES_PER_DATA_SET; t+=256){
        //zero the accumulation buffers
        uint corr_0r_ir[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        uint corr_0i_ir[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};

        //accumulate 256 samples before unpacking
        for (int j=0; j<256; j+=COARSE_BLOCK_SIZE){
            local uint xp[COARSE_BLOCK_SIZE][COARSE_BLOCK_SIZE];
            local uint yp[COARSE_BLOCK_SIZE][COARSE_BLOCK_SIZE];
            //load up 4 inputs from each side of the 4x4 block
            barrier(CLK_GLOBAL_MEM_FENCE); //make sure everyone is done
            xp[yl][xl] = packed[ ((j+t+T)*NUM_FREQS + FREQ_ID) * NUM_ELEMENTS/4 + addr_x ];
            yp[yl][xl] = packed[ ((j+t+T)*NUM_FREQS + FREQ_ID) * NUM_ELEMENTS/4 + addr_y ];
            barrier(CLK_GLOBAL_MEM_FENCE); //make sure everyone is done
            //process 8 timesteps before reloading
            for (int i=0; i<COARSE_BLOCK_SIZE; i++){
                uint xv = xp[(yl+i)%COARSE_BLOCK_SIZE][xl];
                x_re[0] = ((xv & 0x0000000f) >>  0u);
                x_im[0] = ((xv & 0x000000f0) >>  4u);
                x_re[1] = ((xv & 0x00000f00) >>  8u);
                x_im[1] = ((xv & 0x0000f000) >> 12u);
                x_re[2] = ((xv & 0x000f0000) >> 16u);
                x_im[2] = ((xv & 0x00f00000) >> 20u);
                x_re[3] = ((xv & 0x0f000000) >> 24u);
                x_im[3] = ((xv & 0xf0000000) >> 28u);
                uint yv = yp[yl][(xl+i)%COARSE_BLOCK_SIZE];
                y_ir[0] = ((yv & 0x000000f0) << 12u) +  ((yv & 0x0000000f) >>  0u);
                y_ir[1] = ((yv & 0x0000f000) <<  4u) +  ((yv & 0x00000f00) >>  8u);
                y_ir[2] = ((yv & 0x00f00000) >>  4u) +  ((yv & 0x000f0000) >> 16u);
                y_ir[3] = ((yv & 0xf0000000) >> 12u) +  ((yv & 0x0f000000) >> 24u);
                //16x umad24, all rolled up
                #pragma unroll
                for (int y=0; y<4; y++) for (int x=0; x<4; x++) {
                    corr_0r_ir[y][x] = mad24(x_re[x],y_ir[y],corr_0r_ir[y][x]);
                    corr_0i_ir[y][x] = mad24(x_im[x],y_ir[y],corr_0i_ir[y][x]);
                }
            }
        }
        global int *out=(corr_buf + ((FREQ_ID*NUM_BLOCKS + BLOCK_ID)*BLOCK_SIZE*BLOCK_SIZE + yl*BLOCK_SIZE*4 + xl*4)*2);
        #pragma unroll
        for (int y=0; y<4; y++){
            #pragma unroll
            for (int x=0; x<4; x++) {
                atomic_add(out++,(corr_0r_ir[y][x]>>16)-(corr_0i_ir[y][x]&0xffff));
                atomic_add(out++,(corr_0r_ir[y][x]&0xffff)+(corr_0i_ir[y][x]>>16));
            }
            out+=(BLOCK_SIZE-4)*2; //(32-4)*2;
        }
    }
}
