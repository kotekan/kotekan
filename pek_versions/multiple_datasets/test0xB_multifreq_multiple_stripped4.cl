//CASPER ordering
//version 0xB: attempting to clean up and make readable
//vectorizing mad24 operations
//comparable to version 4 for speed

//#pragma OPENCL EXTENSION cl_amd_printf : enable

//HARDCODE ALL OF THESE!
//NUM_ELEMENTS, NUM_FREQUENCIES, NUM_BLOCKS defined at compile time
//#define NUM_ELEMENTS                                32u // 2560u eventually //minimum 32
//#define NUM_FREQUENCIES                             128u
//#define NUM_BLOCKS                                  1u  // N(N+1)/2 where N=(NUM_ELEMENTS/32)

#define NUM_ELEMENTS_div_4                          (NUM_ELEMENTS/4u)  // N/4
#define _256_x_NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES (256u*NUM_ELEMENTS_div_4*NUM_FREQUENCIES)
#define NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES        (NUM_ELEMENTS_div_4*NUM_FREQUENCIES)
#define NUM_BLOCKS_x_2048                           (NUM_BLOCKS*2048u) //each block size is 32 x 32 x 2 = 2048

#define LOCAL_SIZE                                  8u
#define BLOCK_DIM_div_4                             8u
#define N_TIME_CHUNKS_LOCAL                         256u

#define FREQUENCY_BAND                              (get_group_id(1))
#define TIME_STEP_DIV_256                           (get_global_id(2)/NUM_BLOCKS)
#define BLOCK_ID                                    (get_global_id(2)%NUM_BLOCKS)
#define LOCAL_X                                     (get_local_id(0))
#define LOCAL_Y                                     (get_local_id(1))


__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, LOCAL_SIZE, 1)))
void corr ( __global uint *packed,
            __global  int *corr_buf,
            __global uint *id_x_map,
            __global uint *id_y_map,
            __local  uint *stillPacked )
{
    uint block_x = id_x_map[BLOCK_ID]; //column of output block
    uint block_y = id_y_map[BLOCK_ID]; //row of output block  //if NUM_BLOCKS = 1, then BLOCK_ID = 0 then block_x = block_y = 0

    uint addr_x = ( (BLOCK_DIM_div_4*block_x + LOCAL_X)
                    + TIME_STEP_DIV_256 * _256_x_NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES);
    uint addr_y = ( (BLOCK_DIM_div_4*block_y + LOCAL_Y)
                    + LOCAL_X*NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES /*shift local_x timesteps ahead*/
                    + TIME_STEP_DIV_256 * _256_x_NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES);


    uint4 corr_a=(uint4)(0,0,0,0);
    uint4 corr_b=(uint4)(0,0,0,0);
    uint4 corr_c=(uint4)(0,0,0,0);
    uint4 corr_d=(uint4)(0,0,0,0);
    uint4 corr_e=(uint4)(0,0,0,0);
    uint4 corr_f=(uint4)(0,0,0,0);
    uint4 corr_g=(uint4)(0,0,0,0);
    uint4 corr_h=(uint4)(0,0,0,0);
    uint4 corr_temp;

    uint4 temp_stillPacked=(uint4)(0,0,0,0);
    uint4 temp_pa;


    for (uint i = 0; i < N_TIME_CHUNKS_LOCAL; i += LOCAL_SIZE){ //256 is a number of timesteps to do a local accum before saving to global memory

        //uint pa=packed[mad24(i, NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES, addr_y + (get_group_id(1)) * NUM_ELEMENTS_div_4)];
        uint pa=packed[i * NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES + addr_y ];
        uint la=(LOCAL_X*32u + LOCAL_Y*4u);

        barrier(CLK_LOCAL_MEM_FENCE);
        stillPacked[la]   = ((pa & 0x000000f0) << 12u) | ((pa & 0x0000000f) >>  0u);
        stillPacked[la+1u] = ((pa & 0x0000f000) <<  4u) | ((pa & 0x00000f00) >>  8u);
        stillPacked[la+2u] = ((pa & 0x00f00000) >>  4u) | ((pa & 0x000f0000) >> 16u);
        stillPacked[la+3u] = ((pa & 0xf0000000) >> 12u) | ((pa & 0x0f000000) >> 24u);
//         stillPacked[la]   = (((pa >>  0) & 0xf) << 16) + ((pa >>  4) & 0xf);
//         stillPacked[la+1] = (((pa >>  8) & 0xf) << 16) + ((pa >> 12) & 0xf);
//         stillPacked[la+2] = (((pa >> 16) & 0xf) << 16) + ((pa >> 20) & 0xf);
//         stillPacked[la+3] = (((pa >> 24) & 0xf) << 16) + ((pa >> 28) & 0xf);
        barrier(CLK_LOCAL_MEM_FENCE);


        for (uint j=0; j< LOCAL_SIZE; j++){
            temp_stillPacked = vload4(j*8u + LOCAL_Y, stillPacked);

            pa = packed[addr_x + (i+j)*NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES ];

            temp_pa.s0 = (pa >> 4u)  & 0xf; //real
            temp_pa.s1 = (pa >> 0u)  & 0xf; //imag
            temp_pa.s2 = (pa >> 12u) & 0xf; //real
            temp_pa.s3 = (pa >> 8u)  & 0xf; //imag

            corr_temp = temp_stillPacked.s0000;
            corr_a = mad24(temp_pa,  corr_temp, corr_a);

            corr_temp = temp_stillPacked.s1111;
            corr_b=mad24(temp_pa, corr_temp, corr_b);

            corr_temp = temp_stillPacked.s2222;
            corr_c=mad24(temp_pa, corr_temp, corr_c);

            corr_temp = temp_stillPacked.s3333;
            corr_d=mad24(temp_pa, corr_temp, corr_d);

            temp_pa.s0 = (pa >> 20u) & 0xf;
            temp_pa.s1 = (pa >> 16u) & 0xf;
            temp_pa.s2 = (pa >> 28u) & 0xf;
            temp_pa.s3 = (pa >> 24u) & 0xf;

            corr_temp = temp_stillPacked.s0000;
            corr_e=mad24(temp_pa, corr_temp, corr_e);

            corr_temp = temp_stillPacked.s1111;
            corr_f=mad24(temp_pa, corr_temp, corr_f);

            corr_temp = temp_stillPacked.s2222;
            corr_g=mad24(temp_pa, corr_temp, corr_g);

            corr_temp = temp_stillPacked.s3333;
            corr_h=mad24(temp_pa, corr_temp, corr_h);
        }
    }
    //output: 32 numbers--> 16 pairs of real/imag numbers 
    //16 pairs * 8 (local_size(0)) * 8 (local_size(1)) = 1024
    uint addr_o = ((BLOCK_ID * 2048u) + (LOCAL_Y * 256u) + (LOCAL_X * 8u)) ;

    atomic_add(&corr_buf[addr_o+0u],   (corr_a.s0 >> 16u) + (corr_a.s1 & 0xffff) ); //real value
    atomic_add(&corr_buf[addr_o+4u],   (corr_e.s0 >> 16u) + (corr_e.s1 & 0xffff) );
    atomic_add(&corr_buf[addr_o+64u],  (corr_b.s0 >> 16u) + (corr_b.s1 & 0xffff) );
    atomic_add(&corr_buf[addr_o+68u],  (corr_f.s0 >> 16u) + (corr_f.s1 & 0xffff) );
    atomic_add(&corr_buf[addr_o+128u], (corr_c.s0 >> 16u) + (corr_c.s1 & 0xffff) );
    atomic_add(&corr_buf[addr_o+132u], (corr_g.s0 >> 16u) + (corr_g.s1 & 0xffff) );
    atomic_add(&corr_buf[addr_o+192u], (corr_d.s0 >> 16u) + (corr_d.s1 & 0xffff) );
    atomic_add(&corr_buf[addr_o+196u], (corr_h.s0 >> 16u) + (corr_h.s1 & 0xffff) );

    atomic_add(&corr_buf[addr_o+1u],   (corr_a.s1 >> 16u) - (corr_a.s0 & 0xffff) );//imaginary value
    atomic_add(&corr_buf[addr_o+5u],   (corr_e.s1 >> 16u) - (corr_e.s0 & 0xffff) );
    atomic_add(&corr_buf[addr_o+65u],  (corr_b.s1 >> 16u) - (corr_b.s0 & 0xffff) );
    atomic_add(&corr_buf[addr_o+69u],  (corr_f.s1 >> 16u) - (corr_f.s0 & 0xffff) );
    atomic_add(&corr_buf[addr_o+129u], (corr_c.s1 >> 16u) - (corr_c.s0 & 0xffff) );
    atomic_add(&corr_buf[addr_o+133u], (corr_g.s1 >> 16u) - (corr_g.s0 & 0xffff) );
    atomic_add(&corr_buf[addr_o+193u], (corr_d.s1 >> 16u) - (corr_d.s0 & 0xffff) );
    atomic_add(&corr_buf[addr_o+197u], (corr_h.s1 >> 16u) - (corr_h.s0 & 0xffff) );

    atomic_add(&corr_buf[addr_o+2u],   (corr_a.s2 >> 16u) + (corr_a.s3 & 0xffff) );
    atomic_add(&corr_buf[addr_o+6u],   (corr_e.s2 >> 16u) + (corr_e.s3 & 0xffff) );
    atomic_add(&corr_buf[addr_o+66u],  (corr_b.s2 >> 16u) + (corr_b.s3 & 0xffff) );
    atomic_add(&corr_buf[addr_o+70u],  (corr_f.s2 >> 16u) + (corr_f.s3 & 0xffff) );
    atomic_add(&corr_buf[addr_o+130u], (corr_c.s2 >> 16u) + (corr_c.s3 & 0xffff) );
    atomic_add(&corr_buf[addr_o+134u], (corr_g.s2 >> 16u) + (corr_g.s3 & 0xffff) );
    atomic_add(&corr_buf[addr_o+194u], (corr_d.s2 >> 16u) + (corr_d.s3 & 0xffff) );
    atomic_add(&corr_buf[addr_o+198u], (corr_h.s2 >> 16u) + (corr_h.s3 & 0xffff) );

    atomic_add(&corr_buf[addr_o+3u],   (corr_a.s3 >> 16u) - (corr_a.s2 & 0xffff) );
    atomic_add(&corr_buf[addr_o+7u],   (corr_e.s3 >> 16u) - (corr_e.s2 & 0xffff) );
    atomic_add(&corr_buf[addr_o+67u],  (corr_b.s3 >> 16u) - (corr_b.s2 & 0xffff) );
    atomic_add(&corr_buf[addr_o+71u],  (corr_f.s3 >> 16u) - (corr_f.s2 & 0xffff) );
    atomic_add(&corr_buf[addr_o+131u], (corr_c.s3 >> 16u) - (corr_c.s2 & 0xffff) );
    atomic_add(&corr_buf[addr_o+135u], (corr_g.s3 >> 16u) - (corr_g.s2 & 0xffff) );
    atomic_add(&corr_buf[addr_o+195u], (corr_d.s3 >> 16u) - (corr_d.s2 & 0xffff) );
    atomic_add(&corr_buf[addr_o+199u], (corr_h.s3 >> 16u) - (corr_h.s2 & 0xffff) );
}
