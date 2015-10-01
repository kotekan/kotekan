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
void corr ( __global const uint *packed,
            __global  int *corr_buf,
            __global const uint *id_x_map,
            __global const uint *id_y_map,
            __global int *block_lock)
{
    __local uint stillPackedY[256];
    __local uint stillPackedX[256];
    const uint block_x = id_x_map[BLOCK_ID]; //column of output block
    const uint block_y = id_y_map[BLOCK_ID]; //row of output block  //if NUM_BLOCKS = 1, then BLOCK_ID = 0 then block_x = block_y = 0

    /// The address for the x elements for the data
    uint addr_x = (   LOCAL_Y*NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES
                    + TIME_STEP_DIV_256 * _256_x_NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES
                    + FREQUENCY_BAND*NUM_ELEMENTS_div_4); //temporarily precompute an offset

    /// The address for the y elements for the data
    uint addr_y = ( (BLOCK_DIM_div_4*block_y + LOCAL_X) + addr_x);

    addr_x += (BLOCK_DIM_div_4*block_x + LOCAL_X); //the x address

    uint corr_a0=0u;
    uint corr_b0=0u;
    uint corr_c0=0u;
    uint corr_d0=0u;
    uint corr_e0=0u;
    uint corr_f0=0u;
    uint corr_g0=0u;
    uint corr_h0=0u;
    uint corr_a1=0u;
    uint corr_b1=0u;
    uint corr_c1=0u;
    uint corr_d1=0u;
    uint corr_e1=0u;
    uint corr_f1=0u;
    uint corr_g1=0u;
    uint corr_h1=0u;
    uint corr_a2=0u;
    uint corr_b2=0u;
    uint corr_c2=0u;
    uint corr_d2=0u;
    uint corr_e2=0u;
    uint corr_f2=0u;
    uint corr_g2=0u;
    uint corr_h2=0u;
    uint corr_a3=0u;
    uint corr_b3=0u;
    uint corr_c3=0u;
    uint corr_d3=0u;
    uint corr_e3=0u;
    uint corr_f3=0u;
    uint corr_g3=0u;
    uint corr_h3=0u;
    //vectors (i.e., uint4) make clearer code, but empirically had a very slight performance cost

    uint4 temp_stillPackedY;
    uint4 temp_stillPackedX;
    uint temp_pa;

    for (uint i = 0; i < N_TIME_CHUNKS_LOCAL; i += LOCAL_SIZE){ //256 is a number of timesteps to do a local accum before saving to global memory
        uint pa=packed[i * NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES + addr_y ];
        uint la=(LOCAL_Y<<5| LOCAL_X<<2); //a short form for LOCAL_Y * 32 + LOCAL_X * 4 which may or may not be faster...

        barrier(CLK_LOCAL_MEM_FENCE);
        stillPackedY[la]    = ((pa & 0x000000f0) << 12u) | ((pa & 0x0000000f) >>  0u);
        stillPackedY[la+1u] = ((pa & 0x0000f000) <<  4u) | ((pa & 0x00000f00) >>  8u);
        stillPackedY[la+2u] = ((pa & 0x00f00000) >>  4u) | ((pa & 0x000f0000) >> 16u);
        stillPackedY[la+3u] = ((pa & 0xf0000000) >> 12u) | ((pa & 0x0f000000) >> 24u);
        //barrier(CLK_LOCAL_MEM_FENCE);//is removing this a bad idea???  things should be in lock-step...

        temp_pa=packed[i * NUM_ELEMENTS_div_4_x_NUM_FREQUENCIES + addr_x ];

        stillPackedX[la]    = ((temp_pa & 0x000000f0) << 12u) | ((temp_pa & 0x0000000f) >>  0u);
        stillPackedX[la+1u] = ((temp_pa & 0x0000f000) <<  4u) | ((temp_pa & 0x00000f00) >>  8u);
        stillPackedX[la+2u] = ((temp_pa & 0x00f00000) >>  4u) | ((temp_pa & 0x000f0000) >> 16u);
        stillPackedX[la+3u] = ((temp_pa & 0xf0000000) >> 12u) | ((temp_pa & 0x0f000000) >> 24u);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint j=0; j< LOCAL_SIZE; j++){
            temp_stillPackedY = vload4(mad24(j, 8u, LOCAL_Y), stillPackedY);
            temp_stillPackedX = vload4(mad24(j, 8u, LOCAL_X), stillPackedX);

            pa = temp_stillPackedX.s0;

            temp_pa = (pa >> 16u)  & 0xf; //real
            corr_a0 = mad24(temp_pa, temp_stillPackedY.s0, corr_a0);
            corr_b0 = mad24(temp_pa, temp_stillPackedY.s1, corr_b0);
            corr_c0 = mad24(temp_pa, temp_stillPackedY.s2, corr_c0);
            corr_d0 = mad24(temp_pa, temp_stillPackedY.s3, corr_d0);

            temp_pa = pa          & 0xf; //imag
            corr_a1 = mad24(temp_pa, temp_stillPackedY.s0, corr_a1);
            corr_b1 = mad24(temp_pa, temp_stillPackedY.s1, corr_b1);
            corr_c1 = mad24(temp_pa, temp_stillPackedY.s2, corr_c1);
            corr_d1 = mad24(temp_pa, temp_stillPackedY.s3, corr_d1);

            pa = temp_stillPackedX.s1;
            temp_pa = (pa >> 16u) & 0xf; //real
            corr_a2 = mad24(temp_pa, temp_stillPackedY.s0, corr_a2);
            corr_b2 = mad24(temp_pa, temp_stillPackedY.s1, corr_b2);
            corr_c2 = mad24(temp_pa, temp_stillPackedY.s2, corr_c2);
            corr_d2 = mad24(temp_pa, temp_stillPackedY.s3, corr_d2);

            temp_pa = pa          & 0xf; //imag
            corr_a3 = mad24(temp_pa, temp_stillPackedY.s0, corr_a3);
            corr_b3 = mad24(temp_pa, temp_stillPackedY.s1, corr_b3);
            corr_c3 = mad24(temp_pa, temp_stillPackedY.s2, corr_c3);
            corr_d3 = mad24(temp_pa, temp_stillPackedY.s3, corr_d3);

            pa = temp_stillPackedX.s2;
            temp_pa = (pa >> 16u) & 0xf;
            corr_e0 = mad24(temp_pa, temp_stillPackedY.s0, corr_e0);
            corr_f0 = mad24(temp_pa, temp_stillPackedY.s1, corr_f0);
            corr_g0 = mad24(temp_pa, temp_stillPackedY.s2, corr_g0);
            corr_h0 = mad24(temp_pa, temp_stillPackedY.s3, corr_h0);

            temp_pa = pa          & 0xf; //imag
            corr_e1 = mad24(temp_pa, temp_stillPackedY.s0, corr_e1);
            corr_f1 = mad24(temp_pa, temp_stillPackedY.s1, corr_f1);
            corr_g1 = mad24(temp_pa, temp_stillPackedY.s2, corr_g1);
            corr_h1 = mad24(temp_pa, temp_stillPackedY.s3, corr_h1);

            pa = temp_stillPackedX.s3;
            temp_pa = (pa >> 16u) & 0xf;
            corr_e2 = mad24(temp_pa, temp_stillPackedY.s0, corr_e2);
            corr_f2 = mad24(temp_pa, temp_stillPackedY.s1, corr_f2);
            corr_g2 = mad24(temp_pa, temp_stillPackedY.s2, corr_g2);
            corr_h2 = mad24(temp_pa, temp_stillPackedY.s3, corr_h2);

            temp_pa = pa          & 0xf; //imag
            corr_e3 = mad24(temp_pa, temp_stillPackedY.s0, corr_e3);
            corr_f3 = mad24(temp_pa, temp_stillPackedY.s1, corr_f3);
            corr_g3 = mad24(temp_pa, temp_stillPackedY.s2, corr_g3);
            corr_h3 = mad24(temp_pa, temp_stillPackedY.s3, corr_h3);
        }
    }
    //output: 32 numbers--> 16 pairs of real/imag numbers
    //16 pairs * 8 (local_size(0)) * 8 (local_size(1)) = 1024
    uint addr_o = ((BLOCK_ID * 2048u) + (LOCAL_Y * 256u) + (LOCAL_X * 8u)) + (FREQUENCY_BAND * NUM_BLOCKS_x_2048);

    if (LOCAL_X == 0 && LOCAL_Y == 0){
        while(atomic_cmpxchg(&block_lock[FREQUENCY_BAND*NUM_BLOCKS + BLOCK_ID],0,1)); //wait until unlocked
    }
        barrier(CLK_GLOBAL_MEM_FENCE); //sync point for the group
        corr_buf[addr_o+0u]+=   (corr_a0 >> 16u) + (corr_a1 & 0xffff) ; //real value
        corr_buf[addr_o+1u]+=   (corr_a1 >> 16u) - (corr_a0 & 0xffff) ;
        corr_buf[addr_o+2u]+=   (corr_a2 >> 16u) + (corr_a3 & 0xffff) ;
        corr_buf[addr_o+3u]+=   (corr_a3 >> 16u) - (corr_a2 & 0xffff) ;
        corr_buf[addr_o+4u]+=   (corr_e0 >> 16u) + (corr_e1 & 0xffff) ;
        corr_buf[addr_o+5u]+=   (corr_e1 >> 16u) - (corr_e0 & 0xffff) ;
        corr_buf[addr_o+6u]+=   (corr_e2 >> 16u) + (corr_e3 & 0xffff) ;
        corr_buf[addr_o+7u]+=   (corr_e3 >> 16u) - (corr_e2 & 0xffff) ;

        corr_buf[addr_o+64u]+=  (corr_b0 >> 16u) + (corr_b1 & 0xffff) ;
        corr_buf[addr_o+65u]+=  (corr_b1 >> 16u) - (corr_b0 & 0xffff) ;
        corr_buf[addr_o+66u]+=  (corr_b2 >> 16u) + (corr_b3 & 0xffff) ;
        corr_buf[addr_o+67u]+=  (corr_b3 >> 16u) - (corr_b2 & 0xffff) ;
        corr_buf[addr_o+68u]+=  (corr_f0 >> 16u) + (corr_f1 & 0xffff) ;
        corr_buf[addr_o+69u]+=  (corr_f1 >> 16u) - (corr_f0 & 0xffff) ;
        corr_buf[addr_o+70u]+=  (corr_f2 >> 16u) + (corr_f3 & 0xffff) ;
        corr_buf[addr_o+71u]+=  (corr_f3 >> 16u) - (corr_f2 & 0xffff) ;

        corr_buf[addr_o+128u]+= (corr_c0 >> 16u) + (corr_c1 & 0xffff) ;
        corr_buf[addr_o+129u]+= (corr_c1 >> 16u) - (corr_c0 & 0xffff) ;
        corr_buf[addr_o+130u]+= (corr_c2 >> 16u) + (corr_c3 & 0xffff) ;
        corr_buf[addr_o+131u]+= (corr_c3 >> 16u) - (corr_c2 & 0xffff) ;
        corr_buf[addr_o+132u]+= (corr_g0 >> 16u) + (corr_g1 & 0xffff) ;
        corr_buf[addr_o+133u]+= (corr_g1 >> 16u) - (corr_g0 & 0xffff) ;
        corr_buf[addr_o+134u]+= (corr_g2 >> 16u) + (corr_g3 & 0xffff) ;
        corr_buf[addr_o+135u]+= (corr_g3 >> 16u) - (corr_g2 & 0xffff) ;

        corr_buf[addr_o+192u]+= (corr_d0 >> 16u) + (corr_d1 & 0xffff) ;
        corr_buf[addr_o+193u]+= (corr_d1 >> 16u) - (corr_d0 & 0xffff) ;
        corr_buf[addr_o+194u]+= (corr_d2 >> 16u) + (corr_d3 & 0xffff) ;
        corr_buf[addr_o+195u]+= (corr_d3 >> 16u) - (corr_d2 & 0xffff) ;
        corr_buf[addr_o+196u]+= (corr_h0 >> 16u) + (corr_h1 & 0xffff) ;
        corr_buf[addr_o+197u]+= (corr_h1 >> 16u) - (corr_h0 & 0xffff) ;
        corr_buf[addr_o+198u]+= (corr_h2 >> 16u) + (corr_h3 & 0xffff) ;
        corr_buf[addr_o+199u]+= (corr_h3 >> 16u) - (corr_h2 & 0xffff) ;
        barrier(CLK_GLOBAL_MEM_FENCE); //make sure everyone is done

    if (LOCAL_X == 0 && LOCAL_Y == 0)
        block_lock[FREQUENCY_BAND*NUM_BLOCKS + BLOCK_ID]=0;
}
