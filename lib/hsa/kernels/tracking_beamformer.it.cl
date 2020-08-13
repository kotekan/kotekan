//Consider: Pass all beam arrays through global. Then, break BEAM_NUM/4 and read 2 beam arrays into private 
// memory for the first two BEAM_NUM groups and 3 into the last two groups. So BEAM_NUM 1-2,3-4 get
//2 beam arrays in private and 5-7 and 8-10 get 3.
//Finally, break timesteps into groups of 16 and do 16 timestep calculations per WI to take advantage
//of the SIMD0-1 architecture. Read in 16 timestep element matricies into local memory and
//process them, 1-16 arrays, in a single WI.

//OPTIMAL FOR THIS KERNEL. TS_RED = 2, BM_RED = 1.

//Need to get rid of 19 vgprs to optimize.
/*
    #define LOCALSIZE 256
    #define FREQ_NUM 1
    #define PHASE_NUM 1
*/
//#define RE_IM   2
//#define WI_ELEM 4
//#define WI_VAL  2*4

//#define LOCAL_SIZE      LOCAL_WORK_ITEMS
//#define LOCAL_ELEM      (FEED_NUM/WI_ELEM) 256
//#define TOTAL_FREQ      FREQ_NUM 1
//#define TOTAL_BEAM      BEAM_NUM 10
//#define TOTAL_TIMESTEP  TIMESTEP
//#define TOTAL_POLAR     POLARIZATION 2
//#define TOTAL_ELEM      ELEM_NUM//never used, consider removing. FEED_NUM is more useful, use instead.
#define TS_RED 2
#define BM_RED 1
#define TS_BUNDLE       TS_RED
#define BM_BUNDLE       BM_RED

 __kernel void trackingbf(
                            __global uint* data,
                            __global float* beam,//Device cache is possibly small enough to hold phase array.
                            __global float* outputSum) {

    //2(Re/Im)*256(total elem/4)*1(freq)*1(beam)*2(polarization);
    //__local float sum[2*256*FREQ_NUM*PHASE_NUM*2];
    __local float sum_re[256];
    __local float sum_img[256];

    uint unpack;

    float data_val1_re;
    float data_val1_img;
    float data_val2_re;
    float data_val2_img;
    float data_val3_re;
    float data_val3_img;
    float data_val4_re;
    float data_val4_img;


    float b1_re[BM_BUNDLE];
    float b1_im[BM_BUNDLE];
    float b2_re[BM_BUNDLE];
    float b2_im[BM_BUNDLE];
    float b3_re[BM_BUNDLE];
    float b3_im[BM_BUNDLE];
    float b4_re[BM_BUNDLE];
    float b4_im[BM_BUNDLE];

    sum_re[get_local_id(0)] = 0.0;
    sum_img[get_local_id(0)] = 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);

    //Index is (total elem/4)*2(re/im)*4(each element) + freq*256(total elem/4)*4*2(re/im)
    // + beam_num*256(total elem/4)*4*2(re/im)*freq
    // + 1(beam_num)*1(freq)*256(total elem/4)*4*2(re/im)*polarization

    int x;

    for (x = 0; x<BM_BUNDLE; x++){

        b1_re[x] = beam[(get_local_id(0)*8
                             + get_group_id(0)*2048
                             + get_local_id(1)*4096
                             + (BM_BUNDLE*get_group_id(1) + x)*4096)+0];
        b1_im[x] = beam[(get_local_id(0)*8
                             + get_group_id(0)*2048
                             + get_local_id(1)*4096
                             + (BM_BUNDLE*get_group_id(1) + x)*4096)+1];
        b2_re[x] = beam[(get_local_id(0)*8
                             + get_group_id(0)*2048
                             + get_local_id(1)*4096
                             + (BM_BUNDLE*get_group_id(1) + x)*4096)+2];
        b2_im[x] = beam[(get_local_id(0)*8
                             + get_group_id(0)*2048
                             + get_local_id(1)*4096
                             + (BM_BUNDLE*get_group_id(1) + x)*4096)+3];
        b3_re[x] = beam[(get_local_id(0)*8
                             + get_group_id(0)*2048
                             + get_local_id(1)*4096
                             + (BM_BUNDLE*get_group_id(1) + x)*4096)+4];
        b3_im[x] = beam[(get_local_id(0)*8
                             + get_group_id(0)*2048
                             + get_local_id(1)*4096
                             + (BM_BUNDLE*get_group_id(1) + x)*4096)+5];
        b4_re[x] = beam[(get_local_id(0)*8
                             + get_group_id(0)*2048
                             + get_local_id(1)*4096
                             + (BM_BUNDLE*get_group_id(1) + x)*4096)+6];
        b4_im[x] = beam[(get_local_id(0)*8
                             + get_group_id(0)*2048
                             + get_local_id(1)*4096
                             + (BM_BUNDLE*get_group_id(1) + x)*4096)+7];
    }

    for (int j = 0; j < TS_BUNDLE; j++){

            unpack = data[(TS_BUNDLE*get_group_id(2) + j)*512
                            + get_local_id(1)*512 
                            + get_group_id(0)*256
                            + get_local_id(0)]; //should be a coalesced load with local work group

        for (x = 0; x<BM_BUNDLE;x++){
            //we load 4 Byte words, addresses are based on that size
            //index: timestep*polarization*freqNum*elemNum/4

            sum_re[get_local_id(0)]
                    = (((float)((unpack & 0x000000f0) >> 4u) - 8) * b1_re[x]
                             - ((float)((unpack & 0x0000000f) >>  0u) - 8)*b1_im[x])
                        + (((float)((unpack & 0x0000f000) >>  12u) - 8) * b2_re[x]
                             - ((float)((unpack & 0x00000f00) >>  8u) - 8)*b2_im[x])
                        + (((float)((unpack & 0x00f00000) >>  20u) - 8) * b3_re[x]
                             - ((float)((unpack & 0x000f0000) >> 16u) - 8)*b3_im[x])
                        + (((float)((unpack & 0xf0000000) >> 28u) - 8) * b4_re[x]
                             - ((float)((unpack & 0x0f000000) >> 24u) - 8)*b4_im[x]);

            sum_img[get_local_id(0)]
                    = (((float)((unpack & 0x000000f0) >> 4u) - 8) * b1_im[x]
                             + ((float)((unpack & 0x0000000f) >>  0u) - 8)*b1_re[x])
                        + (((float)((unpack & 0x0000f000) >>  12u) - 8) * b2_im[x]
                             + ((float)((unpack & 0x00000f00) >>  8u) - 8)*b2_re[x])
                        + (((float)((unpack & 0x00f00000) >>  20u) - 8) * b3_im[x]
                             + ((float)((unpack & 0x000f0000) >> 16u) - 8)*b3_re[x])
                        + (((float)((unpack & 0xf0000000) >> 28u) - 8) * b4_im[x]
                             + ((float)((unpack & 0x0f000000) >> 24u) - 8)*b4_re[x]);

            barrier(CLK_LOCAL_MEM_FENCE);

            if (get_local_id(0) < 64){

                sum_re[get_local_id(0)] = sum_re[4*get_local_id(0) + 0]
                                 + sum_re[4*get_local_id(0) + 1]
                                 + sum_re[4*get_local_id(0) + 2]
                                 + sum_re[4*get_local_id(0) + 3];
                sum_img[get_local_id(0)] = sum_img[4*get_local_id(0) + 0]
                                 + sum_img[4*get_local_id(0) + 1]
                                 + sum_img[4*get_local_id(0) + 2]
                                 + sum_img[4*get_local_id(0) + 3];
            }
            if (get_local_id(0) < 16){

                        sum_re[get_local_id(0)] = sum_re[4*get_local_id(0) + 0]
                                         + sum_re[4*get_local_id(0) + 1]
                                         + sum_re[4*get_local_id(0) + 2]
                                         + sum_re[4*get_local_id(0) + 3];
                        sum_img[get_local_id(0)] = sum_img[4*get_local_id(0) + 0]
                                         + sum_img[4*get_local_id(0) + 1]
                                         + sum_img[4*get_local_id(0) + 2]
                                         + sum_img[4*get_local_id(0) + 3];

            }
            if (get_local_id(0) < 4){

                sum_re[get_local_id(0)] = sum_re[4*get_local_id(0) + 0]
                                 + sum_re[4*get_local_id(0) + 1]
                                 + sum_re[4*get_local_id(0) + 2]
                                 + sum_re[4*get_local_id(0) + 3];
                sum_img[get_local_id(0)] = sum_img[4*get_local_id(0) + 0]
                                 + sum_img[4*get_local_id(0) + 1]
                                 + sum_img[4*get_local_id(0) + 2]
                                 + sum_img[4*get_local_id(0) + 3];
            }

            if (get_local_id(0) == 0){

                float val_re = sum_re[4*get_local_id(0) + 0]
                                 + sum_re[4*get_local_id(0) + 1]
                                 + sum_re[4*get_local_id(0) + 2]
                                 + sum_re[4*get_local_id(0) + 3];
                float val_img = sum_img[4*get_local_id(0) + 0]
                                 + sum_img[4*get_local_id(0) + 1]
                                 + sum_img[4*get_local_id(0) + 2]
                                 + sum_img[4*get_local_id(0) + 3];

                outputSum[2*get_group_id(0)
                         + 4*get_local_id(1)
                         + 4*(BM_BUNDLE*get_group_id(1) + x)
                         + 40*(TS_BUNDLE*get_group_id(2) + j)
                         + 0] = val_re;
                outputSum[2*get_group_id(0)
                         + 4*get_local_id(1)
                         + 4*(BM_BUNDLE*get_group_id(1) + x)
                         + 40*(TS_BUNDLE*get_group_id(2) + j)
                         + 1] = val_img;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

//Data updates at 2.56microsec
//vec updates at 1sec
//vec should be [256,16,M,inf/100]; M-number of phases done at once.
//Output is one value per freq per phase per timestep per polarization. Max 16 freq and likely 1-10 phases.









